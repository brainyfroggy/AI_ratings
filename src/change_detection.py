"""Key-moment context frame selection via histogram change detection.

Precomputes a visual change signal for a video, then selects frames at
perceptual transition points (scene changes, expression shifts) rather
than uniform temporal spacing.  For short clips the fallback is uniform
sampling, matching the existing behaviour.

Cognitive rationale: Event Segmentation Theory (Zacks & Swallow 2007)
shows that human episodic memory is organized around transition
boundaries, not uniform temporal samples.
"""

import threading
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .sampling import get_video_properties


class VideoChangeProfile:
    """Precomputed visual change signal for a single video.

    Build once with :meth:`from_video`, then call
    :meth:`select_key_frames` for each timestamp — O(1) per query.
    """

    def __init__(
        self,
        probe_timestamps: np.ndarray,
        change_scores: np.ndarray,
    ):
        self.probe_timestamps = probe_timestamps  # ms, shape (N,)
        self.change_scores = change_scores        # shape (N-1,)

    @classmethod
    def from_video(
        cls,
        video_path: Path,
        probe_interval_ms: int = 100,
    ) -> "VideoChangeProfile":
        """Probe every *probe_interval_ms* and compute histogram distances.

        Args:
            video_path: Path to the video file.
            probe_interval_ms: Interval between probes in milliseconds.

        Returns:
            A populated VideoChangeProfile.
        """
        _, duration_sec, _ = get_video_properties(video_path)
        duration_ms = int(duration_sec * 1000)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        try:
            probe_times: List[int] = []
            histograms: List[np.ndarray] = []

            t = 0
            while t <= duration_ms:
                cap.set(cv2.CAP_PROP_POS_MSEC, t)
                ret, frame = cap.read()
                if not ret:
                    break
                # HSV preserves color information important for emotion
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # 3D histogram: 16 hue × 16 saturation × 16 value bins
                hist = cv2.calcHist(
                    [hsv], [0, 1, 2], None,
                    [16, 16, 16], [0, 180, 0, 256, 0, 256],
                )
                cv2.normalize(hist, hist)
                probe_times.append(t)
                histograms.append(hist)
                t += probe_interval_ms
        finally:
            cap.release()

        probe_timestamps = np.array(probe_times, dtype=np.int64)

        if len(histograms) < 2:
            return cls(probe_timestamps, np.array([], dtype=np.float64))

        # Flatten 3D histograms for comparison
        scores = np.array([
            cv2.compareHist(
                histograms[i - 1].flatten(),
                histograms[i].flatten(),
                cv2.HISTCMP_CHISQR,
            )
            for i in range(1, len(histograms))
        ])

        return cls(probe_timestamps, scores)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def select_key_frames(
        self,
        start_ms: int,
        end_ms: int,
        n_frames: int,
    ) -> List[int]:
        """Select *n_frames* timestamps that capture the most salient moments.

        Args:
            start_ms: Start of the context window (inclusive).
            end_ms: End of the context window — the current rating point.
            n_frames: Number of frames to return.

        Returns:
            Sorted list of *n_frames* timestamps in ms.
        """
        if n_frames <= 0:
            return []
        if n_frames == 1:
            return [int(end_ms)]

        # Slots available for change-point selection (last slot = end_ms)
        n_select = n_frames - 1

        # Slice the precomputed signal to the query window.
        # change_scores[i] corresponds to the transition *from*
        # probe_timestamps[i] *to* probe_timestamps[i+1].
        score_times = self.probe_timestamps[1:]  # timestamp of the "after" frame
        mask = (score_times >= start_ms) & (score_times <= end_ms)
        window_times = score_times[mask]
        window_scores = self.change_scores[mask[:len(self.change_scores)]]

        if len(window_scores) == 0:
            # No change data — fall back to uniform
            return self._uniform_fallback(start_ms, end_ms, n_frames)

        # Adaptive threshold: median + 2 * MAD
        median = np.median(window_scores)
        mad = np.median(np.abs(window_scores - median))
        threshold = median + 2.0 * mad

        # Find local maxima above threshold
        peak_indices = self._find_peaks(window_scores, threshold)

        if len(peak_indices) == 0:
            # No peaks above threshold — fall back to uniform
            return self._uniform_fallback(start_ms, end_ms, n_frames)

        # Greedy top-K with minimum separation
        context_span = end_ms - start_ms
        min_sep = context_span / (n_select * 2) if n_select > 0 else 0

        # Sort peaks by descending score
        peak_scores = window_scores[peak_indices]
        order = np.argsort(-peak_scores)
        sorted_peak_indices = peak_indices[order]

        selected_times: List[int] = []
        for idx in sorted_peak_indices:
            t = int(window_times[idx])
            if all(abs(t - s) >= min_sep for s in selected_times):
                selected_times.append(t)
                if len(selected_times) >= n_select:
                    break

        # Always include end_ms as the last frame
        selected_times.append(int(end_ms))
        selected_times = sorted(set(selected_times))

        # Fill any remaining slots with uniform spacing until we hit n_frames
        if len(selected_times) < n_frames:
            # Generate a dense pool of candidates and pick those farthest
            # from existing selections to maximise coverage
            candidates = np.linspace(start_ms, end_ms, n_frames * 4, dtype=int)
            existing = set(selected_times)
            candidates = [int(c) for c in candidates if c not in existing]
            for c in candidates:
                if len(selected_times) >= n_frames:
                    break
                selected_times.append(c)
                selected_times = sorted(set(selected_times))

        # Trim to exactly n_frames (keep end_ms and the highest-scored)
        if len(selected_times) > n_frames:
            selected_times = selected_times[-n_frames:]

        return selected_times

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_peaks(scores: np.ndarray, threshold: float) -> np.ndarray:
        """Find indices of local maxima above *threshold*."""
        peaks = []
        for i in range(len(scores)):
            if scores[i] < threshold:
                continue
            left_ok = (i == 0) or (scores[i] >= scores[i - 1])
            right_ok = (i == len(scores) - 1) or (scores[i] >= scores[i + 1])
            if left_ok and right_ok:
                peaks.append(i)
        return np.array(peaks, dtype=int)

    @staticmethod
    def _uniform_fallback(start_ms: int, end_ms: int, n_frames: int) -> List[int]:
        """Return uniformly spaced timestamps (existing behaviour)."""
        return np.linspace(start_ms, end_ms, n_frames, dtype=int).tolist()


# ── Module-level cache ────────────────────────────────────────────────

_profile_cache: Dict[Path, VideoChangeProfile] = {}
_cache_lock = threading.Lock()
# Per-video build locks so that when many workers start on the same fresh
# video at once, exactly ONE builds the (whole-video-scanning) profile while
# the rest wait — instead of all of them redundantly decoding the file. The
# slow build is held under the per-video lock, NOT the global _cache_lock, so
# lookups for other videos are never blocked.
_build_locks: Dict[Path, threading.Lock] = {}
_build_locks_guard = threading.Lock()


def get_or_create_profile(
    video_path: Path,
    probe_interval_ms: int = 100,
) -> VideoChangeProfile:
    """Thread-safe accessor for the per-video change profile cache.

    Builds each video's profile at most once even under heavy concurrency:
    concurrent first-callers serialize on a per-video build lock and the
    losers read the just-built profile from cache (double-checked).
    """
    key = Path(video_path).resolve()
    with _cache_lock:
        if key in _profile_cache:
            return _profile_cache[key]

    # Obtain (or create) the per-video build lock.
    with _build_locks_guard:
        build_lock = _build_locks.setdefault(key, threading.Lock())

    with build_lock:
        # Double-check: another worker may have built it while we waited.
        with _cache_lock:
            if key in _profile_cache:
                return _profile_cache[key]
        # Only ONE worker reaches here per video; the build is slow but held
        # under the per-video lock, not the global cache lock.
        profile = VideoChangeProfile.from_video(video_path, probe_interval_ms)
        with _cache_lock:
            _profile_cache.setdefault(key, profile)
            return _profile_cache[key]


def clear_cache() -> None:
    """Free all cached profiles (call after finishing a video)."""
    with _cache_lock:
        _profile_cache.clear()
    with _build_locks_guard:
        _build_locks.clear()
