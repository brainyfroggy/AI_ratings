"""Adaptive frame sampling interval computation.

Derives a per-video sampling interval from three constraints:
1. target_interval  — Nyquist optimum from human data (0.5s)
2. frame_floor      — hardware floor to skip identical frames
3. duration_ceil    — ensure enough samples for short clips

See doc/frame_sampling_interval_analysis.md for the full derivation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


# --- Default parameters (derived from empirical analysis) ---
DEFAULT_TARGET_INTERVAL = 0.5   # seconds; Nyquist on reaction-time-corrected signal
DEFAULT_MIN_FRAME_SKIP = 3      # minimum frames between samples
DEFAULT_MIN_SAMPLES = 5         # soft target for rating points per clip


def compute_sample_interval(
    fps: float,
    duration_sec: float,
    target_interval: float = DEFAULT_TARGET_INTERVAL,
    min_frame_skip: int = DEFAULT_MIN_FRAME_SKIP,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> float:
    """Compute the adaptive sampling interval for a single video.

    Args:
        fps: Video frame rate.
        duration_sec: Video duration in seconds.
        target_interval: Ideal interval from human data analysis (default 0.5s).
        min_frame_skip: Minimum frames between samples to avoid identical frames.
        min_samples: Soft target for number of rating points per clip.

    Returns:
        Sampling interval in seconds.
    """
    if fps <= 0 or duration_sec <= 0:
        return target_interval

    frame_floor = min_frame_skip / fps           # hardware constraint
    duration_ceil = duration_sec / min_samples   # sample-count constraint

    return max(frame_floor, min(target_interval, duration_ceil))


def get_video_properties(video_path: Path) -> Tuple[float, float, int]:
    """Read fps, duration, and frame count from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Tuple of (fps, duration_sec, frame_count).

    Raises:
        IOError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = frame_count / fps if fps > 0 else 0.0
    return fps, duration_sec, frame_count


def compute_sample_timestamps(
    video_path: Path,
    target_interval: float = DEFAULT_TARGET_INTERVAL,
    min_frame_skip: int = DEFAULT_MIN_FRAME_SKIP,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Tuple[np.ndarray, float]:
    """Compute adaptive sample timestamps for a video.

    Args:
        video_path: Path to the video file.
        target_interval: Ideal interval from human data analysis.
        min_frame_skip: Minimum frames between samples.
        min_samples: Soft target for number of rating points.

    Returns:
        Tuple of (timestamps_ms array, interval_sec used).
    """
    fps, duration_sec, _ = get_video_properties(video_path)

    interval_sec = compute_sample_interval(
        fps, duration_sec, target_interval, min_frame_skip, min_samples
    )

    interval_ms = int(interval_sec * 1000)
    duration_ms = int(duration_sec * 1000)

    timestamps = np.arange(interval_ms, duration_ms, interval_ms)
    return timestamps, interval_sec


# --- Duration threshold for switching strategies ---
DURATION_THRESHOLD_SEC = 10.0  # clips >= this use fixed 2s like 18V pipeline
FIXED_LONG_INTERVAL_MS = 2000  # fixed 2s interval for longer clips


def compute_pass2_timestamps(
    pass1_df: "pd.DataFrame",
    video_duration_ms: float,
    dense_interval_ms: int = 500,
    change_threshold_frac: float = 0.20,
    window_s: float = 5.0,
) -> np.ndarray:
    """Compute Pass 2 timestamps from Pass 1 ratings.

    Identifies high-change windows in the (valence, arousal) trajectory from
    Pass 1 and returns new timestamps for dense re-scanning within those
    windows, excluding timestamps already rated in Pass 1.

    Args:
        pass1_df: DataFrame with columns [timestamp_ms, valence, arousal],
                  one video, successful ratings only.
        video_duration_ms: Total video duration in milliseconds.
        dense_interval_ms: Pass 2 sampling interval in ms (default 500).
        change_threshold_frac: Fraction of observed VA range that defines a
                               high-change step (default 0.20).
        window_s: Half-width in seconds of window around each detected
                  transition (default 5.0).

    Returns:
        Sorted numpy int array of new timestamp_ms values for Pass 2.
        Empty array if no high-change windows are detected.
    """
    df = (pass1_df[["timestamp_ms", "valence", "arousal"]]
          .dropna()
          .sort_values("timestamp_ms")
          .reset_index(drop=True))

    if len(df) < 2:
        return np.array([], dtype=int)

    v = df["valence"].to_numpy(dtype=float)
    a = df["arousal"].to_numpy(dtype=float)
    ts = df["timestamp_ms"].to_numpy(dtype=int)

    v_range = v.max() - v.min()
    a_range = a.max() - a.min()
    obs_range = max(v_range, a_range)
    if obs_range < 1e-6:
        return np.array([], dtype=int)

    threshold = change_threshold_frac * obs_range
    change = np.sqrt(np.diff(v) ** 2 + np.diff(a) ** 2)
    window_ms = int(window_s * 1000)
    dur_ms = int(video_duration_ms)

    # Collect intervals centered on midpoint of each high-change step
    intervals: List[Tuple[int, int]] = []
    for i, ch in enumerate(change):
        if ch >= threshold:
            mid = int((int(ts[i]) + int(ts[i + 1])) / 2)
            intervals.append((max(0, mid - window_ms), min(dur_ms, mid + window_ms)))

    if not intervals:
        return np.array([], dtype=int)

    # Merge overlapping intervals
    intervals.sort()
    merged: List[Tuple[int, int]] = [intervals[0]]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    # Generate dense candidates; exclude timestamps already in Pass 1
    tol = dense_interval_ms // 2
    cands: List[int] = []
    for lo, hi in merged:
        for t in range(lo + dense_interval_ms, hi + 1, dense_interval_ms):
            if t > dur_ms:
                break
            idx = int(np.searchsorted(ts, t))
            near = False
            for j in (idx - 1, idx):
                if 0 <= j < len(ts) and abs(int(ts[j]) - t) <= tol:
                    near = True
                    break
            if not near:
                cands.append(t)

    return np.array(sorted(set(cands)), dtype=int)


def compute_sample_timestamps_dual(
    video_path: Path,
    duration_threshold: float = DURATION_THRESHOLD_SEC,
    fixed_interval_ms: int = FIXED_LONG_INTERVAL_MS,
    target_interval: float = DEFAULT_TARGET_INTERVAL,
    min_frame_skip: int = DEFAULT_MIN_FRAME_SKIP,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Tuple[np.ndarray, float, str]:
    """Compute sample timestamps using duration-dependent strategy.

    Short clips (<duration_threshold): adaptive sampling (0.5s target)
        with uniform context.
    Long clips (>=duration_threshold): fixed 2s interval
        with full context + key-moment frame selection.

    Args:
        video_path: Path to the video file.
        duration_threshold: Duration cutoff in seconds (default 10s).
        fixed_interval_ms: Fixed interval for long clips in ms (default 2000).
        target_interval: Adaptive target interval for short clips.
        min_frame_skip: Minimum frames between samples (adaptive mode).
        min_samples: Soft target for rating points (adaptive mode).

    Returns:
        Tuple of (timestamps_ms array, interval_sec used, strategy name).
        strategy is "adaptive" for short clips, "fixed" for long clips.
    """
    fps, duration_sec, _ = get_video_properties(video_path)
    duration_ms = int(duration_sec * 1000)

    if duration_sec >= duration_threshold:
        # Long clip: fixed 2s interval (same as 18V pipeline)
        timestamps = np.arange(fixed_interval_ms, duration_ms, fixed_interval_ms)
        return timestamps, fixed_interval_ms / 1000.0, "fixed"
    else:
        # Short clip: adaptive sampling
        interval_sec = compute_sample_interval(
            fps, duration_sec, target_interval, min_frame_skip, min_samples
        )
        interval_ms = int(interval_sec * 1000)
        timestamps = np.arange(interval_ms, duration_ms, interval_ms)
        return timestamps, interval_sec, "adaptive"


# --- Budget-based duration-adaptive policy (cross-dataset) ---
DEFAULT_GLOBAL_MIN_INTERVAL = 0.2   # seconds; preferred default (CASE 0.2s study)
DEFAULT_MAX_INTERVAL = 2.0          # seconds; ceiling for very long videos
DEFAULT_CALL_BUDGET = 900           # target samples per stimulus for long videos


def compute_sample_timestamps_budget(
    video_path: Path,
    global_min: float = DEFAULT_GLOBAL_MIN_INTERVAL,
    max_interval: float = DEFAULT_MAX_INTERVAL,
    call_budget: int = DEFAULT_CALL_BUDGET,
    min_frame_skip: int = DEFAULT_MIN_FRAME_SKIP,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Tuple[np.ndarray, float, str]:
    """Duration-adaptive sampling under a fixed per-stimulus call budget.

    Implements the cross-dataset interval policy (report
    \\S sec:interval-policy):

        interval = max( min_frame_skip / fps,
                        min( duration / min_samples,
                             clip( duration / call_budget,
                                   global_min, max_interval ) ) )

    Short clips sample densely at ``global_min`` (e.g.\\ 0.2s); clips long
    enough that ``duration/call_budget`` exceeds ``global_min`` stretch the
    interval so each receives ~``call_budget`` samples, never sparser than
    ``max_interval``. The ``min_samples`` term guarantees >= that many points
    on sub-second clips; the ``min_frame_skip`` term forbids sampling closer
    than that many frames apart.

    For any clip with ``duration <= global_min * call_budget`` (e.g.
    <= 180s at the 0.2s / 900 defaults) the budget term collapses to
    ``global_min`` and there is no coarsening.

    Returns:
        Tuple of (timestamps_ms array, interval_sec used, regime), where
        regime is "dense" (budget term at the global_min floor),
        "budget" (interval == duration/call_budget), or
        "cap" (interval at the max_interval ceiling).
    """
    fps, duration_sec, _ = get_video_properties(video_path)
    duration_ms = int(duration_sec * 1000)

    if fps <= 0 or duration_sec <= 0:
        return np.array([], dtype=int), global_min, "dense"

    budget_iv = duration_sec / call_budget if call_budget > 0 else global_min
    clipped = min(max(budget_iv, global_min), max_interval)
    if clipped <= global_min + 1e-9:
        regime = "dense"
    elif clipped >= max_interval - 1e-9:
        regime = "cap"
    else:
        regime = "budget"

    frame_floor = min_frame_skip / fps           # never closer than N frames
    sample_ceil = duration_sec / min_samples      # ensure >= min_samples points
    interval_sec = max(frame_floor, min(sample_ceil, clipped))

    interval_ms = max(1, int(interval_sec * 1000))
    timestamps = np.arange(interval_ms, duration_ms, interval_ms)
    return timestamps, interval_sec, regime
