"""Frame extractor for extracting video frames at specific timestamps."""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

from .config import Config, default_config


class FrameExtractor:
    """Extract frames from videos at specific timestamps."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config

    def extract_frame(self, video_path: Path, timestamp_ms: int) -> Image.Image:
        """
        Extract a single frame from a video at a specific timestamp.

        Args:
            video_path: Path to the video file
            timestamp_ms: Timestamp in milliseconds

        Returns:
            PIL Image of the extracted frame
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            # Set position to timestamp
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

            # Read the frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError(
                    f"Cannot read frame at timestamp {timestamp_ms}ms from {video_path}"
                )

            # Convert BGR to RGB and to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        finally:
            cap.release()

    def extract_frames_with_context(
        self,
        video_path: Path,
        timestamp_ms: int,
        n_frames: int = 5,
        context_mode: str = "uniform",
        key_moment_threshold_sec: float = 10.0,
        context_seconds: float | None = None,
    ) -> List[Tuple[int, Image.Image]]:
        """
        Extract multiple frames for temporal context.

        Selects from ``[start, timestamp_ms]``.  When *context_seconds* is
        ``None``, start is 0 (full video watched so far).  When set, start
        is ``timestamp_ms - context_seconds*1000`` (trailing window).

        Args:
            video_path: Path to the video file
            timestamp_ms: Target timestamp in milliseconds
            n_frames: Number of frames to extract (caller computes from duration)
            context_mode: ``"uniform"``, ``"key_moments"``, or ``"auto"``
            key_moment_threshold_sec: Auto mode switches to key moments when
                the video watched so far exceeds this duration (seconds).
            context_seconds: If set, use a trailing window of this many seconds
                instead of the full video from start.

        Returns:
            List of (timestamp_ms, PIL Image) tuples, ordered from oldest to newest
        """

        if context_seconds is not None:
            start_ms = max(0, timestamp_ms - int(context_seconds * 1000))
        else:
            # Full video watched so far: [0, timestamp_ms]
            start_ms = 0

        # Decide whether to use key-moment selection
        use_key_moments = (
            context_mode == "key_moments"
            or (
                context_mode == "auto"
                and timestamp_ms > key_moment_threshold_sec * 1000
            )
        )

        if use_key_moments and n_frames > 1:
            from .change_detection import get_or_create_profile

            profile = get_or_create_profile(video_path)
            timestamps = profile.select_key_frames(start_ms, timestamp_ms, n_frames)
        elif n_frames == 1:
            timestamps = [timestamp_ms]
        else:
            timestamps = np.linspace(start_ms, timestamp_ms, n_frames, dtype=int)
            timestamps = timestamps.tolist()

        # Extract frames at each timestamp
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            frames = []
            for ts in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((int(ts), Image.fromarray(frame_rgb)))
                else:
                    print(f"Warning: Cannot read frame at {ts}ms")

            return frames
        finally:
            cap.release()

    def save_frames(
        self,
        frames: List[Tuple[int, Image.Image]],
        video_id: str,
        target_timestamp_ms: int,
    ) -> List[Path]:
        """
        Save extracted frames to disk.

        Args:
            frames: List of (timestamp_ms, PIL Image) tuples
            video_id: Video ID
            target_timestamp_ms: Target timestamp for organizing frames

        Returns:
            List of paths to saved frames
        """
        frame_dir = self.config.get_frame_dir(video_id, target_timestamp_ms)
        frame_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, (ts, frame) in enumerate(frames):
            frame_path = frame_dir / f"frame_{i}.{self.config.frame_format}"
            frame.save(frame_path, quality=self.config.frame_quality)
            saved_paths.append(frame_path)

        return saved_paths

    def load_frames(
        self, video_id: str, timestamp_ms: int
    ) -> List[Tuple[int, Image.Image]]:
        """
        Load previously saved frames from disk.

        Args:
            video_id: Video ID
            timestamp_ms: Target timestamp

        Returns:
            List of (frame_index, PIL Image) tuples
        """
        frame_dir = self.config.get_frame_dir(video_id, timestamp_ms)
        if not frame_dir.exists():
            raise ValueError(f"No frames found at {frame_dir}")

        frames = []
        frame_files = sorted(frame_dir.glob(f"frame_*.{self.config.frame_format}"))

        for frame_path in frame_files:
            # Extract frame index from filename
            idx = int(frame_path.stem.split("_")[1])
            frame = Image.open(frame_path)
            frames.append((idx, frame))

        return frames

    def frames_exist(self, video_id: str, timestamp_ms: int) -> bool:
        """Check if frames have already been extracted for this timestamp."""
        frame_dir = self.config.get_frame_dir(video_id, timestamp_ms)
        if not frame_dir.exists():
            return False

        frame_files = list(frame_dir.glob(f"frame_*.{self.config.frame_format}"))
        return len(frame_files) > 0

    def extract_and_save(
        self,
        video_id: str,
        timestamp_ms: int,
        overwrite: bool = False,
    ) -> List[Path]:
        """
        Extract frames with context and save to disk.

        Args:
            video_id: Video ID
            timestamp_ms: Target timestamp in milliseconds
            overwrite: Whether to overwrite existing frames

        Returns:
            List of paths to saved frames
        """
        # Check if frames already exist
        if not overwrite and self.frames_exist(video_id, timestamp_ms):
            frame_dir = self.config.get_frame_dir(video_id, timestamp_ms)
            return sorted(frame_dir.glob(f"frame_*.{self.config.frame_format}"))

        # Extract frames
        video_path = self.config.get_video_path(video_id)
        frames = self.extract_frames_with_context(video_path, timestamp_ms)

        # Save to disk
        if self.config.save_frames_to_disk:
            return self.save_frames(frames, video_id, timestamp_ms)
        else:
            return []

    def extract_all_for_video(
        self,
        video_id: str,
        timestamps_ms: List[int],
        overwrite: bool = False,
        progress_callback=None,
    ) -> dict:
        """
        Extract frames for all timestamps in a video.

        Args:
            video_id: Video ID
            timestamps_ms: List of timestamps
            overwrite: Whether to overwrite existing frames
            progress_callback: Optional callback function(current, total)

        Returns:
            Dict mapping timestamp_ms to list of frame paths
        """
        results = {}
        total = len(timestamps_ms)

        for i, ts in enumerate(timestamps_ms):
            try:
                paths = self.extract_and_save(video_id, ts, overwrite)
                results[ts] = paths
            except Exception as e:
                print(f"Error extracting frames for {video_id} at {ts}ms: {e}")
                results[ts] = []

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def get_frames_as_pil(
        self, video_id: str, timestamp_ms: int
    ) -> List[Image.Image]:
        """
        Get frames as PIL Images, either from disk or by extracting.

        Args:
            video_id: Video ID
            timestamp_ms: Target timestamp

        Returns:
            List of PIL Images (oldest to newest)
        """
        if self.frames_exist(video_id, timestamp_ms):
            # Load from disk
            frame_data = self.load_frames(video_id, timestamp_ms)
            return [frame for _, frame in frame_data]
        else:
            # Extract fresh
            video_path = self.config.get_video_path(video_id)
            frame_data = self.extract_frames_with_context(video_path, timestamp_ms)
            return [frame for _, frame in frame_data]
