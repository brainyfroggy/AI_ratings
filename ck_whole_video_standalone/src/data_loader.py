"""Data loader for parsing human emotion ratings."""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, default_config


@dataclass
class RatingData:
    """Container for rating data from a single subject."""

    worker_id: str
    video_id: str
    rating_type: str  # "AROUSAL" or "VALENCE"
    timestamps_ms: np.ndarray
    ratings: np.ndarray


class DataLoader:
    """Load and parse human emotion ratings."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self._file_cache: Dict[str, List[Path]] = {}

    def _get_rating_files(self, video_id: str, rating_type: str) -> List[Path]:
        """Get all rating files for a video and rating type."""
        cache_key = f"{rating_type}_{video_id}"
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        pattern = f"{rating_type}_*_{video_id}.txt"
        files = list(self.config.ratings_dir.glob(pattern))
        self._file_cache[cache_key] = files
        return files

    def _parse_rating_file(self, file_path: Path) -> RatingData:
        """Parse a single rating file."""
        # Extract metadata from filename: AROUSAL_WORKERID_VIDEOID.txt
        filename = file_path.stem
        parts = filename.split("_")
        rating_type = parts[0]
        worker_id = parts[1]
        video_id = parts[2]

        # Read and parse the file content
        with open(file_path, "r") as f:
            content = f.read().strip()

        # Parse timestamp,rating pairs separated by |
        timestamps = []
        ratings = []

        for pair in content.split("|"):
            if "," in pair:
                ts_str, rating_str = pair.split(",")
                ts = int(ts_str)
                rating = int(rating_str)

                # Skip end marker (-1 rating)
                if rating >= 0:
                    timestamps.append(ts)
                    ratings.append(rating)

        return RatingData(
            worker_id=worker_id,
            video_id=video_id,
            rating_type=rating_type,
            timestamps_ms=np.array(timestamps),
            ratings=np.array(ratings),
        )

    def load_human_ratings(self, video_id: str) -> Dict[str, List[RatingData]]:
        """
        Load all human ratings for a video.

        Returns:
            Dict with keys 'arousal' and 'valence', each containing list of RatingData
        """
        result = {"arousal": [], "valence": []}

        for rating_type in ["AROUSAL", "VALENCE"]:
            files = self._get_rating_files(video_id, rating_type)
            for file_path in files:
                try:
                    rating_data = self._parse_rating_file(file_path)
                    result[rating_type.lower()].append(rating_data)
                except Exception as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")

        return result

    def get_rating_timestamps(
        self, video_id: str, sample_interval_ms: Optional[int] = None
    ) -> np.ndarray:
        """
        Get sorted list of timestamps for rating a video.

        If sample_interval_ms is provided, returns evenly spaced timestamps.
        If config.use_adaptive_sampling is True (and sample_interval_ms is None),
        computes a per-video adaptive interval.
        Otherwise, returns union of all subjects' rating change points.
        """
        if sample_interval_ms is not None:
            # Get video duration and create evenly spaced timestamps
            video_duration_ms = self._get_video_duration_ms(video_id)
            timestamps = np.arange(
                sample_interval_ms, video_duration_ms, sample_interval_ms
            )
            return timestamps

        if self.config.use_adaptive_sampling:
            from .sampling import compute_sample_timestamps
            video_path = self.config.get_video_path(video_id)
            timestamps, _ = compute_sample_timestamps(
                video_path,
                target_interval=self.config.target_interval,
                min_frame_skip=self.config.min_frame_skip,
                min_samples=self.config.min_samples,
            )
            return timestamps

        # Union of all rating timestamps
        ratings = self.load_human_ratings(video_id)
        all_timestamps = set()

        for rating_type in ["arousal", "valence"]:
            for rating_data in ratings[rating_type]:
                all_timestamps.update(rating_data.timestamps_ms)

        return np.array(sorted(all_timestamps))

    def _get_video_duration_ms(self, video_id: str) -> int:
        """Get video duration in milliseconds using OpenCV."""
        import cv2

        video_path = self.config.get_video_path(video_id)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = int((frame_count / fps) * 1000)
        cap.release()

        return duration_ms

    def get_video_metadata(self, video_id: str) -> Dict:
        """Get video metadata (duration, fps, frame count)."""
        import cv2

        video_path = self.config.get_video_path(video_id)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        metadata = {
            "video_id": video_id,
            "path": str(video_path),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        metadata["duration_ms"] = int(
            (metadata["frame_count"] / metadata["fps"]) * 1000
        )
        metadata["duration_sec"] = metadata["duration_ms"] / 1000

        cap.release()
        return metadata

    def aggregate_human_ratings(
        self, video_id: str, timestamp_ms: int, rating_type: str = "both"
    ) -> Dict[str, float]:
        """
        Aggregate human ratings at a specific timestamp across all subjects.

        Uses interpolation to get ratings at exact timestamps.

        Args:
            video_id: Video ID
            timestamp_ms: Timestamp in milliseconds
            rating_type: "arousal", "valence", or "both"

        Returns:
            Dict with aggregated ratings (mean) for requested types
        """
        ratings = self.load_human_ratings(video_id)
        result = {}

        types_to_process = (
            ["arousal", "valence"] if rating_type == "both" else [rating_type]
        )

        for rt in types_to_process:
            values = []
            for rating_data in ratings[rt]:
                if len(rating_data.timestamps_ms) < 2:
                    continue

                # Interpolate rating at the given timestamp
                interp_value = np.interp(
                    timestamp_ms,
                    rating_data.timestamps_ms,
                    rating_data.ratings,
                )
                values.append(interp_value)

            if values:
                result[rt] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "n_subjects": len(values),
                }
            else:
                result[rt] = {"mean": np.nan, "std": np.nan, "n_subjects": 0}

        return result

    def get_all_aggregated_ratings(
        self, video_id: str, timestamps_ms: np.ndarray
    ) -> pd.DataFrame:
        """
        Get aggregated human ratings at all specified timestamps.

        Returns:
            DataFrame with columns: timestamp_ms, arousal_mean, arousal_std,
            valence_mean, valence_std, n_subjects
        """
        records = []

        for ts in timestamps_ms:
            agg = self.aggregate_human_ratings(video_id, ts)
            records.append({
                "video_id": video_id,
                "timestamp_ms": ts,
                "arousal_mean": agg["arousal"]["mean"],
                "arousal_std": agg["arousal"]["std"],
                "valence_mean": agg["valence"]["mean"],
                "valence_std": agg["valence"]["std"],
                "n_subjects": agg["arousal"]["n_subjects"],
            })

        return pd.DataFrame(records)

    def get_subject_count(self, video_id: str) -> Dict[str, int]:
        """Get number of subjects who rated this video."""
        ratings = self.load_human_ratings(video_id)
        return {
            "arousal": len(ratings["arousal"]),
            "valence": len(ratings["valence"]),
        }

    def export_human_ratings(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Export all aggregated human ratings for all videos to CSV.

        Uses adaptive sampling when enabled, otherwise fixed sample_interval_ms.
        """
        all_records = []

        for video_id in self.config.video_ids:
            try:
                timestamps = self.get_rating_timestamps(video_id)
                df = self.get_all_aggregated_ratings(video_id, timestamps)
                all_records.append(df)
            except Exception as e:
                print(f"Warning: Failed to process {video_id}: {e}")

        result = pd.concat(all_records, ignore_index=True)

        if output_path is None:
            output_path = self.config.ratings_output_dir / "human_ratings.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Exported human ratings to {output_path}")

        return result
