"""Configuration for the Cowen-Keltner video whole-clip rating pipeline."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .config import Config


@dataclass
class CKConfig:
    """Configuration for CK short video clip rating pipeline.

    Reuses API key properties from the base Config class.
    """

    # Base project root (same as main pipeline)
    project_root: Path = field(default_factory=lambda: Path(
        os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent)
    ))

    # CK video directory (env-overridable for HiPerGator)
    ck_videos_dir: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "CK_VIDEOS_DIR",
            "/Volumes/SSD2T/original_ckvideo_data/ckvideo"
        )
    ))

    # Human ratings CSV
    ck_human_ratings_csv: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "CK_HUMAN_RATINGS_CSV",
            "/Volumes/SSD2T/original_ckvideo_data/CowenKeltnerEmotionalVideos.csv"
        )
    ))

    # Output directories (under project_root/output/ck/)
    @property
    def output_dir(self) -> Path:
        return self.project_root / "output" / "ck"

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "frames"

    @property
    def ratings_output_dir(self) -> Path:
        return self.output_dir / "ratings"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    # Frame sampling (budget-based duration-adaptive policy; see report
    # \S sec:interval-policy and src.sampling.compute_sample_timestamps_budget)
    n_frames: int = 8  # Legacy: fixed frame count across full clip
    use_adaptive_sampling: bool = True  # Use per-video adaptive interval
    target_interval: float = 0.2  # Global default / dense interval in seconds
    max_interval: float = 2.0  # Ceiling for long videos (CK clips never reach it)
    call_budget: int = 900  # Target samples/stimulus for long videos
    min_frame_skip: int = 3  # Adaptive: min frames between samples
    min_samples: int = 5  # Adaptive: soft target for samples per clip

    # Rating scale (1-9 to match CK human ratings)
    rating_scale: Tuple[int, int] = (1, 9)

    # LLM parameters
    rate_limit_delay: float = 1.0
    max_retries: int = 3

    # LLM model names (same defaults as base Config)
    openai_model: str = "gpt-5.4"
    claude_model: str = "claude-opus-4-6-20250514"

    # Delegate API key access to a base Config instance
    _base_config: Config = field(default_factory=Config, repr=False)

    @property
    def openai_api_key(self) -> str:
        return self._base_config.openai_api_key

    @property
    def anthropic_api_key(self) -> str:
        return self._base_config.anthropic_api_key

    @property
    def openai_base_url(self) -> Optional[str]:
        return self._base_config.openai_base_url

    def ensure_directories(self):
        """Create output directories if they don't exist."""
        for dir_path in [
            self.output_dir,
            self.frames_dir,
            self.ratings_output_dir,
            self.results_dir,
            self.results_dir / "figures",
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_video_path(self, video_id: str) -> Path:
        """Get path to CK video file (e.g., '0001' -> .../0001.mp4)."""
        return self.ck_videos_dir / f"{video_id}.mp4"
