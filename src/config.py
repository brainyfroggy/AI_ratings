"""Configuration settings for the emotion rating comparison pipeline."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configuration class for the pipeline."""

    # Base paths (can be overridden via environment variables for HiperGator)
    project_root: Path = field(default_factory=lambda: Path(
        os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent)
    ))
    data_root: Path = field(default_factory=lambda: Path(
        os.environ.get("DATA_ROOT", "/Users/pliu1/Dropbox (Personal)/DATA/EmoVideos")
    ))

    # Optional override for videos directory (set via --videos-dir)
    _videos_dir_override: Optional[Path] = None

    # Data paths (derived)
    @property
    def videos_dir(self) -> Path:
        if self._videos_dir_override is not None:
            return self._videos_dir_override
        return self.data_root / "videos"

    @property
    def ratings_dir(self) -> Path:
        return self.data_root / "valence_arousal_rating"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "frames"

    @property
    def audio_dir(self) -> Path:
        return self.output_dir / "audio"

    @property
    def ratings_output_dir(self) -> Path:
        return self.output_dir / "ratings"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    # Video IDs
    video_ids: List[str] = field(default_factory=lambda: [
        f"A{i}" for i in range(1, 19)  # A1 through A18
    ])

    # Video file extension (overridable for datasets using mp4, etc.)
    video_ext: str = field(default_factory=lambda: os.environ.get("VIDEO_EXT", "avi"))

    # Frame extraction parameters
    context_seconds: float = 5.0  # Seconds of context before each rating point
    context_frames: int = 5  # Number of frames to include for context
    sample_interval_ms: int = 2000  # Sample every 2 seconds (legacy fixed interval)
    use_adaptive_sampling: bool = True  # Use per-video adaptive interval
    target_interval: float = 0.5  # Adaptive: Nyquist optimum in seconds
    min_frame_skip: int = 3  # Adaptive: min frames between samples
    min_samples: int = 5  # Adaptive: soft target for samples per clip
    save_frames_to_disk: bool = True  # Save extracted frames for reuse
    frame_format: str = "jpg"  # Format for saved frames
    frame_quality: int = 95  # JPEG quality

    # Audio parameters
    audio_format: str = "mp3"  # Format for extracted audio segments
    audio_bitrate: str = "128k"  # Audio bitrate

    # LLM parameters
    rate_limit_delay: float = 1.0  # Seconds between API calls
    max_retries: int = 6  # Max retries for failed API calls (honors server retry-after; covers Gemini 25 RPM 429s)

    # API keys (loaded from environment)
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key. Returns 'dummy' for local vLLM servers."""
        key = os.environ.get("OPENAI_API_KEY", "")
        # Local vLLM servers don't need a real API key
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if not key and base_url and ("localhost" in base_url or ":" in base_url):
            return "dummy"
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return key

    @property
    def anthropic_api_key(self) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return key

    # LLM model names (overridable via env var)
    openai_model: str = field(default_factory=lambda: os.environ.get(
        "OPENAI_MODEL", "gpt-5.4"
    ))
    claude_model: str = "claude-opus-4-6-20250514"

    # OpenAI API base URL (direct OpenAI by default; override via OPENAI_BASE_URL)
    @property
    def openai_base_url(self) -> Optional[str]:
        """Base URL for OpenAI API (direct OpenAI by default)."""
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Analysis parameters
    min_correlation_samples: int = 10  # Minimum samples for correlation
    max_lag_samples: int = 5  # Max lag for cross-correlation (samples; 5 × 2s = ±10s)

    def ensure_directories(self):
        """Create output directories if they don't exist."""
        for dir_path in [
            self.output_dir,
            self.frames_dir,
            self.audio_dir,
            self.ratings_output_dir,
            self.results_dir,
            self.results_dir / "figures",
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_video_path(self, video_id: str) -> Path:
        """Get path to video file."""
        return self.videos_dir / f"{video_id}.{self.video_ext}"

    def get_frame_dir(self, video_id: str, timestamp_ms: int) -> Path:
        """Get directory for frames at a specific timestamp."""
        return self.frames_dir / video_id / f"{timestamp_ms}ms"

    def get_audio_path(self, video_id: str, timestamp_ms: int) -> Path:
        """Get path for audio segment file."""
        return self.audio_dir / video_id / f"{timestamp_ms}ms.{self.audio_format}"

    def get_transcription_path(self, video_id: str, timestamp_ms: int) -> Path:
        """Get path for transcription file."""
        return self.audio_dir / video_id / f"{timestamp_ms}ms.txt"


# Default configuration instance
default_config = Config()
