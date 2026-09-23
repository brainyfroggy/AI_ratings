"""Rating strategy configuration and result dataclasses.

Defines the parameters that distinguish different rating pipelines
(18-video time series vs. CK aggregated) so they can share a single
LLMRater implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RatingStrategy:
    """All parameters that vary between rating pipelines.

    Attributes:
        aggregate: If True, average per-timestamp ratings into one VideoRating
            (CK pipeline).  If False, return the full time series (18-video).
        rating_scale: (min, max) for valence/arousal.
        include_audio: Whether to include audio transcription in prompts.
        prompt_mode: "current" rates the final frame's emotion;
            "overall" rates the whole clip.
        min_context_frames: Minimum frames per rating point (for very short clips).
        max_context_frames: Maximum frames per rating point (LLM vision limit).
        frames_per_sec: Target rate for scaling frame count with duration.
        use_adaptive_sampling: Use per-video adaptive sampling interval.
        target_interval: Nyquist optimum in seconds.
        min_frame_skip: Minimum frames between samples.
        min_samples: Soft target for samples per clip.
        context_mode: Frame selection method for the context window
            ``[0, timestamp_ms]``.
            ``"uniform"`` — evenly spaced.
            ``"key_moments"`` — always use histogram change detection.
            ``"auto"`` — key moments when video watched so far exceeds
            ``key_moment_threshold_sec``, else uniform.
        key_moment_threshold_sec: ``"auto"`` switches to key moments when
            the video watched so far exceeds this duration.
        key_moment_probe_interval_ms: Probe interval for change detection.
    """

    aggregate: bool = False
    rating_scale: Tuple[float, float] = (0, 100)
    include_audio: bool = True
    prompt_mode: str = "current"  # "current" or "overall"
    min_context_frames: int = 3
    max_context_frames: int = 20
    frames_per_sec: float = 1.0
    use_adaptive_sampling: bool = True
    target_interval: float = 0.5
    min_frame_skip: int = 3
    min_samples: int = 5
    context_mode: str = "auto"  # "uniform", "key_moments", or "auto"
    context_seconds: Optional[float] = None  # None = full video [0, t]; float = trailing window
    audio_window_seconds: Optional[float] = None  # M2 audio slice span; None = same as frame span (legacy [0,t]); float = trailing [t-W, t] for audio only
    prompt_style: str = "detailed"  # "detailed" (feature branch) or "simple" (GPT-4o main branch)
    image_detail: str = "low"  # "low" (default) or "high"
    system_message: Optional[str] = None  # system message for safety-sensitive content
    video_domain: str = "trailer"  # "trailer" or "clip" — controls prompt wording
    temperature: float = 0.4  # LLM sampling temperature (default 0.4; STRATEGY_AUDIO pins 0.0)
    rating_dimensions: str = "va"  # "va" (valence/arousal only) or "ck_full" (34 cats + 14 dims)
    reasoning_effort: Optional[str] = None  # OpenAI-compat reasoning_effort ("low"/"medium"/"high"); Gemini 3 needs "low" to suppress thinking

    def context_frames_for(self, duration_sec: float) -> int:
        """Compute the number of context frames for a given duration.

        Scales at ``frames_per_sec`` (default 1 fps), clamped to
        ``[min_context_frames, max_context_frames]``.
        """
        n = int(round(duration_sec * self.frames_per_sec))
        return max(self.min_context_frames, min(self.max_context_frames, n))
    key_moment_threshold_sec: float = 10.0
    key_moment_probe_interval_ms: int = 100


@dataclass
class TimestampRating:
    """A single rating at a single timestamp (replaces EmotionRating)."""

    video_id: str
    timestamp_ms: int
    llm: str
    valence: float
    arousal: float
    raw_response: str
    success: bool
    error: Optional[str] = None
    extra_ratings: Optional[Dict[str, float]] = None  # 48-dim ck_full mode


# Backward-compatible alias
EmotionRating = TimestampRating


@dataclass
class VideoRating:
    """Aggregated rating for an entire video (average of TimestampRatings)."""

    video_id: str
    llm: str
    valence: float
    arousal: float
    n_timestamps: int
    success: bool
    error: Optional[str] = None
    extra_ratings: Optional[Dict[str, float]] = None  # 48-dim ck_full mode


def aggregate_ratings(
    ratings: List[TimestampRating],
    method: str = "mean",
    rating_scale: Tuple[float, float] = (1, 9),
) -> VideoRating:
    """Aggregate a list of TimestampRatings into a single VideoRating.

    Args:
        ratings: Per-timestamp ratings for one video.
        method: Aggregation method.
            ``"mean"`` — simple average.
            ``"peak"`` — max arousal; valence with greatest deviation from
            scale midpoint.
            ``"peak_end"`` — average of peak moment and final moment
            (Kahneman peak-end rule).
            ``"top2"`` — mean of top-2 most extreme timestamps (by arousal
            for arousal, by |valence - mid| for valence).
            ``"top3"`` — same but top-3.
            ``"top25pct"`` — mean of top 25% most extreme timestamps.
            ``"peak_arousal_moment"`` — take both valence and arousal from
            the single timestamp with highest arousal.
        rating_scale: (min, max) of the rating scale, used to compute the
            midpoint for valence extremity.

    Only successful ratings are included.  If none succeed, the VideoRating
    is marked as failed.
    """
    if not ratings:
        raise ValueError("Cannot aggregate empty ratings list")

    video_id = ratings[0].video_id
    llm = ratings[0].llm

    successful = [r for r in ratings if r.success]
    if not successful:
        return VideoRating(
            video_id=video_id,
            llm=llm,
            valence=float("nan"),
            arousal=float("nan"),
            n_timestamps=len(ratings),
            success=False,
            error="All timestamp ratings failed",
        )

    mid = (rating_scale[0] + rating_scale[1]) / 2
    n = len(successful)

    if method == "mean":
        agg_valence = sum(r.valence for r in successful) / n
        agg_arousal = sum(r.arousal for r in successful) / n

    elif method == "peak":
        agg_valence = max(successful, key=lambda r: abs(r.valence - mid)).valence
        agg_arousal = max(r.arousal for r in successful)

    elif method == "peak_end":
        # Peak: most extreme by arousal / valence deviation
        peak_by_arousal = max(successful, key=lambda r: r.arousal)
        peak_by_valence = max(successful, key=lambda r: abs(r.valence - mid))
        # End: last timestamp
        end = max(successful, key=lambda r: r.timestamp_ms)
        agg_valence = (peak_by_valence.valence + end.valence) / 2
        agg_arousal = (peak_by_arousal.arousal + end.arousal) / 2

    elif method == "peak_arousal_moment":
        # Single moment with highest arousal — use both V and A from it
        peak = max(successful, key=lambda r: r.arousal)
        agg_valence = peak.valence
        agg_arousal = peak.arousal

    elif method in ("top2", "top3"):
        k = 2 if method == "top2" else 3
        k = min(k, n)
        sorted_by_arousal = sorted(successful, key=lambda r: r.arousal, reverse=True)
        sorted_by_valence = sorted(successful, key=lambda r: abs(r.valence - mid), reverse=True)
        agg_arousal = sum(r.arousal for r in sorted_by_arousal[:k]) / k
        agg_valence = sum(r.valence for r in sorted_by_valence[:k]) / k

    elif method == "top25pct":
        k = max(1, n // 4)
        sorted_by_arousal = sorted(successful, key=lambda r: r.arousal, reverse=True)
        sorted_by_valence = sorted(successful, key=lambda r: abs(r.valence - mid), reverse=True)
        agg_arousal = sum(r.arousal for r in sorted_by_arousal[:k]) / k
        agg_valence = sum(r.valence for r in sorted_by_valence[:k]) / k

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    # Aggregate extra_ratings if present (ck_full mode)
    agg_extra = None
    extras = [r.extra_ratings for r in successful if r.extra_ratings]
    if extras:
        all_keys = extras[0].keys()
        agg_extra = {}
        for key in all_keys:
            vals = [e[key] for e in extras if key in e]
            agg_extra[key] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return VideoRating(
        video_id=video_id,
        llm=llm,
        valence=round(agg_valence, 2),
        arousal=round(agg_arousal, 2),
        n_timestamps=len(ratings),
        success=True,
        extra_ratings=agg_extra,
    )


# ── Presets ──────────────────────────────────────────────────────────

STRATEGY_18VIDEO = RatingStrategy(
    aggregate=False,
    rating_scale=(0, 100),
    include_audio=False,
    prompt_mode="current",
    context_mode="auto",  # trailers: key moments when >10s watched
)

STRATEGY_CK = RatingStrategy(
    aggregate=True,
    rating_scale=(1, 9),
    include_audio=False,
    prompt_mode="current",
    context_mode="auto",  # short clips → uniform; long clips → key moments
    video_domain="clip",
    system_message=(
        "You are a scientific research assistant performing emotion annotation "
        "for an academic study. Your task is to classify the emotional valence "
        "and arousal conveyed in short video clips. Provide structured numerical "
        "ratings only. Do not describe or narrate the video content."
    ),
)

STRATEGY_AUDIO = RatingStrategy(
    aggregate=False,
    rating_scale=(1, 9),
    include_audio=False,
    prompt_mode="current",
    context_mode="uniform",
    prompt_style="detailed",
    image_detail="low",
    system_message=(
        "You are a scientific research assistant performing emotion annotation "
        "for an academic study on music emotion perception."
    ),
    video_domain="clip",
    temperature=0.0,
    rating_dimensions="va",
)

STRATEGY_CK_FULL = RatingStrategy(
    aggregate=True,
    rating_scale=(1, 9),
    include_audio=False,
    prompt_mode="current",
    context_mode="auto",
    video_domain="clip",
    rating_dimensions="ck_full",
    system_message=(
        "You are a scientific research assistant performing emotion annotation "
        "for an academic study. Your task is to rate the emotional content of "
        "short video clips across 34 emotion categories and 14 affective "
        "dimensions. Provide structured numerical ratings only. Do not describe "
        "or narrate the video content."
    ),
)
