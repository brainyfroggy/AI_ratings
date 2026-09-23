"""LLM-based emotion rating using OpenAI and Claude."""

import base64
import json
import logging
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple, Union

logger = logging.getLogger(__name__)


def compute_audio_window(timestamp_ms, frame_span_s, audio_window_seconds):
    """Audio slice [start_s, end_s] to send to the LLM at a given moment.

    The audio window is decoupled from the frame window so audio strategy can be
    varied while the frame strategy is held fixed (M2 sends one contiguous slice).

    Args:
        timestamp_ms: current rating moment, ms.
        frame_span_s: the frame context span (seconds) used at this moment — the
            fallback when no audio-specific window is set (preserves prior M2
            behaviour: cumulative [0, t] when the frame span is the full history).
        audio_window_seconds: if not None, send only the trailing W seconds
            [t-W, t]; clamped at the video start. If None, fall back to frame_span_s.

    Returns (start_s, end_s) with 0 <= start_s <= end_s = timestamp_ms/1000.
    """
    end_s = timestamp_ms / 1000.0
    span = frame_span_s if audio_window_seconds is None else min(audio_window_seconds, end_s)
    start_s = max(0.0, end_s - span)
    return (start_s, end_s)


from PIL import Image

from .config import Config, default_config
from .rating_strategy import (
    RatingStrategy,
    TimestampRating,
    VideoRating,
    aggregate_ratings,
    STRATEGY_18VIDEO,
)
from .prompt_builder import build_rating_prompt

# Backward-compatible alias
EmotionRating = TimestampRating


def encode_image_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format, quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Legacy prompt helpers (thin delegates) ───────────────────────────

def build_prompt(n_frames: int, context_seconds: float, transcription: str) -> str:
    """Build the prompt for 18-video emotion rating (backward compat)."""
    return build_rating_prompt(
        STRATEGY_18VIDEO, n_frames, context_seconds, transcription
    )


# ── Response parsing ─────────────────────────────────────────────────

_RATING_WRAPPER_KEYS = ("rating", "ratings", "result", "results", "data", "output")


def _find_rating_dict(node, _depth: int = 0):
    """Walk a parsed JSON tree and return the first dict that has a
    valence/arousal key (case-insensitive). Handles nested lists like
    [[{...}]] and wrapper envelopes like {"rating": {...}}.
    """
    if _depth > 6:
        return None
    if isinstance(node, dict):
        keys_lower = {k.lower(): k for k in node.keys()}
        if "valence" in keys_lower or "arousal" in keys_lower:
            return node
        for k in _RATING_WRAPPER_KEYS:
            if k in keys_lower:
                found = _find_rating_dict(node[keys_lower[k]], _depth + 1)
                if found is not None:
                    return found
        for v in node.values():
            found = _find_rating_dict(v, _depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(node, list):
        for item in node:
            found = _find_rating_dict(item, _depth + 1)
            if found is not None:
                return found
        return None
    return None


def _coerce_scalar(value, default: float) -> float:
    """Coerce a JSON value into a float, averaging if it's a list of numbers."""
    if isinstance(value, list):
        nums = [float(x) for x in value if isinstance(x, (int, float))]
        return sum(nums) / len(nums) if nums else default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def parse_rating_response(
    response: str,
    rating_scale: Tuple[float, float] = (0, 100),
) -> Dict[str, float]:
    """Parse JSON rating from LLM response.

    Tolerates Gemini quirks: nested lists ([[{...}]]), wrapper envelopes
    ({"rating": {...}}), per-frame arrays ({"valence": [..]}), and
    markdown code fences. Raises ValueError if no rating dict is found.
    """
    response = response.strip()

    if "```" in response:
        lines = response.split("```")
        for line in lines:
            line = line.strip()
            if line.startswith("json"):
                line = line[4:].strip()
            if line.startswith("{") or line.startswith("["):
                response = line
                break

    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        match = re.search(
            r'\{[^{}]*"(?:valence|arousal)"[^{}]*\}', response, re.IGNORECASE
        )
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to parse response: {response[:200]}"
                )
        else:
            raise ValueError(
                f"Failed to parse response: {response[:200]}"
            )

    rating = _find_rating_dict(data)
    if rating is None:
        raise ValueError(
            f"No valence/arousal dict found in response: {response[:200]}"
        )

    keys_lower = {k.lower(): k for k in rating.keys()}
    val_key = keys_lower.get("valence")
    aro_key = keys_lower.get("arousal")
    valence = _coerce_scalar(rating.get(val_key) if val_key else None, 50.0)
    arousal = _coerce_scalar(rating.get(aro_key) if aro_key else None, 50.0)

    lo, hi = rating_scale
    valence = round(max(lo, min(hi, valence)), 2)
    arousal = round(max(lo, min(hi, arousal)), 2)

    return {"valence": valence, "arousal": arousal}


def parse_ck_full_response(response: str) -> Dict[str, float]:
    """Parse a ck_full 48-dimension JSON response from the LLM.

    Expected format:
        {"categories": {"Admiration": 42, ...}, "dimensions": {"approach": 6.23, ...}}

    Returns:
        Flat dict with all 48 keys mapped to canonical names.
    """
    from .ck_dimensions import (
        CK_CATEGORIES,
        CK_DIMENSIONS,
        CK_CATEGORY_SCALE,
        CK_DIMENSION_SCALE,
        normalize_key,
    )

    response = response.strip()

    # Strip markdown code blocks if present
    if "```" in response:
        lines = response.split("```")
        for line in lines:
            line = line.strip()
            if line.startswith("json"):
                line = line[4:].strip()
            if line.startswith("{"):
                response = line
                break

    # Parse JSON
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        # Try to find the JSON object in the response
        match = re.search(r'\{[\s\S]*"categories"[\s\S]*"dimensions"[\s\S]*\}', response)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse ck_full response: {response[:300]}")
        else:
            raise ValueError(f"Failed to parse ck_full response: {response[:300]}")

    # Extract categories and dimensions sections
    raw_cats = data.get("categories", {})
    raw_dims = data.get("dimensions", {})

    if not raw_cats and not raw_dims:
        # Flat format fallback — all keys at top level
        raw_cats = {k: v for k, v in data.items() if normalize_key(k) in CK_CATEGORIES}
        raw_dims = {k: v for k, v in data.items() if normalize_key(k) in CK_DIMENSIONS}

    result = {}

    # Process categories (0-100 integer scale)
    cat_lo, cat_hi = CK_CATEGORY_SCALE
    for raw_key, raw_val in raw_cats.items():
        canon = normalize_key(raw_key)
        if canon and canon in CK_CATEGORIES:
            val = float(raw_val)
            val = round(max(cat_lo, min(cat_hi, val)), 2)
            result[canon] = val

    # Process dimensions (1-9 scale)
    dim_lo, dim_hi = CK_DIMENSION_SCALE
    for raw_key, raw_val in raw_dims.items():
        canon = normalize_key(raw_key)
        if canon and canon in CK_DIMENSIONS:
            val = float(raw_val)
            val = round(max(dim_lo, min(dim_hi, val)), 2)
            result[canon] = val

    # Warn about missing keys
    missing_cats = [c for c in CK_CATEGORIES if c not in result]
    missing_dims = [d for d in CK_DIMENSIONS if d not in result]
    if missing_cats:
        logger.warning("Missing categories in response: %s", missing_cats)
    if missing_dims:
        logger.warning("Missing dimensions in response: %s", missing_dims)

    return result


class LLMRater:
    """Rate emotions using LLMs (OpenAI and Claude)."""

    def __init__(
        self,
        config: Optional[Config] = None,
        strategy: Optional[RatingStrategy] = None,
    ):
        self.config = config or default_config
        self.strategy = strategy or STRATEGY_18VIDEO
        self._openai_client = None
        self._anthropic_client = None
        # Running token-usage tally (filled from API ``response.usage``).
        # Lets a run report real API cost rather than estimates.
        import threading
        self._usage_lock = threading.Lock()
        self.usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # Per-film full-audio cache for M2 (include_audio). Decoding a long
        # film's audio once and slicing [0, t] per point is orders of magnitude
        # faster than re-decoding the .mp4 prefix on every call (which times out
        # on long EmoFilM films under concurrency). Maps video_id -> mp3 Path.
        self._audio_cache: dict = {}
        self._audio_cache_lock = threading.Lock()

    def _record_usage(self, response) -> None:
        """Accumulate token usage from an OpenAI-compatible response."""
        u = getattr(response, "usage", None)
        if u is None:
            return
        with self._usage_lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            self.usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0

    # ── Lazy client initialization ───────────────────────────────────

    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI

            base_url = self.config.openai_base_url
            if base_url:
                self._openai_client = OpenAI(
                    api_key=self.config.openai_api_key,
                    base_url=base_url,
                )
            else:
                self._openai_client = OpenAI(api_key=self.config.openai_api_key)
        return self._openai_client

    @property
    def anthropic_client(self):
        """Lazy initialization of Anthropic client."""
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.Anthropic(
                api_key=self.config.anthropic_api_key
            )
        return self._anthropic_client

    # ── Retry helper ───────────────────────────────────────────────

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True if the exception is transient and worth retrying."""
        from openai import (
            APITimeoutError,
            RateLimitError,
            APIConnectionError,
            InternalServerError,
        )

        if isinstance(exc, (APITimeoutError, RateLimitError,
                            APIConnectionError, InternalServerError)):
            return True

        # Anthropic transient errors
        try:
            import anthropic
            if isinstance(exc, (anthropic.RateLimitError,
                                anthropic.APITimeoutError,
                                anthropic.APIConnectionError,
                                anthropic.InternalServerError)):
                return True
        except ImportError:
            pass

        return False

    @staticmethod
    def _server_retry_delay(exc) -> Optional[float]:
        """Extract a server-suggested retry delay (seconds) from a 429/5xx.

        Honors the OpenAI ``retry-after`` header and the Google quota message
        (``"retryDelay": "11s"`` / ``Please retry in 19.27s``). Returns None if
        none is present.
        """
        try:
            ra = exc.response.headers.get("retry-after")
            if ra:
                return float(ra)
        except Exception:
            pass
        msg = str(exc)
        m = (re.search(r"retry in ([\d.]+)s", msg)
             or re.search(r'retryDelay["\':\s]+([\d.]+)s', msg))
        if m:
            return float(m.group(1))
        return None

    def _call_with_retry(self, fn, *args, **kwargs):
        """Call *fn* with backoff on transient errors.

        Honors any server-suggested retry delay (important for the Gemini
        25 RPM quota, whose 429 asks for ~11–19s — longer than plain
        exponential backoff). Uses ``self.config.max_retries`` (default 6).
        """
        max_retries = getattr(self.config, "max_retries", 6)
        last_exc: Optional[Exception] = None

        for attempt in range(1 + max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt >= max_retries or not self._is_retryable(e):
                    raise
                server_delay = self._server_retry_delay(e)
                if server_delay is not None:
                    wait = min(server_delay + 1.0, 60)  # +1s safety margin
                else:
                    wait = min(2 ** attempt, 30)  # 1, 2, 4 … capped at 30 s
                logger.warning(
                    "Retryable error (attempt %d/%d), waiting %.0fs: %s",
                    attempt + 1, max_retries, wait, e,
                )
                time.sleep(wait)

        raise last_exc  # pragma: no cover

    # ── Private call helpers ─────────────────────────────────────────

    def _call_openai(
        self,
        frames: List[Image.Image],
        prompt: str,
        model: str,
        max_tokens: int = 200,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None,
        audio_b64: Optional[str] = None,
        audio_format: str = "mp3",
    ) -> str:
        """Make an OpenAI-compatible API call with retry.

        Args:
            audio_b64: Optional base64-encoded audio to include alongside frames
                (M2 multimodal condition). Only supported by models with native
                audio input (e.g. Gemini 3.1 Pro via UF proxy).

        Returns the raw response text.
        """
        if temperature is None:
            temperature = self.strategy.temperature
        detail = self.strategy.image_detail
        content = []
        for frame in frames:
            b64 = encode_image_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": detail,
                },
            })
        if audio_b64:
            content.append({
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": audio_format},
            })
        content.append({"type": "text", "text": prompt})

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})

        # Reasoning models (e.g. Gemini 3 Pro) spend completion tokens on
        # hidden "thinking" before the JSON. Gemini 3 cannot disable thinking
        # (budget 0 is rejected); reasoning_effort="low" drives it to ~0 so the
        # JSON is not truncated and cost matches the minimal-thinking baseline.
        extra = {}
        if self.strategy.reasoning_effort is not None:
            extra["reasoning_effort"] = self.strategy.reasoning_effort

        def _do_call():
            # Direct OpenAI gpt-5.2 requires max_completion_tokens;
            # UF LiteLLM proxy uses max_tokens. Try max_tokens first,
            # fall back to max_completion_tokens on error.
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    **extra,
                )
            except Exception as e:
                if "max_tokens" in str(e) and "max_completion_tokens" in str(e):
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        **extra,
                    )
                else:
                    raise
            self._record_usage(response)
            msg = response.choices[0].message
            response_text = msg.content
            if response_text is None:
                raise ValueError(f"Empty response from model {model}")
            return response_text

        return self._call_with_retry(_do_call)

    def _call_claude(
        self,
        frames: List[Image.Image],
        prompt: str,
    ) -> str:
        """Make a Claude API call with retry.  Returns the raw response text."""
        content = []
        for frame in frames:
            b64 = encode_image_base64(frame)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        def _do_call():
            response = self.anthropic_client.messages.create(
                model=self.config.claude_model,
                max_tokens=100,
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text

        return self._call_with_retry(_do_call)

    def _call_openai_audio(
        self,
        audio_b64: str,
        audio_format: str,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None,
    ) -> str:
        """Send raw audio to an OpenAI-compatible LLM. Returns raw response text.

        Works with models that accept ``input_audio`` content blocks
        (e.g. ``gemini-3.1-pro`` via UF proxy, ``gpt-audio-1.5`` direct).
        """
        if temperature is None:
            temperature = self.strategy.temperature

        content = [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}},
        ]

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})

        def _do_call():
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # GPT audio models need modalities=["text"]
            if model.startswith("gpt"):
                kwargs["modalities"] = ["text"]
            response = self.openai_client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            if text is None:
                raise ValueError(f"Empty response from model {model}")
            return text

        return self._call_with_retry(_do_call)

    def rate_audio(
        self,
        audio_b64: str,
        audio_format: str = "mp3",
        context_seconds: float = 0.0,
        timestamp_s: float = 0.0,
    ) -> Dict[str, float]:
        """Rate emotion from raw audio. Returns {"valence": float, "arousal": float}.

        Uses the strategy's system_message, rating_scale, and temperature.
        """
        lo, hi = self.strategy.rating_scale
        if context_seconds > 0 and timestamp_s > 0:
            prompt = (
                f"You are rating the emotional content at the current moment "
                f"(end of this audio segment). The audio covers the last "
                f"{context_seconds:.0f} seconds of a music piece, at timestamp "
                f"{timestamp_s:.1f} seconds.\n\n"
            )
        else:
            prompt = "Rate the emotional content of this music.\n\n"

        prompt += (
            f"1. **Arousal** ({lo}-{hi}): How energetic, intense, or activating is "
            f"this music? {lo} = very calm/relaxing, {hi} = very exciting/energetic.\n\n"
            f"2. **Valence** ({lo}-{hi}): How pleasant or unpleasant is this music? "
            f"{lo} = very unpleasant/negative, {hi} = very pleasant/positive.\n\n"
            f"Use the FULL range of the scale. Provide precise ratings with two "
            f"decimal places.\n\n"
            f'Respond ONLY with a JSON object: {{"arousal": X.XX, "valence": Y.YY}}'
        )

        text = self._call_openai_audio(
            audio_b64,
            audio_format,
            prompt,
            model=self.config.openai_model,
            system_message=self.strategy.system_message,
        )
        return parse_rating_response(text, self.strategy.rating_scale)

    # ── Public rate_with_* methods (backward-compat thin delegates) ──

    def rate_with_openai(
        self,
        frames: List[Image.Image],
        transcription: str = "",
        context_seconds: float = 5.0,
        audio_b64: Optional[str] = None,
        prev_rating: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """Rate emotions using OpenAI model."""
        prompt = build_rating_prompt(
            self.strategy, len(frames), context_seconds, transcription, prev_rating=prev_rating
        )
        # Headroom so a brief burst of reasoning tokens never truncates the
        # JSON (reasoning_effort="low" keeps actual usage near the output size).
        max_tokens = 1500 if self.strategy.rating_dimensions == "ck_full" else 1024
        text = self._call_openai(
            frames, prompt, self.config.openai_model,
            max_tokens=max_tokens,
            system_message=self.strategy.system_message,
            audio_b64=audio_b64,
        )
        if self.strategy.rating_dimensions == "ck_full":
            return parse_ck_full_response(text)
        return parse_rating_response(text, self.strategy.rating_scale)

    def rate_with_claude(
        self,
        frames: List[Image.Image],
        transcription: str = "",
        context_seconds: float = 5.0,
    ) -> Dict[str, float]:
        """Rate emotions using Claude (backward compat)."""
        prompt = build_rating_prompt(
            self.strategy, len(frames), context_seconds, transcription
        )
        text = self._call_claude(frames, prompt)
        return parse_rating_response(text, self.strategy.rating_scale)

    # ── Generic rate() ───────────────────────────────────────────────

    def rate(
        self,
        frames: List[Image.Image],
        transcription: str = "",
        llm: str = "gpt-5.2",
        context_seconds: float = 5.0,
        audio_b64: Optional[str] = None,
        prev_rating: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """Rate emotions using specified LLM."""
        if llm in ("gpt-5.4", "gpt-5.2", "gpt-4.1", "gpt-4o", "gemini-3.1-pro"):
            return self.rate_with_openai(frames, transcription, context_seconds, audio_b64=audio_b64, prev_rating=prev_rating)
        elif llm == "claude":
            return self.rate_with_claude(frames, transcription, context_seconds)
        else:
            raise ValueError(f"Unknown LLM: {llm}")

    # ── Per-timestamp rating ─────────────────────────────────────────

    def _ensure_full_audio(self, video_id: str, video_path) -> Optional[Path]:
        """Decode a film's entire soundtrack to one mp3, once, and cache it.

        Returns the cached mp3 Path, or None if extraction failed. Guarded by a
        lock so that 16 concurrent workers starting the same film trigger only
        one decode (the rest reuse the cache). The cache file lives in the temp
        dir (node-local /scratch under SLURM, auto-cleaned).
        """
        import subprocess, tempfile, threading
        with self._audio_cache_lock:
            cached = self._audio_cache.get(video_id)
            if cached is not None and Path(cached).exists():
                return cached
            out_path = Path(tempfile.gettempdir()) / f"fullaudio_{video_id}.mp3"
            if not out_path.exists():
                try:
                    result = subprocess.run(
                        ["ffmpeg", "-y", "-i", str(video_path),
                         "-c:a", "libmp3lame", "-q:a", "4", "-vn", str(out_path)],
                        capture_output=True, timeout=900,
                    )
                    if result.returncode != 0:
                        return None
                except Exception:
                    return None
            self._audio_cache[video_id] = out_path
            return out_path

    def _clear_audio_cache(self, video_id: str) -> None:
        """Drop a film's cached full-audio mp3 after it finishes (bound disk)."""
        with self._audio_cache_lock:
            cached = self._audio_cache.pop(video_id, None)
        if cached is not None:
            Path(cached).unlink(missing_ok=True)

    def rate_video_timestamp(
        self,
        video_id: str,
        timestamp_ms: int,
        llm: Literal["gpt-5.2", "claude"] = "gpt-5.2",
        frames: Optional[List[Image.Image]] = None,
        transcription: Optional[str] = None,
        video_path: Optional[Path] = None,
        prev_rating: Optional[Tuple[float, float]] = None,
    ) -> TimestampRating:
        """Rate emotions at a specific video timestamp.

        Frames are selected from ``[0, timestamp_ms]`` — the full video
        watched so far.  The selection method (uniform vs key-moment) is
        controlled by ``strategy.context_mode``.

        Args:
            video_id: Video ID.
            timestamp_ms: Timestamp in milliseconds.
            llm: Which LLM to use.
            frames: Pre-loaded frames (optional).
            transcription: Pre-loaded transcription (optional).
            video_path: Explicit path to the video file. When provided,
                frames are extracted directly from this path instead of
                going through config.get_video_path().
        """
        # Actual span shown to the LLM (for prompt text)
        if self.strategy.context_seconds is not None:
            span_sec = min(self.strategy.context_seconds, timestamp_ms / 1000.0)
        else:
            span_sec = timestamp_ms / 1000.0
        n_frames = 1 if span_sec == 0 else self.strategy.context_frames_for(span_sec)
        try:
            if frames is None:
                from .frame_extractor import FrameExtractor

                extractor = FrameExtractor(self.config)
                if video_path is None:
                    video_path = self.config.get_video_path(video_id)
                frame_data = extractor.extract_frames_with_context(
                    video_path,
                    timestamp_ms,
                    n_frames=n_frames,
                    context_mode=self.strategy.context_mode,
                    key_moment_threshold_sec=self.strategy.key_moment_threshold_sec,
                    context_seconds=self.strategy.context_seconds,
                )
                frames = [frame for _, frame in frame_data]

            # Get audio if strategy uses it
            audio_b64 = None
            if self.strategy.include_audio:
                # Extract raw audio segment for M2 multimodal condition
                import subprocess, tempfile, base64
                if video_path is None:
                    video_path = self.config.get_video_path(video_id)
                audio_start, audio_end = compute_audio_window(
                    timestamp_ms, span_sec, self.strategy.audio_window_seconds)
                # Slice [audio_start, audio_end] from the pre-decoded full-film
                # mp3 (one decode per film, cached). Slicing a small local mp3
                # with stream-copy is near-instant and never times out — unlike
                # re-decoding the .mp4 prefix on every call.
                full_audio = self._ensure_full_audio(video_id, video_path)
                if full_audio is not None:
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        result = subprocess.run(
                            ["ffmpeg", "-y", "-ss", str(audio_start),
                             "-to", str(audio_end), "-i", str(full_audio),
                             "-c", "copy", tmp_path],
                            capture_output=True, timeout=60,
                        )
                        if result.returncode == 0:
                            with open(tmp_path, "rb") as f:
                                audio_b64 = base64.b64encode(f.read()).decode()
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

            if transcription is None:
                transcription = ""

            rating = self.rate(frames, transcription, llm, context_seconds=span_sec, audio_b64=audio_b64, prev_rating=prev_rating)

            # In ck_full mode, extract valence/arousal from the 48-key dict
            if self.strategy.rating_dimensions == "ck_full":
                valence = rating.get("valence", 5.0)
                arousal = rating.get("arousal", 5.0)
                extra = rating  # all 48 keys
            else:
                valence = rating["valence"]
                arousal = rating["arousal"]
                extra = None

            return TimestampRating(
                video_id=video_id,
                timestamp_ms=timestamp_ms,
                llm=llm,
                valence=valence,
                arousal=arousal,
                raw_response=json.dumps(rating),
                success=True,
                extra_ratings=extra,
            )

        except Exception as e:
            logger.warning(
                "Failed rating %s @ %dms: %s: %s",
                video_id, timestamp_ms, type(e).__name__, e,
            )
            lo, hi = self.strategy.rating_scale
            mid = (lo + hi) / 2
            return TimestampRating(
                video_id=video_id,
                timestamp_ms=timestamp_ms,
                llm=llm,
                valence=mid,
                arousal=mid,
                raw_response="",
                success=False,
                error=str(e),
            )

    # ── Video-level rating ───────────────────────────────────────────

    def rate_video(
        self,
        video_id: str,
        timestamps_ms: List[int],
        llm: Literal["gpt-5.2", "claude"] = "gpt-5.2",
        progress_callback=None,
        max_concurrent: int = 1,
        video_path: Optional[Path] = None,
    ) -> List[TimestampRating]:
        """Rate emotions for all timestamps in a video.

        Args:
            video_id: Video ID.
            timestamps_ms: List of timestamps.
            llm: Which LLM to use.
            progress_callback: Optional callback(current, total).
            max_concurrent: Max concurrent requests (>1 for local vLLM).
            video_path: Explicit video path (for CK pipeline).
        """
        total = len(timestamps_ms)

        # Pre-warm the per-film audio cache once, single-threaded, before any
        # workers spawn — so the one-time full-film decode can't race or trigger
        # a 16-way I/O storm. No-op unless this strategy sends audio (M2).
        if self.strategy.include_audio and total > 0:
            vp = video_path if video_path is not None else self.config.get_video_path(video_id)
            self._ensure_full_audio(video_id, vp)

        # Autoregressive (dial) rating MUST run sequentially in time order so each
        # point can carry the previous (valence, arousal) forward — that's how the
        # felt emotion accumulates (matches the continuous human dial). Failures keep
        # the last SUCCESSFUL state (don't propagate the neutral fallback).
        autoregressive = self.strategy.prompt_mode in (
            "autoregressive", "autoregressive_adaptive", "autoregressive_neutral")

        if max_concurrent <= 1 or autoregressive:
            results = []
            # "autoregressive_neutral": seed the dial at NEUTRAL so the first point is
            # an UPDATE from center (like the human dial, which starts at the middle and
            # is moved as evidence accumulates) rather than a confident fresh read of
            # frame 1. Other modes start with no prior (prev=None -> "initial felt emotion").
            prev = None
            if self.strategy.prompt_mode == "autoregressive_neutral":
                lo, hi = self.strategy.rating_scale
                mid = (lo + hi) / 2.0
                prev = (mid, mid)
            for i, ts in enumerate(timestamps_ms):
                result = self.rate_video_timestamp(
                    video_id, ts, llm,
                    video_path=video_path,
                    prev_rating=prev if autoregressive else None,
                )
                results.append(result)
                if autoregressive and result.success:
                    prev = (result.valence, result.arousal)
                if self.config.rate_limit_delay > 0:
                    time.sleep(self.config.rate_limit_delay)
                if progress_callback:
                    progress_callback(i + 1, total)
            self._clear_audio_cache(video_id)
            return results

        # Concurrent mode with rate limiting
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        results = [None] * total
        completed = [0]
        lock = threading.Lock()
        rate_delay = self.config.rate_limit_delay

        def _rate_one(idx_ts):
            idx, ts = idx_ts
            return idx, self.rate_video_timestamp(
                video_id, ts, llm,
                video_path=video_path,
            )

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {}
            for i, ts in enumerate(timestamps_ms):
                futures[executor.submit(_rate_one, (i, ts))] = i
                # Throttle submission to stay within RPM budget
                if rate_delay > 0:
                    time.sleep(rate_delay)
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                with lock:
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(completed[0], total)

        self._clear_audio_cache(video_id)
        return results

    def rate_video_with_aggregation(
        self,
        video_id: str,
        timestamps_ms: List[int],
        llm: Literal["gpt-5.2", "claude"] = "gpt-5.2",
        progress_callback=None,
        max_concurrent: int = 1,
        video_path: Optional[Path] = None,
    ) -> Tuple[List[TimestampRating], VideoRating]:
        """Rate all timestamps and return both per-timestamp and aggregated.

        Returns:
            Tuple of (per-timestamp ratings list, aggregated VideoRating).
        """
        ts_ratings = self.rate_video(
            video_id,
            timestamps_ms,
            llm,
            progress_callback=progress_callback,
            max_concurrent=max_concurrent,
            video_path=video_path,
        )
        return ts_ratings, aggregate_ratings(ts_ratings)

    def rate_video_aggregated(
        self,
        video_id: str,
        timestamps_ms: List[int],
        llm: Literal["gpt-5.2", "claude"] = "gpt-5.2",
        progress_callback=None,
        max_concurrent: int = 1,
        video_path: Optional[Path] = None,
    ) -> VideoRating:
        """Rate all timestamps then aggregate into a single VideoRating.

        This is the main entry point for the CK pipeline: rate each
        timestamp with a context window, then average.
        """
        _, aggregated = self.rate_video_with_aggregation(
            video_id,
            timestamps_ms,
            llm,
            progress_callback=progress_callback,
            max_concurrent=max_concurrent,
            video_path=video_path,
        )
        return aggregated

    def rate_video_auto(
        self,
        video_id: str,
        timestamps_ms: List[int],
        llm: Literal["gpt-5.2", "claude"] = "gpt-5.2",
        progress_callback=None,
        max_concurrent: int = 1,
        video_path: Optional[Path] = None,
    ) -> Union[List[TimestampRating], VideoRating]:
        """Rate a video, letting ``strategy.aggregate`` decide the return type.

        When ``strategy.aggregate`` is True (CK pipeline), returns a single
        :class:`VideoRating` (averaged across timestamps).
        When False (18-video pipeline), returns the full
        ``List[TimestampRating]`` time series.
        """
        if self.strategy.aggregate:
            return self.rate_video_aggregated(
                video_id, timestamps_ms, llm,
                progress_callback=progress_callback,
                max_concurrent=max_concurrent,
                video_path=video_path,
            )
        return self.rate_video(
            video_id, timestamps_ms, llm,
            progress_callback=progress_callback,
            max_concurrent=max_concurrent,
            video_path=video_path,
        )

    # ── Batch rating ─────────────────────────────────────────────────

    def rate_all_videos(
        self,
        llm: Literal["gpt-5.2", "claude"] = "gpt-5.2",
        video_ids: Optional[List[str]] = None,
        sample_interval_ms: Optional[int] = None,
        progress_callback=None,
    ) -> List[TimestampRating]:
        """Rate all videos with specified LLM (18-video pipeline)."""
        from .data_loader import DataLoader

        video_ids = video_ids or self.config.video_ids

        if sample_interval_ms is None and not self.config.use_adaptive_sampling:
            sample_interval_ms = self.config.sample_interval_ms

        data_loader = DataLoader(self.config)
        all_results = []

        for video_id in video_ids:
            try:
                timestamps = data_loader.get_rating_timestamps(
                    video_id, sample_interval_ms
                )
                results = self.rate_video(
                    video_id,
                    timestamps.tolist(),
                    llm,
                    progress_callback=lambda c, t: (
                        progress_callback(video_id, c, t)
                        if progress_callback
                        else None
                    ),
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error rating video {video_id}: {e}")

        return all_results

    # ── CSV persistence ──────────────────────────────────────────────

    def save_ratings(
        self,
        ratings: List[TimestampRating],
        llm: str,
        output_path: Optional[Path] = None,
    ):
        """Save ratings to CSV file."""
        import pandas as pd

        records = []
        for r in ratings:
            row = {
                "video_id": r.video_id,
                "timestamp_ms": r.timestamp_ms,
                "llm": r.llm,
                "valence": r.valence,
                "arousal": r.arousal,
                "success": r.success,
                "error": r.error or "",
            }
            # Append extra_ratings columns (ck_full mode)
            if r.extra_ratings:
                for key, val in r.extra_ratings.items():
                    row[key] = val
            records.append(row)

        df = pd.DataFrame(records)

        if output_path is None:
            output_path = self.config.ratings_output_dir / f"{llm}_ratings.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(ratings)} ratings to {output_path}")

        return df

    def load_ratings(self, llm: str) -> "pd.DataFrame":
        """Load ratings from CSV file."""
        import pandas as pd

        path = self.config.ratings_output_dir / f"{llm}_ratings.csv"
        if not path.exists():
            raise FileNotFoundError(f"Ratings file not found: {path}")
        return pd.read_csv(path)
