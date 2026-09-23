"""Parameterized prompt builder for emotion rating.

Replaces both ``build_prompt()`` (18-video) and ``build_ck_prompt()`` (CK)
with a single function driven by :class:`RatingStrategy`.
"""

from typing import Optional, Tuple

from .rating_strategy import RatingStrategy, STRATEGY_18VIDEO


def build_rating_prompt(
    strategy: RatingStrategy = STRATEGY_18VIDEO,
    n_frames: int = 5,
    context_seconds: float = 5.0,
    transcription: Optional[str] = None,
    prev_rating: Optional[Tuple[float, float]] = None,
) -> str:
    """Build a prompt for LLM emotion rating.

    Args:
        strategy: Rating strategy controlling scale, audio, and wording.
        n_frames: Number of frames being sent.
        context_seconds: Duration of the context window in seconds.
        transcription: Audio transcription text (ignored when
            ``strategy.include_audio`` is False).
        prev_rating: For prompt_mode="autoregressive", the (valence, arousal) from
            the previous timestamp — the "dial" position to update from. None at the
            first timestamp (or other modes).

    Returns:
        The fully-assembled prompt string.
    """
    if strategy.rating_dimensions == "ck_full":
        return _build_ck_full_prompt(strategy, n_frames, context_seconds, transcription)

    if strategy.prompt_style == "simple":
        return _build_simple_prompt(strategy, n_frames, context_seconds, transcription)

    return _build_detailed_prompt(strategy, n_frames, context_seconds, transcription, prev_rating)


def _build_simple_prompt(
    strategy: RatingStrategy,
    n_frames: int,
    context_seconds: float,
    transcription: Optional[str] = None,
) -> str:
    """GPT-4o main-branch style: minimal, no guidance on what to attend to."""
    domain = "a short video clip" if strategy.video_domain == "clip" else "a movie trailer"
    prompt = (
        f"You are rating the emotional content of a video segment.\n"
        f"You will see {n_frames} frames from the last {context_seconds:.0f} "
        f"seconds of {domain}, ordered from oldest to most recent.\n\n"
    )

    if transcription:
        prompt += (
            f"Audio transcription from this segment:\n"
            f"\"{transcription}\"\n\n"
        )

    if strategy.prompt_mode == "current":
        target = "CURRENT emotional state (at the final/most recent frame)"
    else:
        target = "OVERALL emotional content of this video"

    lo, hi = strategy.rating_scale
    prompt += (
        f"Rate the {target} on two dimensions:\n"
        f"- Valence (0.00-100.00): 0=very negative, 50=neutral, 100=very positive\n"
        f"- Arousal (0.00-100.00): 0=very calm/low energy, 100=very excited/high energy\n\n"
        f"Use two decimal places (e.g., 62.50, 33.80).\n"
        f"Consider both the visual content and the audio context to make your rating.\n\n"
        f'Respond ONLY with a JSON object in this exact format: '
        f'{{"valence": X.XX, "arousal": Y.YY}}\n'
        f"Do not include any other text or explanation."
    )

    return prompt


def _build_detailed_prompt(
    strategy: RatingStrategy,
    n_frames: int,
    context_seconds: float,
    transcription: Optional[str] = None,
    prev_rating: Optional[Tuple[float, float]] = None,
) -> str:
    """Feature-branch style: detailed guidance on attending to expressions, etc."""
    lo, hi = strategy.rating_scale
    scale_int = (lo == int(lo) and hi == int(hi))
    lo_s = f"{int(lo)}" if scale_int else f"{lo}"
    hi_s = f"{int(hi)}" if scale_int else f"{hi}"

    mid = (lo + hi) / 2
    mid_s = f"{int(mid)}" if mid == int(mid) else f"{mid}"

    # Example values with two decimal places
    ex1 = f"{lo + (hi - lo) * 0.625:.2f}"
    ex2 = f"{lo + (hi - lo) * 0.338:.2f}"

    domain = "a short video clip" if strategy.video_domain == "clip" else "a movie trailer"

    # Opening: describe what the model will see
    if n_frames == 1:
        prompt = (
            f"You are rating the emotional content of a video frame.\n"
            f"You will see a single frame from {domain}.\n\n"
        )
    else:
        prompt = (
            f"You are rating the emotional content of a video segment.\n"
            f"You will see {n_frames} key frames spanning {context_seconds:.0f} "
            f"seconds of {domain}, ordered from oldest to most recent.\n\n"
        )

    # Audio transcription (only when strategy says to include it)
    if strategy.include_audio and transcription:
        prompt += (
            f"Audio transcription from this segment:\n"
            f"\"{transcription}\"\n\n"
        )

    # Context instruction (only when multiple frames)
    if n_frames > 1:
        prompt += (
            f"Use the earlier frames as context to understand how the emotional "
            f"tone builds, shifts, or escalates over time. "
            f"Pay attention to facial expressions, body language, scene changes, "
            f"and the overall narrative progression.\n\n"
        )

    # Autoregressive (dial) mode: update the viewer's felt emotion from the previous
    # value, the way a continuous-annotation dial is nudged rather than re-set. The
    # "_adaptive" variant lets the dial JUMP on sudden/startling events (so sharp
    # transients like a jump-scare aren't damped by the smoothing prior).
    if strategy.prompt_mode in ("autoregressive", "autoregressive_adaptive", "autoregressive_neutral"):
        if strategy.prompt_mode == "autoregressive_adaptive":
            move_rule = (
                "Move smoothly from the previous value for gradual changes — BUT if "
                "something sudden or startling just happened (a scare, shock, or abrupt "
                "shift), JUMP the dial sharply to match it. Match the pace of the moment: "
                "gentle when it builds slowly, fast when it spikes."
            )
        else:
            move_rule = ("Move smoothly from the previous value (small changes unless "
                         "something major just happened); do not reset or jump.")
        if prev_rating is not None:
            pv, pa = prev_rating
            prompt += (
                f"A continuous viewer is rating their felt emotion on a dial as they "
                f"watch. A moment ago the dial read valence {pv:.2f} and arousal "
                f"{pa:.2f} (same {lo_s}-{hi_s} scale). Treat this as the current dial "
                f"position and UPDATE it: nudge it up, hold it, or nudge it down based "
                f"ONLY on what has happened since — letting the felt emotion accumulate "
                f"over the video. {move_rule}\n\n"
            )
        else:
            prompt += (
                "A continuous viewer is rating their felt emotion on a dial as they "
                "watch. This is the very start of the video — give their initial felt "
                "emotion, which subsequent ratings will update from.\n\n"
            )
        target = ("viewer's UPDATED felt emotion right now (the new dial position)")
    elif strategy.prompt_mode == "current":
        target = "CURRENT emotional state (at the final/most recent frame)"
    elif strategy.prompt_mode == "accumulated":
        target = ("emotion a VIEWER FEELS at this moment, having watched the video up "
                  "to and including the most recent frame — their cumulative felt "
                  "response that builds on everything seen so far, not just the current scene")
    else:
        target = "OVERALL emotional content of this video"

    prompt += (
        f"Rate the {target} on two dimensions:\n"
        f"- Valence ({lo_s}.00-{hi_s}.00): {lo_s}=very negative, "
        f"{mid_s}=neutral, {hi_s}=very positive\n"
        f"- Arousal ({lo_s}.00-{hi_s}.00): {lo_s}=very calm/low energy, "
        f"{hi_s}=very excited/high energy\n\n"
        f"Be precise — use the FULL continuous scale with two decimal places.\n"
        f"Avoid snapping to anchors (.00, .25, .50, .75) or multiples of 5 or 10.\n"
        f"Make fine-grained distinctions. For example, prefer "
        f"{ex1} or {ex2} over round numbers like {mid_s}.00 or {mid_s}.50.\n"
    )

    if strategy.include_audio:
        prompt += (
            "Consider both the visual content and the audio context "
            "to make your rating.\n"
        )

    prompt += (
        "\n"
        'Respond ONLY with a JSON object in this exact format: '
        '{"valence": X.XX, "arousal": Y.YY}\n'
        "Do not include any other text or explanation."
    )

    return prompt


def _build_ck_full_prompt(
    strategy: RatingStrategy,
    n_frames: int,
    context_seconds: float,
    transcription: Optional[str] = None,
) -> str:
    """Build prompt for 34 emotion categories + 14 affective dimensions."""
    from .ck_dimensions import (
        CK_CATEGORIES,
        CK_DIMENSIONS,
        CATEGORY_DEFINITIONS,
        DIMENSION_DEFINITIONS,
    )

    domain = "a short video clip" if strategy.video_domain == "clip" else "a movie trailer"

    # Opening
    if n_frames == 1:
        prompt = (
            f"You are rating the emotional content of a video frame.\n"
            f"You will see a single frame from {domain}.\n\n"
        )
    else:
        prompt = (
            f"You are rating the emotional content of a video segment.\n"
            f"You will see {n_frames} key frames spanning {context_seconds:.0f} "
            f"seconds of {domain}, ordered from oldest to most recent.\n\n"
        )

    # Context instruction
    if n_frames > 1:
        prompt += (
            "Use the earlier frames as context to understand how the emotional "
            "tone builds, shifts, or escalates over time. "
            "Pay attention to facial expressions, body language, scene changes, "
            "and the overall narrative progression.\n\n"
        )

    # Rating target
    if strategy.prompt_mode == "current":
        target = "CURRENT emotional state (at the final/most recent frame)"
    elif strategy.prompt_mode == "accumulated":
        target = ("emotion a VIEWER FEELS at this moment, having watched the video up "
                  "to and including the most recent frame — their cumulative felt "
                  "response that builds on everything seen so far, not just the current scene")
    else:
        target = "OVERALL emotional content of this video"

    prompt += f"Rate the {target} on the following scales.\n\n"

    # Section A: 34 emotion categories
    prompt += "## SECTION A: Emotion Categories (34 items)\n"
    prompt += "Rate the intensity of each emotion on a 0-100 integer scale:\n"
    prompt += "  0 = not present at all, 100 = maximum intensity.\n\n"
    for cat in CK_CATEGORIES:
        defn = CATEGORY_DEFINITIONS.get(cat, "")
        prompt += f"- {cat}: {defn}\n"

    prompt += "\n"

    # Section B: 14 affective dimensions
    prompt += "## SECTION B: Affective Dimensions (14 items)\n"
    prompt += "Rate each dimension on a 1.00-9.00 scale with two decimal places.\n"
    prompt += "Use the FULL continuous scale. Avoid snapping to round numbers.\n\n"
    for dim in CK_DIMENSIONS:
        anchor = DIMENSION_DEFINITIONS.get(dim, "")
        prompt += f"- {dim}: {anchor}\n"

    prompt += "\n"

    # Output format
    cat_example = ", ".join(f'"{c}": 0' for c in CK_CATEGORIES[:3])
    dim_example = ", ".join(f'"{d}": 5.00' for d in CK_DIMENSIONS[:3])

    prompt += (
        "Respond ONLY with a JSON object in this exact format:\n"
        "{\n"
        f'  "categories": {{{cat_example}, ...}},\n'
        f'  "dimensions": {{{dim_example}, ...}}\n'
        "}\n\n"
        "Include ALL 34 categories and ALL 14 dimensions. "
        "Do not include any other text or explanation."
    )

    return prompt
