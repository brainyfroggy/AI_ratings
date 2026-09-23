"""Cowen-Keltner 34 emotion categories + 14 affective dimensions.

Single source of truth for dimension names, scales, and prompt anchors.
Used by the ck_full rating mode to generate prompts and parse responses.
"""

from typing import Optional

# ── 34 Emotion Categories ──────────────────────────────────────────
# Names match the CK CSV column headers exactly (title case).
# LLM rates 0-100 integer intensity; normalize to 0-1 post-hoc.

CK_CATEGORIES = [
    "Admiration",
    "Adoration",
    "Aesthetic Appreciation",
    "Amusement",
    "Anger",
    "Anxiety",
    "Awe",
    "Awkwardness",
    "Boredom",
    "Calmness",
    "Confusion",
    "Contempt",
    "Craving",
    "Disappointment",
    "Disgust",
    "Empathic Pain",
    "Entrancement",
    "Envy",
    "Excitement",
    "Fear",
    "Guilt",
    "Horror",
    "Interest",
    "Joy",
    "Nostalgia",
    "Pride",
    "Relief",
    "Romance",
    "Sadness",
    "Satisfaction",
    "Sexual Desire",
    "Surprise",
    "Sympathy",
    "Triumph",
]

CK_CATEGORY_SCALE = (0, 100)  # LLM output range; rescale to 0-1 for comparison

CATEGORY_DEFINITIONS = {
    "Admiration": "respect and warm approval for someone or something",
    "Adoration": "deep love and devotion",
    "Aesthetic Appreciation": "pleasure from beauty in art, nature, or design",
    "Amusement": "finding something funny or entertaining",
    "Anger": "strong displeasure or hostility",
    "Anxiety": "worry, unease, or nervousness about uncertain outcomes",
    "Awe": "wonder and reverence at something vast or powerful",
    "Awkwardness": "social discomfort or embarrassment",
    "Boredom": "lack of interest or engagement",
    "Calmness": "peaceful, relaxed, free from agitation",
    "Confusion": "uncertainty or lack of understanding",
    "Contempt": "disdain or disrespect toward someone or something",
    "Craving": "intense desire or longing",
    "Disappointment": "sadness from unmet expectations",
    "Disgust": "strong revulsion or repugnance",
    "Empathic Pain": "feeling another person's suffering",
    "Entrancement": "being captivated or spellbound",
    "Envy": "wanting what someone else has",
    "Excitement": "eager enthusiasm and heightened energy",
    "Fear": "alarm or dread in response to threat or danger",
    "Guilt": "remorse over a wrongdoing",
    "Horror": "intense shock and revulsion",
    "Interest": "curiosity and attentive engagement",
    "Joy": "happiness and delight",
    "Nostalgia": "bittersweet longing for the past",
    "Pride": "satisfaction in one's own or another's achievements",
    "Relief": "easing of distress or anxiety",
    "Romance": "feelings of love and intimate connection",
    "Sadness": "sorrow, grief, or unhappiness",
    "Satisfaction": "contentment from fulfillment",
    "Sexual Desire": "physical attraction and arousal",
    "Surprise": "reaction to something unexpected",
    "Sympathy": "compassion and concern for another's misfortune",
    "Triumph": "exultation from victory or success",
}


# ── 14 Affective Dimensions ────────────────────────────────────────
# Names match the CK CSV column headers exactly (lowercase).
# LLM rates on a 1-9 Likert scale, matching human data.

CK_DIMENSIONS = [
    "approach",
    "arousal",
    "attention",
    "certainty",
    "commitment",
    "control",
    "dominance",
    "effort",
    "fairness",
    "identity",
    "obstruction",
    "safety",
    "upswing",
    "valence",
]

CK_DIMENSION_SCALE = (1, 9)

DIMENSION_DEFINITIONS = {
    "approach": "1 = strong avoidance/withdrawal, 9 = strong approach/engagement",
    "arousal": "1 = very calm and relaxed, 9 = very excited and activated",
    "attention": "1 = inattentive/disengaged, 9 = highly focused and attentive",
    "certainty": "1 = very uncertain/confused, 9 = very certain/clear",
    "commitment": "1 = no commitment/detached, 9 = deeply committed/invested",
    "control": "1 = no control over the situation, 9 = complete control",
    "dominance": "1 = feeling submissive/powerless, 9 = feeling dominant/powerful",
    "effort": "1 = effortless/easy, 9 = extremely effortful/strenuous",
    "fairness": "1 = very unfair/unjust, 9 = very fair/just",
    "identity": "1 = threatens sense of self, 9 = affirms sense of self",
    "obstruction": "1 = no obstacles/unimpeded, 9 = heavily obstructed/blocked",
    "safety": "1 = very dangerous/threatening, 9 = very safe/secure",
    "upswing": "1 = things getting much worse, 9 = things getting much better",
    "valence": "1 = very negative/unpleasant, 9 = very positive/pleasant",
}


# ── Helpers ─────────────────────────────────────────────────────────

# All 48 keys in canonical order
CK_ALL_KEYS = CK_CATEGORIES + CK_DIMENSIONS

# Quick lookup sets (lowercase for fuzzy matching)
_CATEGORY_SET = {c.lower() for c in CK_CATEGORIES}
_DIMENSION_SET = {d.lower() for d in CK_DIMENSIONS}


def normalize_key(raw_key: str) -> Optional[str]:
    """Map a raw key from LLM output to canonical CK name.

    Case-insensitive, strips leading/trailing whitespace and underscores.
    Returns None if no match found.
    """
    cleaned = raw_key.strip().replace("_", " ")

    # Check categories (title case)
    for cat in CK_CATEGORIES:
        if cleaned.lower() == cat.lower():
            return cat

    # Check dimensions (lowercase)
    for dim in CK_DIMENSIONS:
        if cleaned.lower() == dim.lower():
            return dim

    return None
