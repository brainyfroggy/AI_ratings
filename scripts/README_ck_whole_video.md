# CK whole-video rating

Rate each Cowen-Keltner (CK) clip with **one overall emotion judgment**, the same
task the human raters had: watch the clip once, give one rating. This is the
counterpart to our time-course pipeline, which instead rates the clip
moment-by-moment.

Three files:

| File | Role |
|------|------|
| `run_ck_whole_video_rating.py` | Send each clip to the LLM as one call; write one rating per clip. |
| `analyze_ck_whole_video.py` | Compare the LLM ratings to the human ground truth, across all clips. |
| `HPG_ck_whole_video_gpt54.sh` | SLURM job that runs both on HiPerGator (GPT-5.4 via the NaviGator proxy). |

## What the rating call does

`run_ck_whole_video_rating.py` rates one clip per LLM call:

1. Extract frames spanning the **whole** clip `[0, end]`.
2. Send them in a single prompt asking for the clip's **overall** emotional
   content (`prompt_mode="overall"`).
3. Record one rating per clip — valence/arousal (`--dimensions va`) or the full
   48-dim CK vector (`--dimensions ck_full`: 34 emotion categories + 14 affective
   dimensions, matching the human ground truth).

CK clips have no audio, so this is frames-only.

It reuses the existing engine: it calls `LLMRater.rate_video_timestamp(...)` **once**,
at `timestamp = clip duration` (`run_ck_whole_video_rating.py:80`). The single
context window is therefore the entire clip. A whole-video rating is just a
one-shot special case of the per-timestamp machinery.

## How many frames are sent

Frame count comes from `RatingStrategy.context_frames_for()`:

    n_frames = round(window_seconds × frames_per_sec)   # frames_per_sec = 1.0
    n_frames = clamp(n_frames, min=3, max=20)

For the whole-video call the window is the full clip, so it sends **about one
frame per second of clip**, floored at 3 and capped at 20:

| Clip length | Frames |
|---|---|
| ≤ 3 s | 3 (floor) |
| ~5 s | 5 |
| ~10 s | 10 |
| ≥ 20 s | 20 (cap) |

CK clips are mostly a few seconds, so in practice **~3–6 frames per clip**; the
cap only matters for clips ≥ 20 s.

## Uniform frames or key frames?

Same selection rule as the time-course pipeline. The strategy uses
`context_mode="auto"`, which picks the method by window length
(`frame_extractor.py:88`):

- **window ≤ 10 s** → **uniform** sampling (`np.linspace` across `[0, end]`)
- **window > 10 s** → **key-moment** sampling (change-detection profile)

So the deciding factor is clip length. Most CK clips are under 10 s and get
uniform frames; only clips longer than 10 s get key-frame selection. The script
does not force one method — it applies the same `auto` policy as everywhere else.

## The prompts sent to the LLM

Built by `src/prompt_builder.py` from the strategy. `--dimensions` picks which of
two prompts is used. In both, the frames are attached as images (oldest→newest),
and only the frame count and the seconds in the second line vary per clip;
`OVERALL emotional content of this video` is what makes it a whole-video judgment.
To regenerate either verbatim:

    python -c "from dataclasses import replace; from src.prompt_builder import build_rating_prompt; \
    from src.rating_strategy import STRATEGY_CK; \
    print(build_rating_prompt(replace(STRATEGY_CK, prompt_mode='overall'), n_frames=5, context_seconds=5.0))"

Quirk: the text always says "key frames" and "video segment" even for short clips
that are sampled uniformly over the whole clip. Cosmetic only — the frames are
correct; just the wording is loose.

### `--dimensions va` (default) → valence/arousal, JSON `{"valence", "arousal"}`

System message:

    You are a scientific research assistant performing emotion annotation for an
    academic study. Your task is to classify the emotional valence and arousal
    conveyed in short video clips. Provide structured numerical ratings only. Do
    not describe or narrate the video content.

User prompt (example: 5 frames over 5 s):

    You are rating the emotional content of a video segment.
    You will see 5 key frames spanning 5 seconds of a short video clip, ordered from oldest to most recent.

    Use the earlier frames as context to understand how the emotional tone builds, shifts, or escalates over time. Pay attention to facial expressions, body language, scene changes, and the overall narrative progression.

    Rate the OVERALL emotional content of this video on two dimensions:
    - Valence (1.00-9.00): 1=very negative, 5=neutral, 9=very positive
    - Arousal (1.00-9.00): 1=very calm/low energy, 9=very excited/high energy

    Be precise — use the FULL continuous scale with two decimal places.
    Avoid snapping to anchors (.00, .25, .50, .75) or multiples of 5 or 10.
    Make fine-grained distinctions. For example, prefer 6.00 or 3.70 over round numbers like 5.00 or 5.50.

    Respond ONLY with a JSON object in this exact format: {"valence": X.XX, "arousal": Y.YY}
    Do not include any other text or explanation.

### `--dimensions ck_full` → 34 categories + 14 dimensions, JSON `{"categories", "dimensions"}`

Matches the full human ground truth. System message:

    You are a scientific research assistant performing emotion annotation for an
    academic study. Your task is to rate the emotional content of short video clips
    across 34 emotion categories and 14 affective dimensions. Provide structured
    numerical ratings only. Do not describe or narrate the video content.

User prompt (example: 5 frames over 5 s):

    You are rating the emotional content of a video segment.
    You will see 5 key frames spanning 5 seconds of a short video clip, ordered from oldest to most recent.

    Use the earlier frames as context to understand how the emotional tone builds, shifts, or escalates over time. Pay attention to facial expressions, body language, scene changes, and the overall narrative progression.

    Rate the OVERALL emotional content of this video on the following scales.

    ## SECTION A: Emotion Categories (34 items)
    Rate the intensity of each emotion on a 0-100 integer scale:
      0 = not present at all, 100 = maximum intensity.

    - Admiration: respect and warm approval for someone or something
    - Adoration: deep love and devotion
    - Aesthetic Appreciation: pleasure from beauty in art, nature, or design
    - Amusement: finding something funny or entertaining
    - Anger: strong displeasure or hostility
    - Anxiety: worry, unease, or nervousness about uncertain outcomes
    - Awe: wonder and reverence at something vast or powerful
    - Awkwardness: social discomfort or embarrassment
    - Boredom: lack of interest or engagement
    - Calmness: peaceful, relaxed, free from agitation
    - Confusion: uncertainty or lack of understanding
    - Contempt: disdain or disrespect toward someone or something
    - Craving: intense desire or longing
    - Disappointment: sadness from unmet expectations
    - Disgust: strong revulsion or repugnance
    - Empathic Pain: feeling another person's suffering
    - Entrancement: being captivated or spellbound
    - Envy: wanting what someone else has
    - Excitement: eager enthusiasm and heightened energy
    - Fear: alarm or dread in response to threat or danger
    - Guilt: remorse over a wrongdoing
    - Horror: intense shock and revulsion
    - Interest: curiosity and attentive engagement
    - Joy: happiness and delight
    - Nostalgia: bittersweet longing for the past
    - Pride: satisfaction in one's own or another's achievements
    - Relief: easing of distress or anxiety
    - Romance: feelings of love and intimate connection
    - Sadness: sorrow, grief, or unhappiness
    - Satisfaction: contentment from fulfillment
    - Sexual Desire: physical attraction and arousal
    - Surprise: reaction to something unexpected
    - Sympathy: compassion and concern for another's misfortune
    - Triumph: exultation from victory or success

    ## SECTION B: Affective Dimensions (14 items)
    Rate each dimension on a 1.00-9.00 scale with two decimal places.
    Use the FULL continuous scale. Avoid snapping to round numbers.

    - approach: 1 = strong avoidance/withdrawal, 9 = strong approach/engagement
    - arousal: 1 = very calm and relaxed, 9 = very excited and activated
    - attention: 1 = inattentive/disengaged, 9 = highly focused and attentive
    - certainty: 1 = very uncertain/confused, 9 = very certain/clear
    - commitment: 1 = no commitment/detached, 9 = deeply committed/invested
    - control: 1 = no control over the situation, 9 = complete control
    - dominance: 1 = feeling submissive/powerless, 9 = feeling dominant/powerful
    - effort: 1 = effortless/easy, 9 = extremely effortful/strenuous
    - fairness: 1 = very unfair/unjust, 9 = very fair/just
    - identity: 1 = threatens sense of self, 9 = affirms sense of self
    - obstruction: 1 = no obstacles/unimpeded, 9 = heavily obstructed/blocked
    - safety: 1 = very dangerous/threatening, 9 = very safe/secure
    - upswing: 1 = things getting much worse, 9 = things getting much better
    - valence: 1 = very negative/unpleasant, 9 = very positive/pleasant

    Respond ONLY with a JSON object in this exact format:
    {
      "categories": {"Admiration": 0, "Adoration": 0, "Aesthetic Appreciation": 0, ...},
      "dimensions": {"approach": 5.00, "arousal": 5.00, "attention": 5.00, ...}
    }

    Include ALL 34 categories and ALL 14 dimensions. Do not include any other text or explanation.

## How it differs from the time-course pipeline

Both use the same rater, the same 1-fps / `[3, 20]` frame budget, and the same
`auto` key-moment policy. The difference is **where the window ends**:

| | Whole-video (this code) | Time-course (`run_rating.py`) |
|---|---|---|
| LLM calls per clip | 1 | N (one per sampled timestamp) |
| Frame window | `[0, end]` — the whole clip | `[0, t]` — what's watched so far, growing |
| Prompt target | OVERALL clip emotion | CURRENT moment (or dial modes) |
| Output | one scalar (or 48-vec) | a valence/arousal trajectory |
| Key-vs-uniform decided | once, by total clip length | per call, as `t` grows past 10 s |
| Unit of analysis | the clip (between-video) | time within a clip (within-video) |
| Human task it matches | one post-clip judgment | continuous dial annotation |

In the time-course pipeline the selection method can change within one long video
(early timestamps uniform, later ones key-moment). The whole-video call makes that
choice once.

## What the analysis reports

`analyze_ck_whole_video.py` does a **between-video** comparison: one LLM number
vs one human number per clip, correlated across all ~2185 clips, per dimension.
It answers *"does the LLM rank clips by emotion the way the human panel does?"* —
not *"does it follow the dynamics within a clip"* (that's the time-course question).

- Pearson and Spearman per dimension (Pearson is scale-invariant, so a 1–9 vs
  0–100 scale mismatch does not affect it).

It writes three files:

- `<llm>_vs_human.csv` — one row per dimension (pearson, spearman, n).
- `doc/figures/ck_<llm>_whole_video.{pdf,png}` — sorted per-dimension bar chart of
  Pearson r (2 bars for `va`, 48 for `ck_full`).
- `doc/figures/ck_<llm>_va_scatter.{pdf,png}` — valence and arousal scatter, one
  dot per clip (LLM vs human), with the identity line, a least-squares fit, and
  r/n in each title. Written whenever valence and arousal are present.

Note: the figure paths are hardcoded to `doc/figures/` (no CLI override), so
re-runs overwrite them.

## Running it

Only the `.sh` wrapper is HiPerGator-specific (SLURM directives, `module load`,
`/blue/...` paths). The two Python scripts run anywhere — laptop or HPG. CK is
frames-only, so **no ffmpeg is needed**; OpenCV reads the clips directly.

### Locally (no HPG)

1. Install deps:

       pip install openai opencv-python-headless pillow pandas tqdm numpy scipy matplotlib

2. Point at the clips and the human ratings. The defaults are local Mac paths
   (`/Volumes/SSD2T/original_ckvideo_data/{ckvideo, CowenKeltnerEmotionalVideos.csv}`);
   override with env vars if yours differ:

       export CK_VIDEOS_DIR=/path/to/ckvideo
       export CK_HUMAN_RATINGS_CSV=/path/to/CowenKeltnerEmotionalVideos.csv

3. Set the API key. The code does not pick a provider — it reads `OPENAI_API_KEY`
   and `OPENAI_BASE_URL` and uses whatever you export (`src/config.py:85-113`).

   NaviGator proxy (required for `gpt-5.4`, which is only served there):

       export OPENAI_API_KEY=...        # GPT_NAVIGATOR_API_KEY
       export OPENAI_BASE_URL=...        # GPT_NAVIGATOR_BASE_URL

   Direct OpenAI — set the OpenAI key and leave `OPENAI_BASE_URL` unset (it
   defaults to the OpenAI endpoint). This works only with real OpenAI model ids,
   so also switch the model, e.g. `--llm gpt-4o --model gpt-4o`:

       export OPENAI_API_KEY=...        # your OpenAI key
       unset OPENAI_BASE_URL

4. Run the two scripts:

       python scripts/run_ck_whole_video_rating.py --llm gpt-5.4 --model gpt-5.4 \
           --dimensions va --image-detail low --temperature 0.4 \
           --concurrency 8 --rpm 600 --resume

       python scripts/analyze_ck_whole_video.py \
           --ratings output/ck/ratings/gpt-5.4_ratings.csv --llm gpt-5.4

   Output defaults to `output/ck/ratings/` (override with `--ratings-dir`).
   `--resume` skips clips already in the output CSV, so it is safe to re-run.
   On a laptop, lower `--concurrency`/`--rpm` if you hit rate limits.

### Expected warnings

`Warning: Cannot read frame at <N>ms` is harmless. OpenCV cannot grab the frame
at the exact final timestamp of a clip; the extractor skips it and the rating
still succeeds. You will see one or two of these per clip.

### On HiPerGator

    sbatch scripts/HPG_ck_whole_video_gpt54.sh

Set `DIMENSIONS=ck_full` to rate all 48 dims. Needs `GPT_NAVIGATOR_API_KEY` /
`GPT_NAVIGATOR_BASE_URL` in `.env`, and `CK_VIDEOS_DIR` / `CK_HUMAN_RATINGS_CSV`
pointing at the clips and the human ratings.

## Two other ways to get one CK number

For context, there are three ways to collapse a CK clip to a single rating:

1. **Whole-video overall** (this code) — give the LLM the human's holistic task
   directly.
2. **Peak-end-averaged time course** (`run_ck_rating.py`) — rate many timestamps,
   then aggregate (e.g. peak-end) into one value.
3. **Full trajectory** (`run_rating.py`) — only where continuous human annotation
   exists (CASE, EmoFilM, VEATIC); CK has none.
