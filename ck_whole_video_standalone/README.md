# CK whole-video emotion rating — standalone

Rate Cowen-Keltner (CK) video clips with an LLM, one overall judgment per clip
(the same task the human raters had), then compare to the human ground truth.

This bundle is self-contained: it carries the subset of the project's `src/`
modules it needs plus the ground-truth CSV. It does **not** require the full
research repository.

## Contents

    src/                 vendored modules the scripts depend on (do not edit)
    scripts/
      run_ck_whole_video_rating.py   rate clips -> one rating each
      analyze_ck_whole_video.py      compare LLM ratings to human ground truth
      README_ck_whole_video.md       full method notes (frames, prompts, design)
      HPG_ck_whole_video_gpt54.sh    SLURM job for HiPerGator (edit PROJECT_ROOT)
    data/ground_truth/ck_ground_truth.csv   human GT (2185 clips x 48 dims)
    requirements.txt
    .env.example
    run_local.sh         one-command local run (rating + analysis)

You supply the CK video clips (`0001.mp4`, `0002.mp4`, ...) — they are not bundled.

## Quick start

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

    cp .env.example .env        # fill in OPENAI_API_KEY / OPENAI_BASE_URL, CK_VIDEOS_DIR

    ./run_local.sh 0001 0002 0003   # quick test on 3 clips
    ./run_local.sh                  # full run, all clips (dims=va)
    DIMENSIONS=ck_full ./run_local.sh   # all 48 CK dimensions

Outputs: ratings in `output/ratings/`, figures in `doc/figures/` (a per-dimension
correlation bar chart and a valence/arousal scatter), and a `*_vs_human.csv`.

## Keys

The code reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` and uses whatever you set:

- **NaviGator proxy** (required for `gpt-5.4`): set both.
- **Direct OpenAI**: set `OPENAI_API_KEY`, leave `OPENAI_BASE_URL` unset, and use a
  real OpenAI model id, e.g. `LLM=gpt-4o ./run_local.sh`.

## Notes

- CK clips have no audio, so this is frames-only — **no ffmpeg needed**.
- `--resume` (on by default in `run_local.sh`) skips clips already rated, so it is
  safe to re-run after an interruption.
- The extractor now sends the full requested number of frames, including the
  clip's final (apex) frame. OpenCV cannot decode a frame at the exact end
  timestamp, so the extractor steps back one frame interval to grab the last
  readable frame instead of dropping it. A `Warning: Cannot read frame at <N>ms`
  may still print if all fallbacks fail; the rating still succeeds.
- For the method details — how many frames are sent, uniform vs key-frame
  selection, the exact prompts, and how this differs from the moment-by-moment
  time-course approach — see `scripts/README_ck_whole_video.md`.
