# AI_ratings

## Purpose

Code for using large language models — mainly GPT-5.x via the OpenAI API
(directly, or through UF's NaviGator proxy), with an optional Claude (Anthropic)
rating path — to rate the affective content of visual stimuli, and comparing
those ratings against human ground truth. Two stimulus types are covered:
still images (IAPS, NAPS, and OASIS picture sets, rated for valence/arousal)
and Cowen-Keltner (CK) emotional video clips (rated for valence/arousal, or
the full set of 14/48 CK affective dimensions, from one overall judgment per
clip built from extracted video frames — no audio). The repo includes prompt
design, batch/async OpenAI API orchestration, video-frame extraction, HPC
(HiPerGator/Slurm) job scripts, and analysis code that scores model ratings
against the human CK ground truth.

This is a research-code snapshot, not a packaged library: paths, dataset
locations, and API settings are configured through `.env` files and CLI
flags/environment variables (see `.env.example` in each package below), and
several scripts default to the original lab file-share layout
(`N:\Experimental_Data\...`). No stimulus images, video clips, extracted
frames, model outputs, or the human ground-truth CSV are included in this
repo — only code and documentation.

## Contents

- **`src/`** and **`scripts/`** — The main package (mirrors `codes/` in the
  original project). `src/` holds the shared modules: `llm_rater.py` (OpenAI
  and Claude rating calls), `prompt_builder.py`, `frame_extractor.py`,
  `change_detection.py`, `sampling.py`, `rating_strategy.py`, `config.py` /
  `ck_config.py`, `data_loader.py` / `ck_data_loader.py`, `ck_dimensions.py`.
  `scripts/` holds the entry points: `run_image_dataset_va_rating.py`
  (IAPS/NAPS/OASIS valence/arousal), `run_ck_whole_video_rating.py` (CK video
  rating), `analyze_ck_whole_video.py` (scores ratings against the human CK
  ground truth), HiPerGator Slurm/batch submission scripts
  (`HPG_*.sh`, `submit_hpg_verify.cmd`, `upload_to_hpg.cmd`), and
  `README_ck_whole_video.md` (full method notes: frame selection, prompts,
  design rationale). `.env.example`, `requirements.txt`, and `run_local.sh`
  are the local run entry point; copy `.env.example` to `.env` and fill in
  your own `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if using a proxy) before
  running anything — the real `.env` used in development is intentionally
  **not** included.
- **`code_images/`** — A second, independent implementation of the still-image
  valence/arousal pipeline (`openai_image_va_batch.py`) that rates IAPS, NAPS,
  and OASIS pictures in batches (default 50 images/request) using original
  image bytes (no resizing/recompression), with a dry-run mode, a one-request
  test mode, and an OpenAI Batch API mode. `ck_image_va_prompt.md` documents
  the CK-derived prompt used; `README_image_va_batch.md` documents usage and
  outputs in detail.
- **`code_videos/`** — Video-side tooling: `extract_ck_whole_video_keyframes.py`
  (extracts key frames per CK clip using the validated CK frame-selection
  method), `openai_video_va_batch.py` (rates video datasets from whole-video
  frame summaries, mirroring the CK one-request-per-clip design),
  `ck_keyframe_distribution_summary.ipynb` and
  `update_ck_keyframe_notebook_charts.py` (frame-count/timing distribution
  analysis and chart generation).
- **`ck_whole_video_standalone/`** — A self-contained bundle for rating CK
  clips end to end without the rest of the repo: vendored copies of the
  needed `src/` modules, `scripts/run_ck_whole_video_rating.py` (valence/
  arousal or all 48 CK dimensions) and
  `scripts/run_ck_whole_video_affective_yujun.py` (the 14 core CK affective
  dimensions on a 1-9 scale), `analyze_ck_whole_video.py`, a HiPerGator batch
  script, `run_local.sh` / `run_local_yujun.sh` wrappers, and
  `README.md` / `README_ck_whole_video.md` / `README_affective14_yujun.md`
  for usage. This is the canonical copy — a couple of near-duplicate copies
  that existed in the original project tree were intentionally left out.

## How to Use

This repo covers **two separate use cases** — rating still images and rating
CK videos. They share some underlying code but are run independently; there
is no single end-to-end pipeline that does both.

Neither use case ships with any stimuli, credentials, or trained-model
weights. Everything below assumes you supply your own image/video files and
your own API key(s) — see `.env.example` at the repo root and inside
`ck_whole_video_standalone/` for exactly which variables to set, and copy it
to `.env` before running anything (the development `.env` itself is not
included and never contained anything but placeholder text in this repo's
history).

### Use case 1 — rate still images (IAPS / NAPS / OASIS) for valence/arousal

Expected input: local folders of IAPS, NAPS, and/or OASIS images (each is a
published, licensed picture set; obtain your own copy and point the scripts
at it — none are bundled here).

Two independent implementations are included; either one works standalone —
pick one, you do not need both:

1. `scripts/run_image_dataset_va_rating.py` — one API request per dataset
   with every image attached, in filename order. Set
   `NAPS_IMAGES_DIR` / `IAPS_IMAGES_DIR` / `OASIS_IMAGES_DIR` (in `.env` or
   the environment) or pass `--dataset NAME=/path/to/images`. Example:
   ```
   python scripts/run_image_dataset_va_rating.py --dry-run
   python scripts/run_image_dataset_va_rating.py
   ```
2. `code_images/openai_image_va_batch.py` — chunked batches (default 50
   images/request), with dry-run, single-request test, and OpenAI Batch API
   modes. Full walkthrough (dry run -> test mode -> full run) is in
   `code_images/README_image_va_batch.md`; short version:
   ```
   python code_images/openai_image_va_batch.py --dry-run
   python code_images/openai_image_va_batch.py --test-mode
   python code_images/openai_image_va_batch.py
   ```

Output (both): a CSV of `valence`/`arousal` per image (original filenames
preserved so ratings join back to source files), plus raw API responses —
written under `output/image_va_ratings/` (script 1) or
`code_images/image_va_batch_results/<dataset>/` (script 2).

### Use case 2 — rate CK video clips (whole-clip valence/arousal, or all 48 CK dims)

Expected input: Cowen-Keltner video clips named `0001.mp4`, `0002.mp4`, ...
(the public CK stimulus set; not bundled), and optionally the human
`CowenKeltnerEmotionalVideos.csv` ground-truth ratings if you want the
`analyze_*` comparison step (the ground-truth CSV ships inside
`ck_whole_video_standalone/` already; elsewhere point `CK_HUMAN_RATINGS_CSV`
at your own copy).

Recommended entry point — the self-contained bundle:

```
cd ck_whole_video_standalone
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in OPENAI_API_KEY / OPENAI_BASE_URL, CK_VIDEOS_DIR
./run_local.sh 0001 0002 0003   # quick test on 3 clips
./run_local.sh                  # full run, all clips, dims=va
DIMENSIONS=ck_full ./run_local.sh   # all 48 CK dimensions
```

This runs `scripts/run_ck_whole_video_rating.py` (one LLM call per clip, one
overall rating) followed by `scripts/analyze_ck_whole_video.py` (scores the
ratings against the human ground truth, producing per-dimension correlation
figures and a `*_vs_human.csv`). See `ck_whole_video_standalone/README.md`
and `scripts/README_ck_whole_video.md` (method notes: frame counts, uniform
vs. key-frame selection, exact prompts) for full detail.

The same pipeline also lives at the repo root (`src/` + `scripts/` +
`run_local.sh`) as part of the full research package, for anyone running
from the whole repo instead of the standalone bundle — usage is identical.
`code_videos/` provides supporting/alternate tooling on top of the same
frame-extraction method: standalone keyframe extraction
(`extract_ck_whole_video_keyframes.py`), a second whole-video batch-rating
implementation (`openai_video_va_batch.py`), and a notebook
(`ck_keyframe_distribution_summary.ipynb`) analyzing frame-count/timing
distributions.

For HiPerGator: `scripts/HPG_ck_whole_video_gpt54.sh` (video) and
`scripts/HPG_image_va_gpt54.sh` / `scripts/HPG_verify_image_setup.sh` (images)
are Slurm submission scripts; `submit_hpg_verify.cmd` / `upload_to_hpg.cmd`
are Windows-side helpers for uploading to and launching jobs on HiPerGator.

## Dependencies

- Python 3.9+
- `pip install -r requirements.txt` (repo root) or
  `pip install -r ck_whole_video_standalone/requirements.txt` (standalone
  bundle) — both list: `openai`, `opencv-python-headless`, `pillow`,
  `pandas`, `numpy`, `scipy`, `matplotlib`, `tqdm`, and `anthropic` (optional,
  only needed for the `--llm claude` / `rate_with_claude` path in
  `src/llm_rater.py`).
- Environment variables, set via `.env` (copied from the relevant
  `.env.example`) or exported directly:
  - `OPENAI_API_KEY`, `OPENAI_BASE_URL` — required for any GPT-5.x call
    (direct OpenAI, or a proxy such as UF's NaviGator); `GPT_NAVIGATOR_API_KEY`
    / `GPT_NAVIGATOR_BASE_URL` are an equivalent alternate pair some scripts
    also accept.
  - `ANTHROPIC_API_KEY` — optional, only for the Claude rating path.
  - `CK_VIDEOS_DIR`, `CK_HUMAN_RATINGS_CSV` — for the CK video pipeline.
  - `NAPS_IMAGES_DIR`, `IAPS_IMAGES_DIR`, `OASIS_IMAGES_DIR` — for the
    still-image pipeline.
- No `ffmpeg` is required — CK clips are rated frames-only (no audio), via
  OpenCV (`opencv-python-headless`).
- For the HiPerGator scripts: a Slurm cluster with a Python environment
  matching the above installed.

## Notes / caveats

- **No `.env` files with real credentials are included.** Only `.env.example`
  templates were copied (at the repo root and inside
  `ck_whole_video_standalone/`); the original working `.env` files (which
  contained a live OpenAI API key) were explicitly excluded.
- No data is included: stimulus images (IAPS/NAPS/OASIS), CK video clips,
  extracted keyframes, batch/API output directories, and the human CK
  ground-truth CSV (`ck_ground_truth.csv`) all live outside this repo and
  must be supplied locally by anyone wishing to reproduce the experiments.
- Several scripts hard-code original lab file-share paths as defaults (e.g.
  `N:\Experimental_Data\yujunchen\projects\data\...`) — these will need to be
  adapted to run in another environment.
- `run_image_dataset_va_rating.py` / `openai_image_va_batch.py` in
  `code_images/` are two independent implementations of the same
  still-image rating task (kept as-is from the original project).
