#!/bin/bash
# Run the CK whole-video rating + analysis locally (no SLURM, no HiPerGator).
# Usage:
#   cp .env.example .env   # fill in keys + CK_VIDEOS_DIR
#   ./run_local.sh                 # rate ALL clips, dims=va
#   DIMENSIONS=ck_full ./run_local.sh
#   ./run_local.sh 0001 0002 0003  # rate only these clip ids (quick test)
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && { set -a; source .env; set +a; }
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (see .env.example)}"
: "${CK_VIDEOS_DIR:?set CK_VIDEOS_DIR (folder of 0001.mp4, ...)}"

LLM="${LLM:-gpt-5.4}"
DIMENSIONS="${DIMENSIONS:-va}"
CONCURRENCY="${CONCURRENCY:-8}"
RPM="${RPM:-600}"
RATINGS_DIR="output/ratings"
PY="${PYTHON:-python3}"

export PYTHONPATH="$PWD"

VIDEOS_ARG=()
if [ "$#" -gt 0 ]; then VIDEOS_ARG=(--videos "$@"); fi

echo "=== CK whole-video rating: $LLM dims=$DIMENSIONS -> $RATINGS_DIR ==="
"$PY" scripts/run_ck_whole_video_rating.py \
    --llm "$LLM" --model "$LLM" \
    --dimensions "$DIMENSIONS" \
    --image-detail low --temperature 0.4 \
    --concurrency "$CONCURRENCY" --rpm "$RPM" \
    --ratings-dir "$RATINGS_DIR" \
    "${VIDEOS_ARG[@]}" \
    --resume

echo "=== analysis: LLM vs human ground truth ==="
"$PY" scripts/analyze_ck_whole_video.py --ratings "$RATINGS_DIR/${LLM}_ratings.csv" --llm "$LLM"
echo "Done. Ratings in $RATINGS_DIR/, figures in doc/figures/"
