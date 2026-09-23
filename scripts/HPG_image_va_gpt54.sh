#!/bin/bash
# ===== Still-image valence/arousal rating: GPT-5.4 via NaviGator =====
# Submit with: sbatch scripts/HPG_image_va_gpt54.sh
# This script refuses to run outside a SLURM allocation so ratings are not run
# on a HiPerGator login node.
#SBATCH --job-name=image_va_gpt54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=hpg-default
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=mzding
#SBATCH --qos=mzding

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: submit this script with sbatch; do not run ratings on a login node."
    exit 2
fi

module purge && module load python/3.10

PROJECT_ROOT="${PROJECT_ROOT:-/blue/mzding/yujunchen/projects/AI_ratings/codes}"
DATA_ROOT="${DATA_ROOT:-/blue/mzding/yujunchen/projects/data}"

export NAPS_IMAGES_DIR="${NAPS_IMAGES_DIR:-$DATA_ROOT/NAPS_H/NAPS_H}"
export IAPS_IMAGES_DIR="${IAPS_IMAGES_DIR:-$DATA_ROOT/IAPS1182/IAPS1182}"
export OASIS_IMAGES_DIR="${OASIS_IMAGES_DIR:-$DATA_ROOT/OASIS/images}"

LLM="${LLM:-gpt-5.4}"
MODEL="${MODEL:-gpt-5.4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/image_va_ratings}"
IMAGE_DETAIL="${IMAGE_DETAIL:-low}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-512}"
JPEG_QUALITY="${JPEG_QUALITY:-80}"
BATCH_SIZE="${BATCH_SIZE:-50}"

cd "$PROJECT_ROOT"

if [ -d venv ]; then
    source venv/bin/activate
fi
pip install --quiet --disable-pip-version-check openai pillow

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export OPENAI_API_KEY="${GPT_NAVIGATOR_API_KEY:-${OPENAI_API_KEY:-}}"
export OPENAI_BASE_URL="${GPT_NAVIGATOR_BASE_URL:-${OPENAI_BASE_URL:-}}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: set GPT_NAVIGATOR_API_KEY or OPENAI_API_KEY in $PROJECT_ROOT/.env"
    exit 1
fi
if [ -z "${OPENAI_BASE_URL:-}" ]; then
    echo "ERROR: set GPT_NAVIGATOR_BASE_URL or OPENAI_BASE_URL in $PROJECT_ROOT/.env"
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT"

python3 scripts/run_image_dataset_va_rating.py \
    --model "$MODEL" \
    --llm "$LLM" \
    --output-dir "$OUTPUT_DIR" \
    --image-detail "$IMAGE_DETAIL" \
    --max-image-side "$MAX_IMAGE_SIDE" \
    --jpeg-quality "$JPEG_QUALITY" \
    --batch-size "$BATCH_SIZE"
