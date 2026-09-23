#!/bin/bash
# Verify image-rating code and data on a HiPerGator compute node.
# Submit with: sbatch scripts/HPG_verify_image_setup.sh
# This does not call the model; it only counts files and runs the image script in
# --dry-run mode to produce filename manifests.
#SBATCH --job-name=verify_image_va
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --partition=hpg-default
#SBATCH --time=00:30:00
#SBATCH --output=%x.%j.out
#SBATCH --account=mzding
#SBATCH --qos=mzding

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: submit this script with sbatch; do not run verification on a login node."
    exit 2
fi

module purge && module load python/3.10

PROJECT_ROOT="${PROJECT_ROOT:-/blue/mzding/yujunchen/projects/AI_ratings/codes}"
DATA_ROOT="${DATA_ROOT:-/blue/mzding/yujunchen/projects/data}"

export NAPS_IMAGES_DIR="${NAPS_IMAGES_DIR:-$DATA_ROOT/NAPS_H/NAPS_H}"
export IAPS_IMAGES_DIR="${IAPS_IMAGES_DIR:-$DATA_ROOT/IAPS1182/IAPS1182}"
export OASIS_IMAGES_DIR="${OASIS_IMAGES_DIR:-$DATA_ROOT/OASIS/images}"

cd "$PROJECT_ROOT"

echo "Host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Project: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo

test -f scripts/run_image_dataset_va_rating.py
test -f scripts/HPG_image_va_gpt54.sh
test -f .env

count_images() {
    local label="$1"
    local dir="$2"
    local count
    count=$(find "$dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l)
    echo "$label images: $count ($dir)"
}

count_images "NAPS" "$NAPS_IMAGES_DIR"
count_images "IAPS" "$IAPS_IMAGES_DIR"
count_images "OASIS" "$OASIS_IMAGES_DIR"

python3 -m py_compile scripts/run_image_dataset_va_rating.py
python3 scripts/run_image_dataset_va_rating.py \
    --dry-run \
    --output-dir output/image_va_ratings_dryrun_verify

echo
echo "Dry-run manifests:"
wc -l output/image_va_ratings_dryrun_verify/*_va_ratings.csv
echo "Verification complete. No model/API calls were made."
