#!/bin/bash
# ===== CK WHOLE-VIDEO overall emotion rating: GPT-5.4, frames-only =====
# Rates all 2,185 Cowen-Keltner clips with ONE overall judgment each (like a human watching
# the clip once), NOT moment-by-moment. Frames-only (CK has no audio). Matches the human
# protocol in Cowen & Keltner 2017 (rate the emotional response the clip evokes in a viewer:
# 14 affective dimensions on a 1-9 scale; optionally the 34 emotion categories).
#
# Start with --dimensions va (valence/arousal). Switch to ck_full for all 48 dims.
# GPT-5.4 via the NaviGator proxy (GPT_NAVIGATOR_API_KEY in .env). ~2185 calls.
#SBATCH --job-name=CK_wholevideo_gpt54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --partition=hpg-default
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liupengUFL@gmail.com
#SBATCH --account=mzding
#SBATCH --qos=mzding

set -euo pipefail
module purge && module load python/3.10

PROJECT_ROOT="/blue/mzding/pliu1/LabelEmotionVideoFrames"
export CK_VIDEOS_DIR="/blue/mzding/pliu1/DATA/Emo_Video_Sets/ckvideo"
export CK_HUMAN_RATINGS_CSV="/blue/mzding/pliu1/DATA/Emo_Video_Sets/CowenKeltnerEmotionalVideos.csv"

LLM="gpt-5.4"; MODEL="gpt-5.4"
DIMENSIONS="${DIMENSIONS:-va}"          # va | ck_full
CONCURRENCY="${CONCURRENCY:-8}"
RPM="${RPM:-600}"
RATINGS_DIR="$PROJECT_ROOT/output/results_ck_wholevideo/ratings"

cd "$PROJECT_ROOT"
source venv/bin/activate
pip install --quiet --disable-pip-version-check openai opencv-python-headless pillow pandas tqdm numpy scipy matplotlib

export PYTHONPATH="$PROJECT_ROOT"
source "$PROJECT_ROOT/.env"
export OPENAI_API_KEY="$GPT_NAVIGATOR_API_KEY"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$GPT_NAVIGATOR_BASE_URL}"
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: GPT_NAVIGATOR_API_KEY not set in $PROJECT_ROOT/.env"; exit 1
fi

echo "=== CK whole-video rating: $LLM, dims=$DIMENSIONS -> $RATINGS_DIR ==="
python3 scripts/run_ck_whole_video_rating.py \
    --llm "$LLM" --model "$MODEL" \
    --dimensions "$DIMENSIONS" \
    --image-detail low \
    --temperature 0.4 \
    --concurrency "$CONCURRENCY" --rpm "$RPM" \
    --ratings-dir "$RATINGS_DIR/" \
    --resume

echo "=== analysis: LLM vs human ground truth ==="
python3 scripts/analyze_ck_whole_video.py --ratings "$RATINGS_DIR/${LLM}_ratings.csv" --llm "$LLM"
echo "Done!"
