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
KEYFRAMES_DIR="${KEYFRAMES_DIR:-output/keyframes}"

find_python() {
    if [ -n "${PYTHON:-}" ]; then
        if "$PYTHON" -c "import sys" >/dev/null 2>&1; then
            printf '%s\n' "$PYTHON"
            return 0
        fi
        echo "PYTHON is set but is not runnable: $PYTHON" >&2
        return 1
    fi
    for candidate in python python3 py; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import sys" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "Could not find a runnable Python. Set PYTHON=/path/to/python and rerun." >&2
    return 1
}

PY="$(find_python)"

export PYTHONPATH="$PWD"

configure_cert_bundle() {
    local cert_path
    cert_path="$("$PY" - <<'PY'
try:
    import certifi
    print(certifi.where())
except Exception:
    print("")
PY
)"
    if [ -n "$cert_path" ] && [ -f "$cert_path" ]; then
        export SSL_CERT_FILE="$cert_path"
        export REQUESTS_CA_BUNDLE="$cert_path"
        echo "=== SSL cert bundle: $SSL_CERT_FILE ==="
    elif [ -n "${SSL_CERT_FILE:-}" ] && [ ! -f "$SSL_CERT_FILE" ]; then
        echo "=== ignoring missing SSL_CERT_FILE: $SSL_CERT_FILE ==="
        unset SSL_CERT_FILE
        unset REQUESTS_CA_BUNDLE
    fi
}

configure_cert_bundle

VIDEOS_ARG=()
if [ "$#" -gt 0 ]; then
    for video_id in "$@"; do
        if [[ ! "$video_id" =~ ^[0-9]{1,4}(\.mp4)?$ ]]; then
            echo "Invalid video id argument: $video_id" >&2
            echo "Use video ids like: ./run_local.sh 0001 0002" >&2
            echo "Set paths in .env or export CK_VIDEOS_DIR before the command, not after it." >&2
            exit 2
        fi
    done
    VIDEOS_ARG=(--videos "$@")
fi

echo "=== CK whole-video rating: $LLM dims=$DIMENSIONS -> $RATINGS_DIR ==="
echo "=== keeping keyframes in $KEYFRAMES_DIR ==="
echo "=== python: $PY ==="
"$PY" scripts/run_ck_whole_video_rating.py \
    --llm "$LLM" --model "$LLM" \
    --dimensions "$DIMENSIONS" \
    --image-detail low --temperature 0.4 \
    --concurrency "$CONCURRENCY" --rpm "$RPM" \
    --ratings-dir "$RATINGS_DIR" \
    --keyframes-dir "$KEYFRAMES_DIR" \
    "${VIDEOS_ARG[@]}" \
    --resume

echo "=== analysis: LLM vs human ground truth ==="
"$PY" scripts/analyze_ck_whole_video.py --ratings "$RATINGS_DIR/${LLM}_ratings.csv" --llm "$LLM"
echo "Done. Ratings in $RATINGS_DIR/, figures in doc/figures/"
