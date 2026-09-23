#!/bin/bash
# Run Yujun's CK whole-video affective-dimension rating locally.
# Usage:
#   ./run_local_yujun.sh                    # extract frames + rate ALL videos
#   TEST_MODE=1 ./run_local_yujun.sh        # one API request only
#   ./run_local_yujun.sh 0001 0002 0003     # rate only these clip ids
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && { set -a; source .env; set +a; }
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (see .env.example)}"

LLM="${LLM:-gpt-5.4}"
MODEL="${MODEL:-$LLM}"
CONCURRENCY="${CONCURRENCY:-4}"
RPM="${RPM:-120}"
IMAGE_DETAIL="${IMAGE_DETAIL:-low}"
TEMPERATURE="${TEMPERATURE:-0.4}"
FRAMES_DIR="${FRAMES_DIR:-output/ck/frames_yujun}"
RATINGS_DIR="${RATINGS_DIR:-output/ck/ratings}"

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
if [ "$#" -gt 0 ]; then VIDEOS_ARG=(--videos "$@"); fi

TEST_ARG=()
if [ "${TEST_MODE:-0}" = "1" ] || [ "${TEST_MODE:-false}" = "true" ]; then
    TEST_ARG=(--test-mode)
fi

VIDEO_DIR_ARG=()
if [ -n "${VIDEO_DIR:-}" ]; then
    VIDEO_DIR_ARG=(--video-dir "$VIDEO_DIR")
elif [ -n "${CK_VIDEOS_DIR:-}" ]; then
    VIDEO_DIR_ARG=(--video-dir "$CK_VIDEOS_DIR")
fi

OVERWRITE_FRAMES_ARG=()
if [ "${OVERWRITE_FRAMES:-0}" = "1" ] || [ "${OVERWRITE_FRAMES:-false}" = "true" ]; then
    OVERWRITE_FRAMES_ARG=(--overwrite-frames)
fi

echo "=== CK whole-video affective14 Yujun rating: $MODEL -> $RATINGS_DIR ==="
echo "=== videos -> all generated frames for each video -> OpenAI | frames: $FRAMES_DIR ==="
echo "=== python: $PY ==="
"$PY" scripts/run_ck_whole_video_affective_yujun.py \
    --llm "$LLM" --model "$MODEL" \
    "${VIDEO_DIR_ARG[@]}" \
    --frames-dir "$FRAMES_DIR" \
    --ratings-dir "$RATINGS_DIR" \
    --image-detail "$IMAGE_DETAIL" \
    --temperature "$TEMPERATURE" \
    --concurrency "$CONCURRENCY" --rpm "$RPM" \
    "${VIDEOS_ARG[@]}" \
    "${TEST_ARG[@]}" \
    "${OVERWRITE_FRAMES_ARG[@]}" \
    --resume

echo "Done. Ratings in $RATINGS_DIR/${LLM}_affective14_yujun.csv"
