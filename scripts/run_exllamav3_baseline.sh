#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${QWASAR_MANIFEST:-$ROOT_DIR/benchmarks/manifests/qwen38-27b-rtx5090-v1.json}"
FIXTURES="${QWASAR_FIXTURES:-$ROOT_DIR/benchmarks/fixtures/smoke.jsonl}"
MODEL_DIR="${QWASAR_MODEL_PATH:-/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw}"
DRAFT_DIR="${QWASAR_DRAFT_PATH:-/home/rekeyea/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw}"
EXLLAMA_HOME="${QWASAR_EXLLAMA_HOME:-/home/rekeyea/Documents/llm/qwen38-exl3-mia}"
EXLLAMA_PYTHON="${QWASAR_EXLLAMA_PYTHON:-$EXLLAMA_HOME/.venv/bin/python}"
HOST="${QWASAR_HOST:-127.0.0.1}"
PORT="${QWASAR_PORT:-8890}"
CUDA_DEVICE="${QWASAR_CUDA_DEVICE:-0}"
REUSE_BACKEND="${QWASAR_REUSE_BACKEND:-0}"
QUALIFICATION="${QWASAR_QUALIFICATION:-acceptance}"
BACKEND_LABEL="${QWASAR_BACKEND_LABEL:-exllamav3-5bpw}"
RUN_LABEL="${QWASAR_RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)-exllamav3-5bpw}"
OUTPUT_DIR="${QWASAR_OUTPUT_DIR:-$ROOT_DIR/results/$RUN_LABEL}"
BASE_URL="http://$HOST:$PORT"
BACKEND_LOG="$(mktemp --tmpdir qwasar-exllamav3.XXXXXX.log)"
STARTED_BACKEND=0
BACKEND_PID=""

cleanup() {
    local status=$?
    if [[ "$STARTED_BACKEND" == "1" && -n "$BACKEND_PID" ]]; then
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
    if [[ -d "$OUTPUT_DIR" ]]; then
        mv "$BACKEND_LOG" "$OUTPUT_DIR/backend.log"
    elif [[ -f "$BACKEND_LOG" ]]; then
        printf 'Backend log preserved at %s\n' "$BACKEND_LOG" >&2
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Required file not found: %s\n' "$1" >&2
        exit 2
    fi
}

require_file "$MANIFEST"
require_file "$FIXTURES"
require_file "$MODEL_DIR/config.json"
require_file "$MODEL_DIR/tokenizer.json"
require_file "$DRAFT_DIR/config.json"
require_file "$EXLLAMA_HOME/tools/serve_openai.py"
require_file "$EXLLAMA_PYTHON"
printf '%s  %s\n' \
    c696c060b2a58fbc989b8145e81f0fb8c4745dfa412ae441445ee3fe86363f3a \
    "$EXLLAMA_HOME/tools/serve_openai.py" | sha256sum --check --status

export QWASAR_MODEL_PATH="$MODEL_DIR"
export QWASAR_CUDA_DEVICE="$CUDA_DEVICE"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR"
UV_CACHE_DIR="$ROOT_DIR/.uv-cache" uv run python -m qwasar_bench doctor \
    --manifest "$MANIFEST" \
    --qualification "$QUALIFICATION"

if curl --fail --silent --show-error "$BASE_URL/health" >/dev/null 2>&1; then
    if [[ "$REUSE_BACKEND" != "1" ]]; then
        printf 'A backend already answers at %s. Set QWASAR_REUSE_BACKEND=1 only after verifying its exact model and launch parameters.\n' "$BASE_URL" >&2
        exit 2
    fi
    printf 'Using the explicitly approved existing backend at %s; its external log is not managed by Qwasar.\n' "$BASE_URL"
else
    (
        cd "$EXLLAMA_HOME"
        exec env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$EXLLAMA_PYTHON" -u tools/serve_openai.py \
            --model "$MODEL_DIR" \
            --draft_model "$DRAFT_DIR" \
            --host "$HOST" \
            --port "$PORT" \
            --cache_size 262144 \
            --grid_size 30 \
            --cache_quant nvfp4
    ) >"$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
    STARTED_BACKEND=1

    ready=0
    for _ in $(seq 1 180); do
        if curl --fail --silent --show-error "$BASE_URL/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            printf 'ExLlamaV3 exited during startup.\n' >&2
            tail -100 "$BACKEND_LOG" >&2
            exit 2
        fi
        sleep 1
    done
    if [[ "$ready" != "1" ]]; then
        printf 'Timed out waiting for ExLlamaV3 health endpoint.\n' >&2
        tail -100 "$BACKEND_LOG" >&2
        exit 2
    fi
fi

UV_CACHE_DIR="$ROOT_DIR/.uv-cache" uv run --extra benchmark python -m qwasar_bench baseline \
    --manifest "$MANIFEST" \
    --fixtures "$FIXTURES" \
    --base-url "$BASE_URL" \
    --api-model qwen3.8-27b \
    --backend "$BACKEND_LABEL" \
    --protocol chat-completions \
    --qualification "$QUALIFICATION" \
    --tokenizer "$MODEL_DIR/tokenizer.json" \
    --timeout 3600 \
    --output "$OUTPUT_DIR"

UV_CACHE_DIR="$ROOT_DIR/.uv-cache" uv run python -m qwasar_bench validate-run "$OUTPUT_DIR"
