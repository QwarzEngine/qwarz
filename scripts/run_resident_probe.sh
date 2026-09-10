#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${QWASAR_MODEL_PATH:-/home/rekeyea/models/Qwen3.8-27B-EXL3-3.5bpw}"
DRAFT_DIR="${QWASAR_DRAFT_PATH:-/home/rekeyea/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw}"
EXLLAMA_HOME="${QWASAR_EXLLAMA_HOME:-/home/rekeyea/Documents/llm/qwen38-exl3-mia}"
EXLLAMA_PYTHON="${QWASAR_EXLLAMA_PYTHON:-$EXLLAMA_HOME/.venv/bin/python}"
CUDA_DEVICE="${QWASAR_CUDA_DEVICE:-0}"
OUTPUT_DIR="${QWASAR_OUTPUT_DIR:-$ROOT_DIR/results/$(date -u +%Y%m%dT%H%M%SZ)-exl3-resident-probe}"
ALLOW_BUSY_GPU="${QWASAR_ALLOW_BUSY_GPU:-0}"
PROBE_KIND="${QWASAR_PROBE_KIND:-resident}"
DRAFT_METHOD="${QWASAR_DRAFT_METHOD:-dflash2}"

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Required file not found: %s\n' "$1" >&2
        exit 2
    fi
}

require_file "$MODEL_DIR/config.json"
if [[ "$PROBE_KIND" == "resident" || "$DRAFT_METHOD" == "dflash2" ]]; then
    require_file "$DRAFT_DIR/config.json"
fi
require_file "$EXLLAMA_PYTHON"

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
    printf 'Refusing to overwrite completed probe: %s\n' "$OUTPUT_DIR" >&2
    exit 2
fi

gpu_row="$(nvidia-smi \
    --query-gpu=index,uuid,name \
    --format=csv,noheader,nounits | awk -F ', ' -v gpu_index="$CUDA_DEVICE" '$1 == gpu_index { print; exit }')"
if [[ -z "$gpu_row" ]]; then
    printf 'GPU index %s is not available.\n' "$CUDA_DEVICE" >&2
    exit 2
fi

IFS=', ' read -r selected_index selected_uuid selected_name <<<"$gpu_row"
if [[ "$selected_name" != *"RTX 5090"* ]]; then
    printf 'GPU %s is %s, not an RTX 5090.\n' "$selected_index" "$selected_name" >&2
    exit 2
fi

busy_processes="$(nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits | awk -F ', ' -v uuid="$selected_uuid" \
    '$1 == uuid && ($4 + 0) >= 512 { print }')"
if [[ -n "$busy_processes" && "$ALLOW_BUSY_GPU" != "1" ]]; then
    printf 'RTX 5090 already has a compute workload using at least 512 MiB:\n%s\n' \
        "$busy_processes" >&2
    exit 2
fi

export QWASAR_CUDA_DEVICE="$CUDA_DEVICE"
export QWASAR_EXLLAMA_HOME="$EXLLAMA_HOME"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

cd "$ROOT_DIR"
if [[ "$PROBE_KIND" == "decode" ]]; then
    cache_args=()
    if [[ -n "${QWASAR_CACHE_SIZE:-}" ]]; then
        cache_args=(--cache-size "$QWASAR_CACHE_SIZE")
    fi
    mtp_args=()
    if [[ -n "${QWASAR_MTP_POLICY:-}" ]]; then
        mtp_args=(--mtp-policy "$QWASAR_MTP_POLICY")
    fi
    prefill_args=()
    if [[ -n "${QWASAR_PREFILL_SETTINGS:-}" ]]; then
        prefill_args=(--prefill-settings "$QWASAR_PREFILL_SETTINGS")
    fi
    exec env PYTHONPATH="$ROOT_DIR/src" \
        "$EXLLAMA_PYTHON" -m qwasar_bench.decode_probe \
        "${prefill_args[@]}" \
        "${mtp_args[@]}" \
        "${cache_args[@]}" \
        --model "$MODEL_DIR" \
        --draft-model "$DRAFT_DIR" \
        --draft-method "$DRAFT_METHOD" \
        --output "$OUTPUT_DIR" \
        --corpus-root "${QWASAR_CORPUS_ROOT:-$EXLLAMA_HOME/.venv/lib/python3.12/site-packages/exllamav3}" \
        --contexts "${QWASAR_CONTEXTS:-32768}" \
        --max-new-tokens "${QWASAR_MAX_NEW_TOKENS:-512}" \
        --repetitions "${QWASAR_REPETITIONS:-2}" \
        --thinking "${QWASAR_THINKING:-medium}" \
        --sampler "${QWASAR_SAMPLER:-greedy}" \
        --workload "${QWASAR_WORKLOAD:-lru}" \
        --tool-max-new-tokens "${QWASAR_TOOL_MAX_NEW_TOKENS:-512}" \
        --appended-tokens "${QWASAR_APPENDED_TOKENS:-128,512,2048,8192}" \
        --profile-matrix "${QWASAR_PROFILE_MATRIX:-0}" \
        --source-run "${QWASAR_SOURCE_RUN:-$ROOT_DIR/results/20260904-session-5.0bpw-medium-long}" \
        --gpu-split-gb "${QWASAR_GPU_SPLIT_GB:-30}" \
        --cache-quant "${QWASAR_CACHE_QUANT:-nvfp4}"
fi
if [[ "$PROBE_KIND" != "resident" ]]; then
    printf 'Unknown probe kind: %s\n' "$PROBE_KIND" >&2
    exit 2
fi
exec env PYTHONPATH="$ROOT_DIR/src" \
    "$EXLLAMA_PYTHON" -m qwasar_bench resident-probe \
    --model "$MODEL_DIR" \
    --draft-model "$DRAFT_DIR" \
    --output "$OUTPUT_DIR" \
    --contexts "${QWASAR_CONTEXTS:-1024,32768,131072,262144}" \
    --appended-tokens "${QWASAR_APPENDED_TOKENS:-128}" \
    --max-new-tokens "${QWASAR_MAX_NEW_TOKENS:-1}" \
    --repetitions "${QWASAR_REPETITIONS:-3}" \
    --cache-size "${QWASAR_CACHE_SIZE:-262400}" \
    --gpu-split-gb "${QWASAR_GPU_SPLIT_GB:-30}" \
    --cache-quant "${QWASAR_CACHE_QUANT:-nvfp4}"
