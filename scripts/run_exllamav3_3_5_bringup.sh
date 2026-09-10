#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export QWASAR_MANIFEST="${QWASAR_MANIFEST:-$ROOT_DIR/benchmarks/manifests/qwen38-27b-rtx5090-exl3-3.5-bringup-v1.json}"
export QWASAR_MODEL_PATH="${QWASAR_MODEL_PATH:-/home/rekeyea/models/Qwen3.8-27B-EXL3-3.5bpw}"
export QWASAR_PORT="${QWASAR_PORT:-8004}"
export QWASAR_REUSE_BACKEND="${QWASAR_REUSE_BACKEND:-1}"
export QWASAR_QUALIFICATION="bringup"
export QWASAR_BACKEND_LABEL="exllamav3-3.5bpw"
export QWASAR_RUN_LABEL="${QWASAR_RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)-exllamav3-3.5bpw-bringup}"

exec "$ROOT_DIR/scripts/run_exllamav3_baseline.sh"
