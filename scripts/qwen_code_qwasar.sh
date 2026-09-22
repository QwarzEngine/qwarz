#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_BIN="${QWEN_BIN:-qwen}"

python3 "$ROOT/scripts/configure_qwen_code.py"

if ! curl -fsS --max-time 2 http://127.0.0.1:8800/health | grep -q '"status":"ready"'; then
  echo "Qwarz is not reachable at http://127.0.0.1:8800; start it first:" >&2
  echo "  python3 $ROOT/scripts/qwasar.py start" >&2
  exit 1
fi

export QWASAR_API_KEY="${QWASAR_API_KEY:-qwasar-local}"
exec "$QWEN_BIN" --model qwasar-qwen38-27b "$@"
