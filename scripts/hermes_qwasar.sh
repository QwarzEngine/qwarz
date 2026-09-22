#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_BIN="${HERMES_BIN:-hermes}"

python3 "$ROOT/scripts/configure_hermes.py"

if ! curl -fsS --max-time 2 http://127.0.0.1:8800/health | grep -q '"status":"ready"'; then
  echo "Qwarz is not reachable at http://127.0.0.1:8800; start it first:" >&2
  echo "  python3 $ROOT/scripts/qwasar.py start" >&2
  exit 1
fi

exec "$HERMES_BIN" chat --provider qwasar -m qwasar-qwen38-27b --reasoning xhigh "$@"
