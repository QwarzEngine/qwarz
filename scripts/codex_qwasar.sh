#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_BIN="${CODEX_BIN:-codex}"

python3 "$ROOT/scripts/configure_codex.py"

if ! curl -fsS --max-time 2 http://127.0.0.1:8800/health | grep -q '"status":"ready"'; then
  echo "Qwarz is not reachable at http://127.0.0.1:8800; start it first:" >&2
  echo "  python3 $ROOT/scripts/qwasar.py start" >&2
  exit 1
fi

exec "$CODEX_BIN" -p qwasar "$@"
