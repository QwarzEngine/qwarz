#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DROID_BIN="${DROID_BIN:-droid}"

python3 "$ROOT/scripts/configure_droid.py"

if ! curl -fsS http://127.0.0.1:8800/health >/dev/null; then
  echo "Qwasar is not reachable at http://127.0.0.1:8800; start it first:" >&2
  echo "  python3 $ROOT/scripts/qwasar.py start" >&2
  exit 1
fi

exec "$DROID_BIN" "$@"
