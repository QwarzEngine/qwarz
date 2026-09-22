#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
ENV_FILE="${CLAUDE_QWASAR_ENV:-$HOME/.claude/qwasar.env}"

python3 "$ROOT/scripts/configure_claude.py" --target "$ENV_FILE"

if ! curl -fsS --max-time 2 http://127.0.0.1:8800/health | grep -q '"status":"ready"'; then
  echo "Qwarz is not reachable at http://127.0.0.1:8800; start it first:" >&2
  echo "  python3 $ROOT/scripts/qwasar.py start" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

exec "$CLAUDE_BIN" "$@"
