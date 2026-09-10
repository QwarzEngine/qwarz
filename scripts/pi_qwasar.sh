#!/usr/bin/env bash
set -euo pipefail
PI_BIN="${PI_BIN:-$HOME/.local/share/mise/installs/pi/0.84.4/pi/pi}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! curl --fail --silent --max-time 2 http://127.0.0.1:8800/health | grep -q '"status":"ready"'; then
    echo "Qwasar is not ready. Run: python3 $ROOT/scripts/qwasar.py start" >&2
    exit 1
fi
if [[ ! -x "$PI_BIN" ]]; then
    echo 'Pi 0.84.4 is not installed at the tested path. Set PI_BIN to your Pi executable.' >&2
    exit 1
fi
export PI_OFFLINE="${PI_OFFLINE:-1}"
exec "$PI_BIN" --provider qwasar --model qwasar-qwen38-27b --thinking medium "$@"
