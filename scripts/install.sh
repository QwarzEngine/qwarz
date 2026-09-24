#!/usr/bin/env bash
# One-command installer for Qwarz on an RTX 5090 machine.
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/QwarzEngine/qwarz/master/scripts/install.sh)
#
# Clones the engine into ~/Documents/llm/qwarz (or fast-forwards an existing
# clone), builds the qwarz CLI and hands control to `qwarz start`, which
# verifies or downloads the pinned artifacts, installs the systemd service
# and waits for the worker. Any qwarz start flag can be appended, e.g.
#
#   bash <(curl -fsSL ...) --gpu 1 --download
#
# Prerequisites this script does NOT install (qwarz start fails fast with the
# exact instruction for whatever is missing): the Rust toolchain (rustup),
# CUDA with nvcc, and the ExLlamaV3 runtime venv — normally the sibling
# qwen38-exl3-mia checkout under ~/Documents/llm.
set -euo pipefail

QWASAR_HOME="${QWASAR_HOME:-$HOME/Documents/llm}"
REPOSITORY="https://github.com/QwarzEngine/qwarz.git"
DESTINATION="$QWASAR_HOME/qwarz"

fail() { echo "qwarz install: $*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || fail "git is required"
command -v cargo >/dev/null 2>&1 || fail "cargo (Rust toolchain) is required; install it from https://rustup.rs and re-run"

if [ -d "$DESTINATION/.git" ]; then
    echo "qwarz install: updating the existing clone at $DESTINATION"
    if git -C "$DESTINATION" fetch origin master && git -C "$DESTINATION" merge --ff-only FETCH_HEAD; then
        :
    else
        echo "qwarz install: could not fast-forward (dirty tree or local commits); using the current checkout"
    fi
else
    echo "qwarz install: cloning qwarz into $DESTINATION"
    git clone "$REPOSITORY" "$DESTINATION"
fi

cargo build --release --locked --manifest-path "$DESTINATION/Cargo.toml" --package qwarz
exec "$DESTINATION/target/release/qwarz" start "$@"
