import argparse
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "integrations/claude/env").read_text()
OWNED = "ANTHROPIC_BASE_URL=http://127.0.0.1:8800"


def configure(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    if original is not None and original.decode() == ENV:
        return None
    if original is not None and OWNED not in original.decode() and "ANTHROPIC_BASE_URL=" in original.decode():
        raise ValueError("Existing Claude Code env points at a different base URL; refusing to overwrite it")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if original is not None:
        backup = target.with_name(f"{target.name}.qwasar-backup-{time.time_ns()}")
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(original)
            output.flush()
            os.fsync(output.fileno())
    payload = ENV if ENV.endswith("\n") else ENV + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".qwasar-claude-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def main():
    parser = argparse.ArgumentParser(
        description="Write Claude Code env for Qwarz without changing ~/.claude/settings.json"
    )
    parser.add_argument("--target", type=Path, default=Path.home() / ".claude/qwasar.env")
    arguments = parser.parse_args()
    backup = configure(arguments.target)
    print(f"Claude Code Qwarz env configured: {arguments.target}")
    if backup is not None:
        print(f"Original configuration backup: {backup}")


if __name__ == "__main__":
    main()
