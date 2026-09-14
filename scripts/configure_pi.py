import argparse
import json
import os
from pathlib import Path
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def _same_qwasar(existing, provider):
    return (isinstance(existing, dict) and existing.get("baseUrl") == provider["baseUrl"]
            and existing.get("api") == provider["api"])


def configure(target: Path) -> Path | None:
    provider = json.loads((ROOT / "integrations/pi/models.json").read_text())["providers"]["qwasar"]
    original = target.read_bytes() if target.exists() else None
    config = json.loads(original) if original is not None else {}
    if not isinstance(config, dict) or not isinstance(config.get("providers", {}), dict):
        raise ValueError("Pi models.json must contain an object with object providers")
    providers = config.setdefault("providers", {})
    existing = providers.get("qwasar")
    if existing == provider:
        return None
    if existing is not None and not _same_qwasar(existing, provider):
        raise ValueError("Existing qwasar provider differs; refusing to overwrite it")
    providers["qwasar"] = provider
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if original is not None:
        backup = target.with_name(f"{target.name}.qwasar-backup-{time.time_ns()}")
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(original)
            output.flush()
            os.fsync(output.fileno())
    descriptor, temporary = tempfile.mkstemp(prefix=".qwasar-models-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(config, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def main():
    parser = argparse.ArgumentParser(description="Register Qwasar in Pi without changing other providers")
    parser.add_argument("--target", type=Path, default=Path.home() / ".pi/agent/models.json")
    arguments = parser.parse_args()
    backup = configure(arguments.target)
    print(f"Pi provider configured: {arguments.target}")
    if backup is not None:
        print(f"Original configuration backup: {backup}")


if __name__ == "__main__":
    main()
