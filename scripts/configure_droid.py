import argparse
import json
import os
from pathlib import Path
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
QWASAR_MODEL = json.loads((ROOT / "integrations/droid/customModels.json").read_text())[0]


def _same_qwasar(existing):
    return (
        isinstance(existing, dict)
        and existing.get("model") == QWASAR_MODEL["model"]
        and existing.get("baseUrl") == QWASAR_MODEL["baseUrl"]
        and existing.get("provider") == QWASAR_MODEL["provider"]
    )


def _find_qwasar(models):
    for index, entry in enumerate(models):
        if isinstance(entry, dict) and entry.get("model") == QWASAR_MODEL["model"]:
            return index, entry
    return None, None


def _is_legacy_local_qwen(entry):
    if not isinstance(entry, dict):
        return False
    if entry.get("model") not in {"qwen3.8-27b", "qwen3.8-27b-exl3-3.5bpw-wm"}:
        return False
    base_url = entry.get("baseUrl", "")
    return base_url.startswith("http://127.0.0.1:8004") or base_url.startswith("http://localhost:8004")


def _desired_models(models):
    index, existing = _find_qwasar(models)
    if existing is not None and not _same_qwasar(existing):
        raise ValueError("Existing qwasar-qwen38-27b entry differs; refusing to overwrite it")

    filtered = [entry for entry in models if not _is_legacy_local_qwen(entry)]
    if existing is not None:
        filtered = [entry for entry in filtered if entry.get("model") != QWASAR_MODEL["model"]]

    qwasar_entry = dict(QWASAR_MODEL)
    qwasar_entry["index"] = 0
    filtered.insert(0, qwasar_entry)
    for offset, entry in enumerate(filtered[1:], start=1):
        if isinstance(entry, dict):
            entry["index"] = offset
    return filtered


def configure(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    config = json.loads(original) if original is not None else {}
    if not isinstance(config, dict):
        raise ValueError("Droid settings.json must contain a JSON object")

    models = config.setdefault("customModels", [])
    if not isinstance(models, list):
        raise ValueError("customModels must be an array")

    desired = _desired_models(models)
    if models == desired:
        return None
    config["customModels"] = desired

    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if original is not None:
        backup = target.with_name(f"{target.name}.qwasar-backup-{time.time_ns()}")
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(original)
            output.flush()
            os.fsync(output.fileno())

    descriptor, temporary = tempfile.mkstemp(prefix=".qwasar-settings-", dir=target.parent)
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
    parser = argparse.ArgumentParser(description="Register Qwasar in Droid without changing other custom models")
    parser.add_argument("--target", type=Path, default=Path.home() / ".factory/settings.json")
    arguments = parser.parse_args()
    backup = configure(arguments.target)
    print(f"Droid custom model configured: {arguments.target}")
    if backup is not None:
        print(f"Original configuration backup: {backup}")


if __name__ == "__main__":
    main()
