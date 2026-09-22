import argparse
import json
import os
from pathlib import Path
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SNIPPET = json.loads((ROOT / "integrations/qwen-code/settings.json").read_text())
MODEL = SNIPPET["modelProviders"]["openai"][0]
OWNED_URLS = {"http://127.0.0.1:8800/v1", "http://localhost:8800/v1"}
PREFERRED_FAST = ("spark-x2.5-4b", "qwen3.8-27b")


def _same_qwasar(existing):
    return isinstance(existing, dict) and existing.get("baseUrl") in OWNED_URLS


def _find_qwasar(models):
    for index, entry in enumerate(models):
        if isinstance(entry, dict) and entry.get("id") == MODEL["id"]:
            return index, entry
    return None, None


def _is_qwasar_model_id(value):
    name = str(value or "").strip()
    if not name:
        return False
    return name == MODEL["id"] or name.endswith(f":{MODEL['id']}") or name.endswith(f"/{MODEL['id']}")


def _fast_candidate(models):
    ids = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        identity = entry.get("id")
        if not isinstance(identity, str) or not identity or _is_qwasar_model_id(identity):
            continue
        if entry.get("baseUrl") in OWNED_URLS:
            continue
        ids.append(identity)
    for preferred in PREFERRED_FAST:
        if preferred in ids:
            return preferred
    return ids[0] if ids else None


def _pin_fast_off_qwasar(config, models):
    current = config.get("fastModel")
    if isinstance(current, str) and current.strip() and not _is_qwasar_model_id(current):
        return False
    candidate = _fast_candidate(models)
    if candidate is None or current == candidate:
        return False
    config["fastModel"] = candidate
    return True


def configure(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    if original is None:
        config = {}
    else:
        config = json.loads(original)
    if not isinstance(config, dict):
        raise ValueError("Qwen Code settings.json must contain a JSON object")

    env = config.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError("Qwen Code env must be an object")
    providers = config.setdefault("modelProviders", {})
    if not isinstance(providers, dict):
        raise ValueError("Qwen Code modelProviders must be an object")
    models = providers.setdefault("openai", [])
    if not isinstance(models, list):
        raise ValueError("Qwen Code modelProviders.openai must be an array")

    index, existing = _find_qwasar(models)
    if existing is not None and not _same_qwasar(existing) and existing != MODEL:
        raise ValueError("Existing qwasar-qwen38-27b entry differs; refusing to overwrite it")

    desired_models = list(models)
    provider_changed = existing != MODEL or env.get("QWASAR_API_KEY") != SNIPPET["env"]["QWASAR_API_KEY"]
    if provider_changed:
        if index is None:
            desired_models.append(json.loads(json.dumps(MODEL)))
        else:
            desired_models[index] = json.loads(json.dumps(MODEL))
        providers["openai"] = desired_models
        env["QWASAR_API_KEY"] = SNIPPET["env"]["QWASAR_API_KEY"]

    fast_changed = _pin_fast_off_qwasar(config, providers["openai"])
    if not provider_changed and not fast_changed:
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if original is not None:
        backup = target.with_name(f"{target.name}.qwasar-backup-{time.time_ns()}")
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(original)
            output.flush()
            os.fsync(output.fileno())
    descriptor, temporary = tempfile.mkstemp(prefix=".qwasar-qwen-code-", dir=target.parent)
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
    parser = argparse.ArgumentParser(description="Register Qwarz in Qwen Code without changing other models")
    parser.add_argument("--target", type=Path, default=Path.home() / ".qwen/settings.json")
    arguments = parser.parse_args()
    backup = configure(arguments.target)
    print(f"Qwen Code Qwarz model configured: {arguments.target}")
    if backup is not None:
        print(f"Original configuration backup: {backup}")


if __name__ == "__main__":
    main()
