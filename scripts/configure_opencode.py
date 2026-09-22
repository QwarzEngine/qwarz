import argparse
import json
import os
from pathlib import Path
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
PROVIDER = json.loads((ROOT / "integrations/opencode/opencode.json").read_text())["provider"]["qwasar"]
OWNED_URLS = {"http://127.0.0.1:8800/v1", "http://localhost:8800/v1"}


def _base_url(existing):
    if not isinstance(existing, dict):
        return None
    options = existing.get("options")
    if not isinstance(options, dict):
        return None
    return options.get("baseURL")


def _same_qwasar(existing):
    return (
        isinstance(existing, dict)
        and existing.get("npm", "@ai-sdk/openai-compatible") == "@ai-sdk/openai-compatible"
        and _base_url(existing) in OWNED_URLS
    )


def configure(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    if original is None:
        config = {"$schema": "https://opencode.ai/config.json"}
    else:
        config = json.loads(original)
    if not isinstance(config, dict):
        raise ValueError("OpenCode config must contain a JSON object")
    providers = config.setdefault("provider", {})
    if not isinstance(providers, dict):
        raise ValueError("OpenCode provider must be an object")
    existing = providers.get("qwasar")
    if existing == PROVIDER:
        return None
    if existing is not None and not _same_qwasar(existing):
        raise ValueError("Existing qwasar provider differs; refusing to overwrite it")
    providers["qwasar"] = json.loads(json.dumps(PROVIDER))

    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if original is not None:
        backup = target.with_name(f"{target.name}.qwasar-backup-{time.time_ns()}")
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(original)
            output.flush()
            os.fsync(output.fileno())
    descriptor, temporary = tempfile.mkstemp(prefix=".qwasar-opencode-", dir=target.parent)
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
    parser = argparse.ArgumentParser(description="Register Qwarz in OpenCode without changing other providers")
    parser.add_argument("--target", type=Path, default=Path.home() / ".config/opencode/opencode.json")
    arguments = parser.parse_args()
    backup = configure(arguments.target)
    print(f"OpenCode Qwarz provider configured: {arguments.target}")
    if backup is not None:
        print(f"Original configuration backup: {backup}")


if __name__ == "__main__":
    main()
