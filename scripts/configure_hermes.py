import argparse
from io import StringIO
import os
from pathlib import Path
import tempfile
import time

import yaml

try:
    from ruamel.yaml import YAML
except ImportError:
    YAML = None


ROOT = Path(__file__).resolve().parents[1]
PROVIDER = yaml.safe_load((ROOT / "integrations/hermes/provider.yaml").read_text())["qwasar"]
OWNED_URLS = {"http://127.0.0.1:8800/v1", "http://localhost:8800/v1"}
TITLE_PROVIDER = "qwen38-local"
TITLE_MODEL = "qwen3.8-27b"


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _provider_url(entry):
    if not isinstance(entry, dict):
        return None
    return entry.get("api") or entry.get("base_url") or entry.get("url")


def _same_qwasar(existing):
    url = str(_provider_url(existing) or "").rstrip("/")
    return url in OWNED_URLS and existing.get("transport", "chat_completions") == "chat_completions"


def _load(original: bytes | None):
    if original is None:
        return {}
    if YAML is not None:
        parser = YAML()
        parser.preserve_quotes = True
        loaded = parser.load(original.decode())
        return loaded if loaded is not None else {}
    return yaml.safe_load(original) or {}


def _dump(config) -> str:
    if YAML is not None:
        parser = YAML()
        parser.preserve_quotes = True
        parser.width = 4096
        parser.indent(mapping=2, sequence=4, offset=2)
        buffer = StringIO()
        parser.dump(config, buffer)
        return buffer.getvalue()
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def _pin_title_off_qwasar(config) -> bool:
    providers = config.get("providers")
    if not isinstance(providers, dict) or TITLE_PROVIDER not in providers:
        return False
    auxiliary = config.setdefault("auxiliary", {})
    if not isinstance(auxiliary, dict):
        return False
    title = auxiliary.setdefault("title_generation", {})
    if not isinstance(title, dict):
        return False
    provider = str(title.get("provider") or "auto").strip().lower()
    if provider not in ("", "auto", "qwasar"):
        return False
    if title.get("provider") == TITLE_PROVIDER and title.get("model") == TITLE_MODEL:
        return False
    title["provider"] = TITLE_PROVIDER
    title["model"] = TITLE_MODEL
    return True


def configure(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    config = _load(original)
    if not isinstance(config, dict):
        raise ValueError("Hermes config.yaml must contain a YAML object")
    providers = config.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("Hermes providers must be a mapping")
    existing = providers.get("qwasar")
    provider_changed = _plain(existing) != PROVIDER
    if existing is not None and not _same_qwasar(existing) and provider_changed:
        raise ValueError("Existing qwasar provider differs; refusing to overwrite it")
    if provider_changed:
        providers["qwasar"] = yaml.safe_load(yaml.safe_dump(PROVIDER))
    title_changed = _pin_title_off_qwasar(config)
    if not provider_changed and not title_changed:
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
    payload = _dump(config)
    descriptor, temporary = tempfile.mkstemp(prefix=".qwasar-hermes-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(payload)
            if not payload.endswith("\n"):
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def main():
    parser = argparse.ArgumentParser(description="Register Qwarz in Hermes without changing other providers")
    parser.add_argument("--target", type=Path, default=Path.home() / ".hermes/config.yaml")
    arguments = parser.parse_args()
    backup = configure(arguments.target)
    print(f"Hermes Qwarz provider configured: {arguments.target}")
    if backup is not None:
        print(f"Original configuration backup: {backup}")


if __name__ == "__main__":
    main()
