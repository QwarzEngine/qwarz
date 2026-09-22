import argparse
import json
import os
from pathlib import Path
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "integrations/codex/catalog.json").read_text())
MODEL = CATALOG["models"][0]
PROVIDER_BLOCK = """
[model_providers.qwasar]
name = "Qwarz"
base_url = "http://127.0.0.1:8800/v1"
wire_api = "responses"
requires_openai_auth = false
stream_idle_timeout_ms = 600000
"""


def _atomic_write(target: Path, data: bytes, backup_prefix: str, original: bytes | None) -> Path | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if original is not None:
        backup = target.with_name(f"{target.name}.qwasar-backup-{time.time_ns()}")
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(original)
            output.flush()
            os.fsync(output.fileno())
    descriptor, temporary = tempfile.mkstemp(prefix=backup_prefix, dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def _owned_entry(existing):
    return (
        isinstance(existing, dict)
        and existing.get("slug") == MODEL["slug"]
        and existing.get("shell_type") == MODEL["shell_type"]
        and existing.get("context_window") == MODEL["context_window"]
    )


def _desired_catalog(models):
    index = next((i for i, entry in enumerate(models) if isinstance(entry, dict) and entry.get("slug") == MODEL["slug"]), None)
    if index is not None and not _owned_entry(models[index]):
        raise ValueError("Existing qwasar-qwen38-27b catalog entry differs; refusing to overwrite it")
    filtered = [entry for entry in models if not (isinstance(entry, dict) and entry.get("slug") == MODEL["slug"])]
    filtered.insert(0, dict(MODEL))
    return filtered


def configure_catalog(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    config = json.loads(original) if original is not None else {"models": []}
    if not isinstance(config, dict):
        raise ValueError("Codex catalog must contain a JSON object")
    models = config.setdefault("models", [])
    if not isinstance(models, list):
        raise ValueError("catalog models must be an array")
    desired = _desired_catalog(models)
    if models == desired:
        return None
    config["models"] = desired
    return _atomic_write(target, (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode(), ".qwasar-catalog-", original)


def _profile_text(catalog: Path) -> str:
    template = (ROOT / "integrations/codex/qwasar.config.toml").read_text()
    return template.replace("CATALOG_JSON", str(catalog))


def configure_profile(target: Path, catalog: Path) -> Path | None:
    desired = _profile_text(catalog)
    original = target.read_bytes() if target.exists() else None
    if original is None:
        return _atomic_write(target, desired.encode(), ".qwasar-profile-", None)
    text = original.decode()
    has_provider = 'model_provider = "qwasar"' in text
    has_model = 'model = "qwasar-qwen38-27b"' in text
    if not has_provider and not has_model and "model_provider =" not in text and "\nmodel =" not in text and not text.startswith("model ="):
        return _atomic_write(target, desired.encode(), ".qwasar-profile-", original)
    if not has_provider or not has_model:
        raise ValueError("Existing qwasar profile uses a different provider or model; refusing to overwrite it")
    catalog_line = f'model_catalog_json = "{catalog}"'
    if catalog_line in text:
        return None
    if "model_catalog_json =" in text:
        updated = "\n".join(
            catalog_line if line.startswith("model_catalog_json") else line
            for line in text.splitlines()
        ) + ("\n" if text.endswith("\n") else "")
    else:
        updated = text.rstrip() + "\n" + catalog_line + "\n"
    if updated == text:
        return None
    return _atomic_write(target, updated.encode(), ".qwasar-profile-", original)


def configure_provider(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    text = original.decode() if original is not None else ""
    if "[model_providers.qwasar]" in text:
        if "http://127.0.0.1:8800/v1" not in text.split("[model_providers.qwasar]", 1)[1].split("\n[", 1)[0]:
            raise ValueError("Existing qwasar provider differs; refusing to overwrite it")
        return None
    if original is None:
        body = 'model_provider = "qwasar"\nmodel = "qwasar-qwen38-27b"\n' + PROVIDER_BLOCK
        return _atomic_write(target, body.encode(), ".qwasar-config-", None)
    updated = text.rstrip() + "\n" + PROVIDER_BLOCK
    if not updated.endswith("\n"):
        updated += "\n"
    return _atomic_write(target, updated.encode(), ".qwasar-config-", original)


def configure(codex_home: Path) -> dict[str, Path | None]:
    catalog = codex_home / "qwasar-catalog.json"
    return {
        "catalog": configure_catalog(catalog),
        "profile": configure_profile(codex_home / "qwasar.config.toml", catalog),
        "provider": configure_provider(codex_home / "config.toml"),
    }


def main():
    parser = argparse.ArgumentParser(description="Register Qwarz in Codex with native image input")
    parser.add_argument("--codex-home", type=Path, default=Path.home() / ".codex")
    arguments = parser.parse_args()
    backups = configure(arguments.codex_home)
    print(f"Codex Qwarz catalog configured: {arguments.codex_home / 'qwasar-catalog.json'}")
    for name, backup in backups.items():
        if backup is not None:
            print(f"Original {name} backup: {backup}")


if __name__ == "__main__":
    main()
