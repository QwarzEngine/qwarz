import argparse
import os
from pathlib import Path
import tempfile
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
OWNED_URLS = {"http://127.0.0.1:8800/v1", "http://localhost:8800/v1"}
_BOOLISH = {"off", "on", "yes", "no", "true", "false", "Off", "On"}


class _Dumper(yaml.SafeDumper):
    pass


def _represent_str(dumper, data):
    style = '"' if data in _BOOLISH else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _represent_str)
PROVIDER = yaml.safe_load((ROOT / "integrations/omp/models.yml").read_text())["providers"]["qwasar"]
BACKGROUND_MODEL = "openai-codex/gpt-5.5:off"
BACKGROUND_ROLES = ("tiny", "smol")


def _dump(config) -> str:
    return yaml.dump(config, Dumper=_Dumper, sort_keys=False, allow_unicode=True)


def default_target() -> Path:
    home = Path.home() / ".omp/agent"
    for name in ("models.yml", "models.yaml"):
        candidate = home / name
        if candidate.exists():
            return candidate
    return home / "models.yml"


def default_config() -> Path:
    return Path.home() / ".omp/agent/config.yml"


def _uses_qwasar(value):
    return str(value or "").startswith("qwasar/")


def _atomic_write(target: Path, payload: str, original: bytes | None, prefix: str) -> Path | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if original is not None:
        backup = target.with_name(f"{target.name}.qwasar-backup-{time.time_ns()}")
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(original)
            output.flush()
            os.fsync(output.fileno())
    if not payload.endswith("\n"):
        payload += "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=prefix, dir=target.parent)
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


def _same_qwasar(existing):
    return (
        isinstance(existing, dict)
        and existing.get("baseUrl") in OWNED_URLS
        and existing.get("api") == PROVIDER["api"]
    )


def configure(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    if original is None:
        config = {}
    else:
        config = yaml.safe_load(original) or {}
    if not isinstance(config, dict):
        raise ValueError("OMP models.yml must contain a YAML object")
    providers = config.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("OMP providers must be a mapping")
    existing = providers.get("qwasar")
    if existing == PROVIDER:
        return None
    if existing is not None and not _same_qwasar(existing):
        raise ValueError("Existing qwasar provider differs; refusing to overwrite it")
    providers["qwasar"] = yaml.safe_load(_dump(PROVIDER))
    return _atomic_write(target, _dump(config), original, ".qwasar-omp-")


def configure_roles(target: Path) -> Path | None:
    original = target.read_bytes() if target.exists() else None
    if original is None:
        config = {}
    else:
        config = yaml.safe_load(original) or {}
    if not isinstance(config, dict):
        raise ValueError("OMP config.yml must contain a YAML object")
    roles = config.setdefault("modelRoles", {})
    if not isinstance(roles, dict):
        raise ValueError("OMP modelRoles must be a mapping")
    changed = False
    for role in BACKGROUND_ROLES:
        if _uses_qwasar(roles.get(role)) or role not in roles:
            if roles.get(role) != BACKGROUND_MODEL:
                roles[role] = BACKGROUND_MODEL
                changed = True
    if not changed:
        return None
    return _atomic_write(target, _dump(config), original, ".qwasar-omp-config-")


def main():
    parser = argparse.ArgumentParser(description="Register Qwarz in OMP without changing other providers")
    parser.add_argument("--target", type=Path, default=default_target())
    parser.add_argument("--config", type=Path, default=default_config())
    arguments = parser.parse_args()
    backup = configure(arguments.target)
    print(f"OMP Qwarz provider configured: {arguments.target}")
    if backup is not None:
        print(f"Original models backup: {backup}")
    roles_backup = configure_roles(arguments.config)
    print(f"OMP background roles configured: {arguments.config}")
    if roles_backup is not None:
        print(f"Original config backup: {roles_backup}")


if __name__ == "__main__":
    main()
