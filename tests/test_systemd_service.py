import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "integrations/systemd/qwasar.service"


def launcher():
    spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/qwasar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unit_uses_foreground_pinned_profile_and_waits_until_ready():
    assert UNIT.exists(), "systemd unit missing"
    unit = UNIT.read_text()
    for expected in ("Type=exec", "qwasar.py serve --prefill flash", "qwasar.py wait-ready --pid ${MAINPID}",
                     "Qwen3.8-27B-EXL3-5.0bpw", "CUDA_VISIBLE_DEVICES=0", "Restart=on-failure",
                     "KillMode=mixed", "WantedBy=default.target"):
        assert expected in unit
    assert "cargo" not in unit


def test_readiness_rejects_another_supervisors_worker():
    module = launcher()
    response = {"worker": {"status": "ready", "pid": 1234}}
    with patch.object(module, "same_process", return_value=True), patch.object(module, "health", return_value=response), patch.object(module.Path, "read_text", return_value="1234 (python) S 9999 0"), patch.object(module.time, "sleep"):
        with pytest.raises(RuntimeError, match="pending"):
            module.wait_ready({"pid": 5678, "start": "123"})


def test_readiness_accepts_only_the_owned_worker():
    module = launcher()
    response = {"worker": {"status": "ready", "pid": 1234}}
    with patch.object(module, "same_process", return_value=True), patch.object(module, "health", return_value=response), patch.object(module.Path, "read_text", return_value="1234 (python) S 5678 0"):
        module.wait_ready({"pid": 5678, "start": "123"})


def test_managed_commands_use_systemd_after_installation(tmp_path):
    module = launcher()
    module.SYSTEMD_UNIT = tmp_path / "qwasar.service"
    assert hasattr(module, "systemd_command"), "systemd routing missing"
    assert module.systemd_command("start", "flash") is None
    module.SYSTEMD_UNIT.symlink_to(UNIT)
    assert module.systemd_command("start", "flash") == ["systemctl", "--user", "start", "qwasar.service"]
    assert module.systemd_command("stop", "flash") == ["systemctl", "--user", "stop", "qwasar.service"]
    assert module.systemd_command("logs", "flash")[:3] == ["journalctl", "--user", "-u"]
    assert module.systemd_command("serve", "flash") is None
    assert module.systemd_command("wait-ready", "flash") is None
    with pytest.raises(ValueError, match="flash"):
        module.systemd_command("start", "baseline")
