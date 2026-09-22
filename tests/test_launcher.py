import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def launcher():
    source = ROOT / "scripts/qwasar.py"
    if not source.exists():
        pytest.fail("Qwasar launcher is missing")
    spec = importlib.util.spec_from_file_location("launcher", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_guard_refuses_wrong_device_busy_device_and_unknown_usage():
    module = launcher()
    for values in (["NVIDIA GeForce RTX 3090 Ti"], ["NVIDIA GeForce RTX 5090", "234, 10000"], ["NVIDIA GeForce RTX 5090", "234, [N/A]"]):
        with patch.object(module.subprocess, "check_output", side_effect=values):
            with pytest.raises(RuntimeError):
                module.check_gpu()


def test_gpu_guard_allows_desktop_without_compute_and_queries_only_zero():
    module = launcher()
    with patch.object(module.subprocess, "check_output", side_effect=["NVIDIA GeForce RTX 5090", ""]) as query:
        module.check_gpu()
    assert all(call.args[0][1:3] == ["-i", "0"] for call in query.call_args_list)


def test_process_identity_rejects_unrelated_pid_and_pid_reuse():
    module = launcher()
    assert not module.same_process({"pid": 1, "start": "not-real"})
    assert not module.same_process({"pid": 999999999, "start": "1"})


def test_process_identity_survives_binary_replacement():
    module = launcher()
    with patch.object(module, "process_start", return_value="123"), patch.object(module.os, "readlink", return_value=str(module.BINARY) + " (deleted)"), patch.object(module.Path, "read_bytes", return_value=bytes(str(module.BINARY), "utf8") + b"\0"):
        assert module.same_process({"pid": 1234, "start": "123"})


def test_process_identity_accepts_qwasar_symlink_script_path():
    module = launcher()
    script = Path("/home/rekeyea/Documents/llm/qwasar/scripts/qwasar.py")
    command = [b"/usr/bin/python3", os.fsencode(str(script)), b"serve", b"--prefill", b"flash"]
    resolved = module.resolved_command(command)
    assert os.fsencode(str((ROOT / "scripts/qwasar.py").resolve())) in resolved
    assert b"serve" in command


def test_existing_process_is_not_reported_ready_when_worker_is_loading():
    module = launcher()
    with patch.object(module, "record", return_value={"pid": 1234, "start": "123"}), patch.object(module, "same_process", return_value=True), patch.object(module, "health", return_value={"worker": {"status": "loading"}}), patch.object(module.time, "sleep"):
        with pytest.raises(RuntimeError, match="pending"):
            module.start("flash")
