from __future__ import annotations

import json
import subprocess
from pathlib import Path

from qwasar_bench.environment import capture_environment, sha256_path


def fake_commands(command: list[str]) -> subprocess.CompletedProcess[str]:
    if len(command) > 1 and command[0] == "nvidia-smi" and command[1].startswith("--query-gpu"):
        return subprocess.CompletedProcess(
            command,
            0,
            "0, GPU-5090, NVIDIA GeForce RTX 5090, 32607, 590.12, 41, 575.00, 2407, 12.0\n"
            "1, GPU-3090, NVIDIA GeForce RTX 3090 Ti, 24564, 590.12, 45, 450.00, 1800, 8.6\n",
            "",
        )
    if command == ["nvidia-smi"]:
        return subprocess.CompletedProcess(
            command,
            0,
            "NVIDIA-SMI 590.12 KMD Version: 590.12 CUDA UMD Version: 13.0\n",
            "",
        )
    if command == ["lscpu", "--json"]:
        payload = {
            "lscpu": [
                {"field": "Model name:", "data": "13th Gen Intel(R) Core(TM) i9-12900K"},
                {"field": "CPU(s):", "data": "24"},
            ]
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
    if command[:2] == ["git", "rev-parse"]:
        return subprocess.CompletedProcess(command, 0, "0123456789abcdef\n", "")
    raise AssertionError(f"unexpected command: {command}")


def fake_commands_without_gpu(command: list[str]) -> subprocess.CompletedProcess[str]:
    if command[0] == "nvidia-smi":
        raise FileNotFoundError("nvidia-smi")
    return fake_commands(command)


def test_environment_capture_parses_gpu_and_cpu() -> None:
    snapshot = capture_environment(run_command=fake_commands)

    assert snapshot.gpu.available is True
    assert snapshot.gpu.index == 0
    assert snapshot.gpu.uuid == "GPU-5090"
    assert snapshot.gpu.name == "NVIDIA GeForce RTX 5090"
    assert snapshot.gpu.total_memory_mib == 32607
    assert snapshot.gpu.compute_capability == "12.0"
    assert snapshot.visible_gpu_count == 2
    assert snapshot.gpu.cuda_version == "13.0"
    assert snapshot.cpu.model_name == "13th Gen Intel(R) Core(TM) i9-12900K"
    assert snapshot.cpu.logical_cpus == 24
    assert snapshot.engine_revision == "0123456789abcdef"


def test_environment_capture_marks_nvidia_unavailable() -> None:
    snapshot = capture_environment(run_command=fake_commands_without_gpu)

    assert snapshot.gpu.available is False
    assert snapshot.gpu.name is None
    assert snapshot.gpu.error is not None
    assert "nvidia-smi" in snapshot.gpu.error


def test_sha256_path_is_deterministic_for_directory_content(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.txt").write_text("beta", encoding="utf-8")
    (first / "a.txt").write_text("alpha", encoding="utf-8")
    (second / "a.txt").write_text("alpha", encoding="utf-8")
    (second / "b.txt").write_text("beta", encoding="utf-8")

    assert sha256_path(first) == sha256_path(second)


def test_sha256_path_ignores_downloader_state(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    downloading = tmp_path / "downloading"
    clean.mkdir()
    downloading.mkdir()
    (clean / "model.safetensors").write_bytes(b"weights")
    (downloading / "model.safetensors").write_bytes(b"weights")
    (downloading / "model.safetensors.part").write_bytes(b"partial")
    cache = downloading / ".cache/huggingface"
    cache.mkdir(parents=True)
    (cache / "metadata.json").write_text("{}", encoding="utf-8")

    assert sha256_path(clean) == sha256_path(downloading)
