from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwasar_bench import cli
from qwasar_bench import exllamav3_probe
from qwasar_bench.compare import ComparisonReport, GateResult, MetricSummary
from qwasar_bench.environment import CpuEnvironment, EnvironmentSnapshot, GpuEnvironment
from qwasar_bench.openai_client import ChatCompletionsClient, ResponsesClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "benchmarks/manifests/qwen38-27b-rtx5090-v1.json"


def unavailable_environment(**_: object) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        captured_at="2026-09-04T12:00:00+00:00",
        operating_system="Linux",
        machine="x86_64",
        python_version="3.12.0",
        gpu=GpuEnvironment(available=False, error="nvidia-smi unavailable"),
        cpu=CpuEnvironment(model_name="Intel Core i9-12900K", logical_cpus=24),
        engine_revision=None,
    )


def accepted_report() -> ComparisonReport:
    summary = MetricSummary(count=3, mean=1.0, p50=1.0, p95=1.0, standard_deviation=0.0)
    return ComparisonReport(
        accepted=True,
        gates={
            "decode_tps": GateResult(
                passed=True,
                measured_value=20.0,
                limit=None,
                required_delta_percent=15.0,
                detail="test",
            )
        },
        baseline={"decode_tps": summary},
        candidate={"decode_tps": summary},
    )


def test_doctor_returns_two_without_required_gpu(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "capture_environment", unavailable_environment)

    exit_code = cli.main(["doctor", "--manifest", str(MANIFEST_PATH)])

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out)["gpu_ready"] is False


def test_validate_manifest_prints_stable_hash(capsys) -> None:
    exit_code = cli.main(["validate-manifest", str(MANIFEST_PATH)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["version"] == 1
    assert len(payload["sha256"]) == 64
    assert payload["case_count"] == 12


def test_compare_writes_machine_readable_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "compare_run_directories", lambda *_: accepted_report())
    output = tmp_path / "comparison.json"

    exit_code = cli.main(
        ["compare", "baseline", "candidate", "--output", str(output)]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["accepted"] is True


def test_baseline_protocol_selects_explicit_client() -> None:
    responses = cli.build_streaming_client("responses", "http://127.0.0.1:8000", 30.0)
    chat = cli.build_streaming_client(
        "chat-completions", "http://127.0.0.1:8000", 30.0
    )

    assert isinstance(responses, ResponsesClient)
    assert isinstance(chat, ChatCompletionsClient)


def test_doctor_passes_explicit_gpu_index(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> EnvironmentSnapshot:
        captured.update(kwargs)
        return unavailable_environment()

    monkeypatch.setenv("QWASAR_CUDA_DEVICE", "1")
    monkeypatch.setattr(cli, "capture_environment", capture)

    cli._doctor_report(cli.load_manifest(MANIFEST_PATH))

    assert captured["selected_gpu_index"] == 1


def test_bringup_allows_uncommitted_engine_but_not_acceptance(
    tmp_path: Path, monkeypatch
) -> None:
    profile = cli.load_manifest(
        REPOSITORY_ROOT
        / "benchmarks/manifests/qwen38-27b-rtx5090-exl3-3.5-bringup-v1.json"
    )
    artifact = tmp_path / "model"
    artifact.mkdir()

    def capture(**_: object) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            captured_at="2026-09-04T12:00:00+00:00",
            operating_system="Linux",
            machine="x86_64",
            python_version="3.12.0",
            gpu=GpuEnvironment(
                available=True,
                error=None,
                index=0,
                name="NVIDIA GeForce RTX 5090",
                total_memory_mib=32607,
                compute_capability="12.0",
            ),
            cpu=CpuEnvironment(model_name="Intel Core i9-12900K", logical_cpus=24),
            engine_revision=None,
            artifact_sha256=profile.model.artifact_sha256,
        )

    monkeypatch.setenv("QWASAR_MODEL_PATH", str(artifact))
    monkeypatch.setattr(cli, "capture_environment", capture)

    bringup, _ = cli._doctor_report(profile, qualification="bringup")
    acceptance, _ = cli._doctor_report(profile, qualification="acceptance")

    assert bringup["ready"] is True
    assert bringup["acceptance_ready"] is False
    assert acceptance["ready"] is False
    assert acceptance["quantization_ready"] is False


def test_parser_accepts_resident_probe_defaults() -> None:
    arguments = cli._parser().parse_args(
        [
            "resident-probe",
            "--model",
            "/model",
            "--output",
            "/results",
        ]
    )

    assert arguments.contexts == (1024, 32768, 131072, 262144)
    assert arguments.appended_tokens == 128
    assert arguments.max_new_tokens == 1
    assert arguments.repetitions == 3
    assert arguments.cache_size == 262400
    assert arguments.cache_quant == "nvfp4"
    assert arguments.gpu_split_gb == 30.0


def test_parser_rejects_unsorted_resident_contexts() -> None:
    parser = cli._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "resident-probe",
                "--model",
                "/model",
                "--output",
                "/results",
                "--contexts",
                "32768,1024",
            ]
        )


def test_parser_uses_donor_cache_quant_spelling() -> None:
    arguments = cli._parser().parse_args(
        [
            "resident-probe",
            "--model",
            "/model",
            "--output",
            "/results",
            "--cache-quant",
            "4",
        ]
    )

    assert arguments.cache_quant == "4"


def test_resident_probe_command_delegates_without_eager_cuda(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def run_probe(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"completed": True}

    monkeypatch.setattr(exllamav3_probe, "run_probe", run_probe)

    exit_code = cli.main(
        [
            "resident-probe",
            "--model",
            "/model",
            "--draft-model",
            "/draft",
            "--output",
            str(tmp_path),
            "--contexts",
            "1024",
            "--repetitions",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["context_tokens"] == (1024,)
    assert captured["cache_quant"] == "nvfp4"
    assert json.loads(capsys.readouterr().out) == {"completed": True}
