from __future__ import annotations

import json
from pathlib import Path

from qwasar_bench.environment import CpuEnvironment, EnvironmentSnapshot, GpuEnvironment
from qwasar_bench.openai_client import StreamResult
from qwasar_bench.runner import run_manifest
from qwasar_bench.workloads import load_manifest


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_stream(self, **kwargs: object) -> StreamResult:
        self.calls.append(kwargs)
        index = len(self.calls)
        started = index * 1_000_000_000
        return StreamResult(
            response_id=f"resp_{index}",
            text=f"output {index}",
            input_tokens=384 if index % 2 else 512,
            output_tokens=16,
            request_started_ns=started,
            headers_received_ns=started + 10_000_000,
            first_delta_ns=started + 20_000_000,
            completed_ns=started + 1_020_000_000,
            events=(
                {
                    "type": "qwasar.metrics",
                    "prefill_elapsed_ms": 10.0,
                    "reused_prefix_tokens": 0 if index % 2 else 384,
                    "peak_vram_bytes": 27_000_000_000,
                },
            ),
        )


class UninstrumentedClient(RecordingClient):
    def create_stream(self, **kwargs: object) -> StreamResult:
        result = super().create_stream(**kwargs)
        return StreamResult(
            response_id=result.response_id,
            text=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            request_started_ns=result.request_started_ns,
            headers_received_ns=result.headers_received_ns,
            first_delta_ns=result.first_delta_ns,
            completed_ns=result.completed_ns,
            events=(),
        )


def write_persistent_manifest(tmp_path: Path) -> Path:
    values = {
        "version": 1,
        "model": {
            "architecture": "Qwen3.8ForCausalLM",
            "repository": "Qwen/Qwen3.8-27B",
            "quantization": "EXL3",
            "target_bpw": 5.0,
            "native_context_tokens": 262144,
            "artifact_path_env": "QWASAR_MODEL_PATH",
            "artifact_sha256": None,
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 16,
            "seed": 0,
        },
        "cases": [
            {
                "case_id": "seed",
                "fixture_id": "agent-seed",
                "mode": "persistent_turn",
                "context_bucket": "1k",
                "target_context_tokens": 384,
                "appended_tokens": 384,
                "warmups": 0,
                "repetitions": 1,
                "session_id": "agent",
                "turn_index": 0,
            },
            {
                "case_id": "turn",
                "fixture_id": "agent-turn",
                "mode": "persistent_turn",
                "context_bucket": "1k",
                "target_context_tokens": 512,
                "appended_tokens": 128,
                "warmups": 0,
                "repetitions": 1,
                "session_id": "agent",
                "turn_index": 1,
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        captured_at="2026-09-04T12:00:00+00:00",
        operating_system="Linux",
        machine="x86_64",
        python_version="3.12.0",
        gpu=GpuEnvironment(
            available=True,
            error=None,
            name="NVIDIA GeForce RTX 5090",
            total_memory_mib=32607,
        ),
        cpu=CpuEnvironment(model_name="Intel Core i9-12900K", logical_cpus=24),
        engine_revision="0123456789abcdef",
        manifest_sha256=None,
        artifact_sha256=None,
    )


def test_runner_reuses_previous_response_for_persistent_turn(tmp_path: Path) -> None:
    client = RecordingClient()
    manifest = load_manifest(write_persistent_manifest(tmp_path))

    run_manifest(
        manifest,
        client,
        tmp_path / "run",
        backend="exllamav3",
        model="qwen38",
        environment=environment(),
        build_input=lambda case: [{"role": "user", "content": case.case_id}],
    )

    assert "previous_response_id" not in client.calls[0]
    assert client.calls[0]["model"] == "qwen38"
    assert client.calls[1]["previous_response_id"] == "resp_1"


def test_runner_does_not_infer_physical_prefix_reuse(tmp_path: Path) -> None:
    result = run_manifest(
        load_manifest(write_persistent_manifest(tmp_path)),
        UninstrumentedClient(),
        tmp_path / "run-uninstrumented",
        backend="exllamav3",
        model="qwen38",
        environment=environment(),
        build_input=lambda case: [{"role": "user", "content": case.case_id}],
    )

    assert all(sample.reused_prefix_tokens is None for sample in result.samples)


def test_runner_supports_more_cached_turn_samples_than_seed_samples(
    tmp_path: Path,
) -> None:
    manifest_path = write_persistent_manifest(tmp_path)
    values = json.loads(manifest_path.read_text(encoding="utf-8"))
    values["cases"][1]["repetitions"] = 3
    manifest_path.write_text(json.dumps(values), encoding="utf-8")
    client = RecordingClient()

    result = run_manifest(
        load_manifest(manifest_path),
        client,
        tmp_path / "run-asymmetric",
        backend="exllamav3",
        model="qwen38",
        environment=environment(),
        build_input=lambda case: [{"role": "user", "content": case.case_id}],
    )

    assert len(client.calls) == 6
    assert [sample.case_id for sample in result.samples] == [
        "seed",
        "turn",
        "turn",
        "turn",
    ]
    assert client.calls[1]["previous_response_id"] == "resp_1"
    assert client.calls[3]["previous_response_id"] == "resp_3"
    assert client.calls[5]["previous_response_id"] == "resp_5"


def test_run_directory_is_published_atomically(tmp_path: Path) -> None:
    output = tmp_path / "baseline-001"
    manifest = load_manifest(write_persistent_manifest(tmp_path))

    result = run_manifest(
        manifest,
        RecordingClient(),
        output,
        backend="exllamav3",
        model="qwen38",
        environment=environment(),
        build_input=lambda case: [{"role": "user", "content": case.case_id}],
    )

    assert result.output_directory == output
    assert (output / "run.json").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "samples.jsonl").is_file()
    assert len(list((output / "events").glob("*.jsonl"))) == 2
    assert not list(tmp_path.glob(".baseline-001.tmp-*"))
