from __future__ import annotations

import pytest

from qwasar_bench.schema import BenchmarkSample, RunMetadata


def make_sample(**overrides: object) -> BenchmarkSample:
    values: dict[str, object] = {
        "case_id": "interactive-1k",
        "backend": "exllamav3",
        "model": "Qwen3.8-27B-EXL3-5.0bpw",
        "context_bucket": "1k",
        "prompt_tokens": 1024,
        "reused_prefix_tokens": 1000,
        "output_tokens": 64,
        "time_to_first_delta_ms": 120.0,
        "elapsed_ms": 2120.0,
        "prefill_elapsed_ms": 12.0,
        "decode_elapsed_ms": 2000.0,
        "peak_vram_bytes": 28_000_000_000,
        "status": "completed",
    }
    values.update(overrides)
    return BenchmarkSample(**values)


def test_benchmark_sample_round_trips_and_derives_metrics() -> None:
    sample = make_sample()

    assert BenchmarkSample.from_dict(sample.to_dict()) == sample
    assert sample.prefill_tokens == 24
    assert sample.prefill_tokens_per_second == 2000.0
    assert sample.accepted_decode_tokens_per_second == 32.0


def test_invalid_token_accounting_is_rejected() -> None:
    with pytest.raises(ValueError, match="reused_prefix_tokens"):
        make_sample(prompt_tokens=10, reused_prefix_tokens=11)


def test_failed_sample_requires_an_error_message() -> None:
    with pytest.raises(ValueError, match="error"):
        make_sample(status="failed", error=None)


def test_optional_backend_counters_may_be_absent_from_wire_data() -> None:
    values = make_sample().to_dict()
    values["reused_prefix_tokens"] = None
    values.pop("prefill_elapsed_ms")
    values.pop("decode_elapsed_ms")
    values.pop("peak_vram_bytes")

    sample = BenchmarkSample.from_dict(values)

    assert sample.prefill_elapsed_ms is None
    assert sample.decode_elapsed_ms is None
    assert sample.peak_vram_bytes is None
    assert sample.reused_prefix_tokens is None
    assert sample.prefill_tokens is None
    assert sample.prefill_tokens_per_second is None


def test_run_metadata_round_trips() -> None:
    metadata = RunMetadata(
        run_id="20260904T120000Z-exllamav3",
        manifest_sha256="a" * 64,
        backend="exllamav3",
        model="Qwen3.8-27B-EXL3-5.0bpw",
        started_at="2026-09-04T12:00:00Z",
        completed_at="2026-09-04T12:30:00Z",
        engine_revision="0123456789abcdef",
        artifact_sha256="b" * 64,
        sample_count=12,
    )

    assert RunMetadata.from_dict(metadata.to_dict()) == metadata
