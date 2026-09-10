from __future__ import annotations

from dataclasses import replace

from qwasar_bench.compare import QualityScore, compare_samples
from qwasar_bench.schema import BenchmarkSample

GIB = 1024**3


def sample(
    case_id: str,
    bucket: str,
    *,
    decode_tps: float,
    prefill_tokens: int,
    prefill_tps: float,
    ttft_ms: float = 250.0,
) -> BenchmarkSample:
    output_tokens = 120
    prompt_tokens = 200_000 if bucket == "256k" else 100_000
    return BenchmarkSample(
        case_id=case_id,
        backend="backend",
        model="qwen38",
        context_bucket=bucket,
        prompt_tokens=prompt_tokens,
        reused_prefix_tokens=prompt_tokens - prefill_tokens,
        output_tokens=output_tokens,
        time_to_first_delta_ms=ttft_ms,
        elapsed_ms=ttft_ms + output_tokens * 1000.0 / decode_tps,
        status="completed",
        prefill_elapsed_ms=prefill_tokens * 1000.0 / prefill_tps,
        decode_elapsed_ms=output_tokens * 1000.0 / decode_tps,
        peak_vram_bytes=28 * GIB,
    )


def baseline_samples() -> list[BenchmarkSample]:
    return [
        sample("fresh-1k", "1k", decode_tps=20, prefill_tokens=1024, prefill_tps=1000),
        sample("fresh-32k", "32k", decode_tps=20, prefill_tokens=32000, prefill_tps=1000),
        sample("fresh-128k", "128k", decode_tps=20, prefill_tokens=64000, prefill_tps=1000),
        sample("fresh-256k", "256k", decode_tps=20, prefill_tokens=64000, prefill_tps=1000),
        sample(
            "persistent-128k-turn",
            "128k",
            decode_tps=20,
            prefill_tokens=128,
            prefill_tps=1000,
            ttft_ms=240,
        ),
    ]


def candidate_samples() -> list[BenchmarkSample]:
    return [
        sample("fresh-1k", "1k", decode_tps=24, prefill_tokens=1024, prefill_tps=1300),
        sample("fresh-32k", "32k", decode_tps=24, prefill_tokens=32000, prefill_tps=1300),
        sample("fresh-128k", "128k", decode_tps=24, prefill_tokens=64000, prefill_tps=1300),
        sample("fresh-256k", "256k", decode_tps=24, prefill_tokens=64000, prefill_tps=1300),
        sample(
            "persistent-128k-turn",
            "128k",
            decode_tps=24,
            prefill_tokens=128,
            prefill_tps=1300,
            ttft_ms=180,
        ),
    ]


def test_acceptance_requires_decode_large_prefill_latency_memory_and_quality() -> None:
    report = compare_samples(
        baseline_samples(),
        candidate_samples(),
        baseline_quality=QualityScore(aggregate_coding_score=0.80, tool_call_validity_percent=99.0),
        candidate_quality=QualityScore(aggregate_coding_score=0.795, tool_call_validity_percent=98.5),
    )

    assert report.gates["decode_tps"].required_delta_percent == 15.0
    assert report.gates["prefill_tps_large_suffix"].required_delta_percent == 25.0
    assert report.gates["warm_128k_ttft"].passed is True
    assert report.gates["peak_vram"].passed is True
    assert report.gates["quality"].passed is True
    assert report.accepted is True


def test_any_bucket_more_than_five_percent_slower_rejects_candidate() -> None:
    candidate = candidate_samples()
    candidate[2] = replace(candidate[2], decode_elapsed_ms=120 * 1000.0 / 18.0)

    report = compare_samples(
        baseline_samples(),
        candidate,
        baseline_quality=QualityScore(0.80, 99.0),
        candidate_quality=QualityScore(0.80, 99.0),
    )

    assert report.gates["context_bucket_floor"].passed is False
    assert report.accepted is False


def test_missing_quality_data_cannot_pass_acceptance() -> None:
    report = compare_samples(baseline_samples(), candidate_samples())

    assert report.gates["quality"].passed is False
    assert report.accepted is False
