from __future__ import annotations

import pytest

from qwasar_bench.resident import (
    JobObservation,
    ResidentProbeConfig,
    evaluate_resident_slo,
    summarize_observations,
)


def make_turn(bucket_tokens: int, ttft_ms: float) -> JobObservation:
    return JobObservation(
        bucket_tokens=bucket_tokens,
        phase="turn",
        repetition=0,
        input_tokens=bucket_tokens,
        cached_tokens=bucket_tokens - 256,
        prefill_tokens=255,
        ttft_ms=ttft_ms,
        prefill_ms=ttft_ms - 5.0,
        generated_tokens=1,
        page_metrics={"alloc_cached_pages": (bucket_tokens - 256) // 256},
    )


def test_config_rejects_bucket_without_turn_headroom() -> None:
    with pytest.raises(ValueError, match="larger"):
        ResidentProbeConfig(
            context_tokens=(128,),
            appended_tokens=128,
            max_new_tokens=1,
            repetitions=3,
        )


def test_config_rejects_native_context_overflow():
    with pytest.raises(ValueError, match="262144"):
        ResidentProbeConfig(
            context_tokens=(262400,), appended_tokens=128,
            max_new_tokens=1, repetitions=1,
        )


def test_config_reserves_both_seed_and_branch_generation():
    with pytest.raises(ValueError, match="larger"):
        ResidentProbeConfig(
            context_tokens=(256,), appended_tokens=128,
            max_new_tokens=64, repetitions=1,
        )


def test_observation_rejects_impossible_cache_accounting() -> None:
    with pytest.raises(ValueError, match="cached_tokens"):
        JobObservation(
            bucket_tokens=1024,
            phase="turn",
            repetition=0,
            input_tokens=100,
            cached_tokens=101,
            prefill_tokens=0,
            ttft_ms=10.0,
            prefill_ms=8.0,
            generated_tokens=1,
            page_metrics={},
        )


def test_observation_requires_exact_prefill_accounting() -> None:
    with pytest.raises(ValueError, match="prefill_tokens"):
        JobObservation(
            bucket_tokens=1024,
            phase="turn",
            repetition=0,
            input_tokens=1024,
            cached_tokens=768,
            prefill_tokens=128,
            ttft_ms=10.0,
            prefill_ms=8.0,
            generated_tokens=1,
            page_metrics={},
        )


def test_summary_and_slo_use_turn_medians() -> None:
    observations = [
        make_turn(131072, 190.0),
        make_turn(131072, 210.0),
        make_turn(131072, 200.0),
        make_turn(262144, 280.0),
        make_turn(262144, 300.0),
        make_turn(262144, 290.0),
    ]

    summary = summarize_observations(observations)
    slo = evaluate_resident_slo(summary)

    assert summary["buckets"]["131072"]["turn_ttft_ms_p50"] == 200.0
    assert summary["buckets"]["262144"]["turn_ttft_ms_p50"] == 290.0
    assert summary["buckets"]["262144"]["max_prefill_tokens"] == 255
    assert slo == {
        "ttft_128k_pass": True,
        "ttft_256k_pass": True,
        "ratio_256k_to_128k": 1.45,
        "ratio_pass": True,
        "all_pass": True,
    }


def test_slo_fails_cleanly_when_required_buckets_are_missing() -> None:
    summary = summarize_observations([make_turn(1024, 10.0)])

    assert evaluate_resident_slo(summary) == {
        "ttft_128k_pass": False,
        "ttft_256k_pass": False,
        "ratio_256k_to_128k": None,
        "ratio_pass": False,
        "all_pass": False,
    }
