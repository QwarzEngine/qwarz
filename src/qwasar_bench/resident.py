from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Literal, Mapping, Sequence


@dataclass(frozen=True)
class ResidentProbeConfig:
    context_tokens: tuple[int, ...]
    appended_tokens: int
    max_new_tokens: int
    repetitions: int
    page_size: int = 256

    def __post_init__(self) -> None:
        if not self.context_tokens:
            raise ValueError("context_tokens must not be empty")
        if any(value <= 0 for value in self.context_tokens):
            raise ValueError("context_tokens must be positive")
        if any(value > 262144 for value in self.context_tokens):
            raise ValueError("context_tokens must not exceed native limit 262144")
        if tuple(sorted(set(self.context_tokens))) != self.context_tokens:
            raise ValueError("context_tokens must be unique and increasing")
        if self.appended_tokens <= 0:
            raise ValueError("appended_tokens must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        minimum = self.appended_tokens + 2 * self.max_new_tokens
        if any(bucket <= minimum for bucket in self.context_tokens):
            raise ValueError(
                "every context bucket must be larger than appended_tokens plus "
                "twice max_new_tokens"
            )


@dataclass(frozen=True)
class JobObservation:
    bucket_tokens: int
    phase: Literal["seed", "turn"]
    repetition: int
    input_tokens: int
    cached_tokens: int
    prefill_tokens: int
    ttft_ms: float
    prefill_ms: float
    generated_tokens: int
    page_metrics: dict[str, int]

    def __post_init__(self) -> None:
        prefill_input_tokens = max(self.input_tokens - 1, 0)
        if self.bucket_tokens <= 0:
            raise ValueError("bucket_tokens must be positive")
        if self.phase not in ("seed", "turn"):
            raise ValueError("phase must be seed or turn")
        if self.repetition < 0:
            raise ValueError("repetition must be non-negative")
        if self.input_tokens <= 0:
            raise ValueError("input_tokens must be positive")
        if not 0 <= self.cached_tokens <= prefill_input_tokens:
            raise ValueError("cached_tokens must fit inside prefill input")
        expected_prefill = prefill_input_tokens - self.cached_tokens
        if self.prefill_tokens != expected_prefill:
            raise ValueError("prefill_tokens does not match physical cache accounting")
        if self.ttft_ms < 0 or self.prefill_ms < 0:
            raise ValueError("latencies must be non-negative")
        if self.generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative")
        if any(value < 0 for value in self.page_metrics.values()):
            raise ValueError("page_metrics values must be non-negative")


def summarize_observations(
    observations: Sequence[JobObservation],
) -> dict[str, object]:
    turns_by_bucket: dict[int, list[JobObservation]] = {}
    for observation in observations:
        if observation.phase == "turn":
            turns_by_bucket.setdefault(observation.bucket_tokens, []).append(observation)

    buckets: dict[str, object] = {}
    for bucket_tokens in sorted(turns_by_bucket):
        turns = turns_by_bucket[bucket_tokens]
        cache_ratios = [
            turn.cached_tokens / max(turn.input_tokens - 1, 1) for turn in turns
        ]
        buckets[str(bucket_tokens)] = {
            "turn_count": len(turns),
            "turn_ttft_ms_p50": float(median(turn.ttft_ms for turn in turns)),
            "turn_prefill_ms_p50": float(median(turn.prefill_ms for turn in turns)),
            "min_cached_ratio": min(cache_ratios),
            "max_prefill_tokens": max(turn.prefill_tokens for turn in turns),
        }

    return {
        "observation_count": len(observations),
        "turn_count": sum(len(turns) for turns in turns_by_bucket.values()),
        "buckets": buckets,
    }


def evaluate_resident_slo(summary: Mapping[str, object]) -> dict[str, object]:
    raw_buckets = summary.get("buckets")
    buckets = raw_buckets if isinstance(raw_buckets, Mapping) else {}
    bucket_128k = buckets.get("131072")
    bucket_256k = buckets.get("262144")
    ttft_128k = _bucket_ttft(bucket_128k)
    ttft_256k = _bucket_ttft(bucket_256k)

    ratio = None
    if ttft_128k is not None and ttft_256k is not None and ttft_128k > 0:
        ratio = round(ttft_256k / ttft_128k, 3)

    ttft_128k_pass = ttft_128k is not None and ttft_128k <= 200.0
    ttft_256k_pass = ttft_256k is not None and ttft_256k <= 300.0
    ratio_pass = ratio is not None and ratio <= 1.5
    return {
        "ttft_128k_pass": ttft_128k_pass,
        "ttft_256k_pass": ttft_256k_pass,
        "ratio_256k_to_128k": ratio,
        "ratio_pass": ratio_pass,
        "all_pass": ttft_128k_pass and ttft_256k_pass and ratio_pass,
    }


def _bucket_ttft(bucket: object) -> float | None:
    if not isinstance(bucket, Mapping):
        return None
    value = bucket.get("turn_ttft_ms_p50")
    if not isinstance(value, (int, float)):
        return None
    return float(value)
