from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from qwasar_bench.schema import BenchmarkSample, RunMetadata

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class QualityScore:
    aggregate_coding_score: float
    tool_call_validity_percent: float


@dataclass(frozen=True, slots=True)
class MetricSummary:
    count: int
    mean: float
    p50: float
    p95: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    measured_value: float | None
    limit: float | None
    required_delta_percent: float | None
    detail: str


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    accepted: bool
    gates: dict[str, GateResult]
    baseline: dict[str, MetricSummary]
    candidate: dict[str, MetricSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _summary(values: Iterable[float]) -> MetricSummary:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot summarize an empty metric")
    return MetricSummary(
        count=len(materialized),
        mean=statistics.fmean(materialized),
        p50=_percentile(materialized, 0.50),
        p95=_percentile(materialized, 0.95),
        standard_deviation=statistics.pstdev(materialized),
    )


def _delta_percent(baseline: float, candidate: float) -> float:
    if baseline == 0:
        raise ValueError("cannot calculate a percent delta from a zero baseline")
    return (candidate / baseline - 1.0) * 100.0


def _completed(samples: Iterable[BenchmarkSample]) -> list[BenchmarkSample]:
    completed = [sample for sample in samples if sample.status == "completed"]
    if not completed:
        raise ValueError("comparison requires completed samples")
    return completed


def _metric_values(
    samples: Iterable[BenchmarkSample], attribute: str
) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = getattr(sample, attribute)
        if value is not None:
            values.append(float(value))
    return values


def _paired_case_floor(
    baseline: list[BenchmarkSample], candidate: list[BenchmarkSample]
) -> tuple[float, str]:
    baseline_by_case: dict[str, list[float]] = {}
    candidate_by_case: dict[str, list[float]] = {}
    for sample in baseline:
        value = sample.accepted_decode_tokens_per_second
        if value is not None:
            baseline_by_case.setdefault(sample.case_id, []).append(value)
    for sample in candidate:
        value = sample.accepted_decode_tokens_per_second
        if value is not None:
            candidate_by_case.setdefault(sample.case_id, []).append(value)
    if baseline_by_case.keys() != candidate_by_case.keys():
        raise ValueError("baseline and candidate decode cases are incompatible")
    deltas = {
        case_id: _delta_percent(
            _summary(baseline_by_case[case_id]).p50,
            _summary(candidate_by_case[case_id]).p50,
        )
        for case_id in baseline_by_case
    }
    if not deltas:
        raise ValueError("comparison requires decode throughput metrics")
    worst_case = min(deltas, key=deltas.__getitem__)
    return deltas[worst_case], worst_case


def compare_samples(
    baseline_samples: Iterable[BenchmarkSample],
    candidate_samples: Iterable[BenchmarkSample],
    *,
    baseline_quality: QualityScore | None = None,
    candidate_quality: QualityScore | None = None,
) -> ComparisonReport:
    baseline = _completed(baseline_samples)
    candidate = _completed(candidate_samples)
    baseline_cases = {sample.case_id for sample in baseline}
    candidate_cases = {sample.case_id for sample in candidate}
    if baseline_cases != candidate_cases:
        raise ValueError("baseline and candidate case sets are incompatible")

    baseline_decode = _metric_values(baseline, "accepted_decode_tokens_per_second")
    candidate_decode = _metric_values(candidate, "accepted_decode_tokens_per_second")
    baseline_large_prefill = [
        sample.prefill_tokens_per_second
        for sample in baseline
        if sample.prefill_tokens is not None
        and sample.prefill_tokens >= 8192
        and sample.prefill_tokens_per_second is not None
    ]
    candidate_large_prefill = [
        sample.prefill_tokens_per_second
        for sample in candidate
        if sample.prefill_tokens is not None
        and sample.prefill_tokens >= 8192
        and sample.prefill_tokens_per_second is not None
    ]
    baseline_summaries = {
        "decode_tps": _summary(baseline_decode),
        "prefill_tps_large_suffix": _summary(baseline_large_prefill),
    }
    candidate_summaries = {
        "decode_tps": _summary(candidate_decode),
        "prefill_tps_large_suffix": _summary(candidate_large_prefill),
    }

    decode_delta = _delta_percent(
        baseline_summaries["decode_tps"].p50,
        candidate_summaries["decode_tps"].p50,
    )
    prefill_delta = _delta_percent(
        baseline_summaries["prefill_tps_large_suffix"].p50,
        candidate_summaries["prefill_tps_large_suffix"].p50,
    )
    bucket_floor_delta, bucket_floor_case = _paired_case_floor(baseline, candidate)

    warm_128k_ttft = [
        sample.time_to_first_delta_ms
        for sample in candidate
        if sample.context_bucket == "128k"
        and sample.prefill_tokens is not None
        and sample.prefill_tokens <= 128
    ]
    peak_vram = [
        float(sample.peak_vram_bytes)
        for sample in candidate
        if sample.peak_vram_bytes is not None
    ]

    gates: dict[str, GateResult] = {
        "decode_tps": GateResult(
            passed=decode_delta >= 15.0,
            measured_value=decode_delta,
            limit=None,
            required_delta_percent=15.0,
            detail="candidate p50 accepted decode throughput versus baseline",
        ),
        "prefill_tps_large_suffix": GateResult(
            passed=prefill_delta >= 25.0,
            measured_value=prefill_delta,
            limit=None,
            required_delta_percent=25.0,
            detail="candidate p50 prefill throughput for suffixes >=8192 tokens",
        ),
        "context_bucket_floor": GateResult(
            passed=bucket_floor_delta >= -5.0,
            measured_value=bucket_floor_delta,
            limit=-5.0,
            required_delta_percent=None,
            detail=f"worst paired decode case: {bucket_floor_case}",
        ),
        "warm_128k_ttft": GateResult(
            passed=bool(warm_128k_ttft) and _summary(warm_128k_ttft).p50 <= 200.0,
            measured_value=_summary(warm_128k_ttft).p50 if warm_128k_ttft else None,
            limit=200.0,
            required_delta_percent=None,
            detail="p50 first output delta for <=128 appended tokens at 128K",
        ),
        "peak_vram": GateResult(
            passed=bool(peak_vram) and max(peak_vram) <= 28.5 * GIB,
            measured_value=max(peak_vram) if peak_vram else None,
            limit=28.5 * GIB,
            required_delta_percent=None,
            detail="maximum recorded device memory high-water mark",
        ),
    }

    if baseline_quality is None or candidate_quality is None:
        quality_gate = GateResult(
            passed=False,
            measured_value=None,
            limit=None,
            required_delta_percent=None,
            detail="baseline and candidate quality scores are required",
        )
    else:
        coding_delta = _delta_percent(
            baseline_quality.aggregate_coding_score,
            candidate_quality.aggregate_coding_score,
        )
        tool_delta = (
            candidate_quality.tool_call_validity_percent
            - baseline_quality.tool_call_validity_percent
        )
        quality_gate = GateResult(
            passed=coding_delta >= -1.0 and tool_delta >= -1.0,
            measured_value=coding_delta,
            limit=-1.0,
            required_delta_percent=None,
            detail=f"coding delta {coding_delta:.3f}%; tool validity delta {tool_delta:.3f} pp",
        )
    gates["quality"] = quality_gate
    return ComparisonReport(
        accepted=all(gate.passed for gate in gates.values()),
        gates=gates,
        baseline=baseline_summaries,
        candidate=candidate_summaries,
    )


def _load_samples(run_directory: Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    for line in (run_directory / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(BenchmarkSample.from_dict(json.loads(line)))
    return samples


def _load_quality(run_directory: Path) -> QualityScore | None:
    path = run_directory / "quality.json"
    if not path.is_file():
        return None
    return QualityScore(**json.loads(path.read_text(encoding="utf-8")))


def compare_run_directories(
    baseline_directory: str | Path, candidate_directory: str | Path
) -> ComparisonReport:
    baseline_path = Path(baseline_directory)
    candidate_path = Path(candidate_directory)
    baseline_metadata = RunMetadata.from_dict(
        json.loads((baseline_path / "run.json").read_text(encoding="utf-8"))
    )
    candidate_metadata = RunMetadata.from_dict(
        json.loads((candidate_path / "run.json").read_text(encoding="utf-8"))
    )
    if baseline_metadata.manifest_sha256 != candidate_metadata.manifest_sha256:
        raise ValueError("baseline and candidate manifest hashes differ")
    if baseline_metadata.model != candidate_metadata.model:
        raise ValueError("baseline and candidate model identities differ")
    return compare_samples(
        _load_samples(baseline_path),
        _load_samples(candidate_path),
        baseline_quality=_load_quality(baseline_path),
        candidate_quality=_load_quality(candidate_path),
    )
