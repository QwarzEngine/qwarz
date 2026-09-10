from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

import qwasar_bench.exllamav3_probe as probe
from qwasar_bench.exllamav3_probe import (
    _model_init_argv,
    build_branch_token_ids,
    run_generator_job,
    write_probe_artifacts,
)
from qwasar_bench.resident import JobObservation, ResidentProbeConfig


@pytest.mark.parametrize("count", [None, 1, 4, 7])
def test_mtp_width_reaches_cache_allocation_and_generator(monkeypatch, tmp_path, count):
    import sys
    from types import SimpleNamespace

    def add_args(parser, **kwargs):
        for flag in ("model_dir", "gpu_split", "cache_size", "cache_quant"):
            parser.add_argument("--" + flag)
        parser.add_argument("--mtp", action="store_true")
        parser.add_argument("--num_draft_tokens", type=int, default=None)

    seen = []
    allocations = []
    def init_model(args, **kwargs):
        allocations.append(args.num_draft_tokens)
        return object(), None, object(), object(), object(), None, object()

    def make_generator(*args, **kwargs):
        seen.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "exllamav3", SimpleNamespace(
        Generator=make_generator, Job=object, model_init=SimpleNamespace(
            add_args=add_args, init=init_model)))
    monkeypatch.setitem(sys.modules, "exllamav3.generator.sampler.presets", SimpleNamespace(GreedySampler=object))
    probe._load_exllamav3_runtime(model_path=tmp_path, draft_model_path=tmp_path,
        cache_size=262144, cache_quant="8,4", gpu_split_gb=30, draft_method="mtp",
        num_draft_tokens=count)
    assert len(seen) == 1
    assert allocations == [count]
    if count is None:
        assert "num_draft_tokens" not in seen[0]
    else:
        assert seen[0]["num_draft_tokens"] == count


@pytest.mark.parametrize("count,method", [(0,"mtp"),(8,"mtp"),(True,"mtp"),(2,"dflash2")])
def test_invalid_fixed_width_fails_before_loading_gpu(tmp_path, count, method):
    with pytest.raises(ValueError):
        probe._load_exllamav3_runtime(model_path=tmp_path, draft_model_path=tmp_path,
            cache_size=262144, cache_quant="8,4", gpu_split_gb=30,
            draft_method=method, num_draft_tokens=count)


class FakeTokens:
    def __init__(self, count: int) -> None:
        self.shape = (1, count)


@dataclass
class FakeJob:
    input_ids: FakeTokens
    max_new_tokens: int
    seed: int
    cached_pages: int
    cached_tokens: int
    time_prefill: float


class FakeJobFactory:
    def __init__(
        self, *, cached_pages: int, cached_tokens: int, time_prefill: float
    ) -> None:
        self.cached_pages = cached_pages
        self.cached_tokens = cached_tokens
        self.time_prefill = time_prefill

    def __call__(
        self, *, input_ids: FakeTokens, max_new_tokens: int, seed: int
    ) -> FakeJob:
        return FakeJob(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            seed=seed,
            cached_pages=self.cached_pages,
            cached_tokens=self.cached_tokens,
            time_prefill=self.time_prefill,
        )


class FakePageTable:
    def __init__(self) -> None:
        self.metrics = {"alloc_cached_pages": 10, "evictions": 2}


class FakeGenerator:
    def __init__(self, rounds: list[list[dict[str, object]]]) -> None:
        self.rounds = list(rounds)
        self.pagetable = FakePageTable()
        self.remaining = 0

    def enqueue(self, _: FakeJob) -> int:
        self.remaining = 1
        return 0

    def num_remaining_jobs(self) -> int:
        return self.remaining

    def iterate(self) -> list[dict[str, object]]:
        results = self.rounds.pop(0)
        if any(result.get("eos") for result in results):
            self.remaining = 0
            self.pagetable.metrics["alloc_cached_pages"] += 3
        return results


def test_run_generator_job_uses_first_streaming_event_for_ttft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([1_000_000_000, 1_011_000_000])
    monkeypatch.setattr(probe.time, "perf_counter_ns", lambda: next(clock))
    job_factory = FakeJobFactory(
        cached_pages=3, cached_tokens=0, time_prefill=0.008
    )
    generator = FakeGenerator(
        [
            [{"stage": "started"}, {"stage": "prefill", "curr_progress": 768}],
            [
                {
                    "stage": "streaming",
                    "token_ids": FakeTokens(1),
                    "new_tokens": 1,
                    "eos": True,
                }
            ],
        ]
    )

    observation, job = run_generator_job(
        generator,
        job_factory,
        FakeTokens(1025),
        max_new_tokens=1,
        seed=7,
        bucket_tokens=1024,
        phase="turn",
        repetition=0,
    )

    assert isinstance(job, FakeJob)
    assert observation.ttft_ms == 11.0
    assert observation.prefill_ms == 8.0
    assert observation.cached_tokens == 768
    assert observation.prefill_tokens == 256
    assert observation.generated_tokens == 1
    assert observation.page_metrics == {"alloc_cached_pages": 3, "evictions": 0}


def test_run_generator_job_rejects_queue_without_streaming_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe.time, "perf_counter_ns", lambda: 1_000_000_000)
    generator = FakeGenerator([[{"stage": "started", "eos": True}]])

    with pytest.raises(RuntimeError, match="streaming token"):
        run_generator_job(
            generator,
            FakeJobFactory(cached_pages=0, cached_tokens=0, time_prefill=0.0),
            FakeTokens(10),
            max_new_tokens=1,
            seed=7,
            bucket_tokens=10,
            phase="seed",
            repetition=0,
        )


def test_build_branch_token_ids_preserves_seed_and_exact_bucket_length() -> None:
    branch = build_branch_token_ids(
        [11, 12, 13], [21, 22, 23], bucket_tokens=5
    )

    assert branch == [11, 12, 13, 21, 22]


def test_build_branch_token_ids_rejects_short_suffix() -> None:
    with pytest.raises(ValueError, match="suffix"):
        build_branch_token_ids([11, 12, 13], [21], bucket_tokens=5)


def test_write_artifacts_is_atomic(tmp_path) -> None:
    config = ResidentProbeConfig(
        context_tokens=(1024,),
        appended_tokens=128,
        max_new_tokens=1,
        repetitions=1,
    )
    observations = [
        JobObservation(
            bucket_tokens=1024,
            phase="turn",
            repetition=0,
            input_tokens=1024,
            cached_tokens=768,
            prefill_tokens=255,
            ttft_ms=11.0,
            prefill_ms=8.0,
            generated_tokens=1,
            page_metrics={"alloc_cached_pages": 3},
        )
    ]
    summary = {"buckets": {"1024": {"turn_ttft_ms_p50": 11.0}}}
    environment = {"gpu": {"name": "NVIDIA GeForce RTX 5090"}}

    write_probe_artifacts(
        tmp_path,
        config=config,
        observations=observations,
        summary=summary,
        environment=environment,
        model_path="/model",
        draft_model_path="/draft",
        cache_size=262400,
        cache_quant="nvfp4",
        gpu_split_gb=30.0,
    )

    summary_payload = json.loads((tmp_path / "summary.json").read_text())
    assert summary_payload["schema_version"] == 1
    assert summary_payload["qualification"] == "bringup_only"
    assert summary_payload["cache_quant"] == "nvfp4"
    assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_model_init_argv_uses_unambiguous_donor_flags() -> None:
    assert _model_init_argv(
        model_path="/model",
        draft_model_path="/draft",
        cache_size=262400,
        cache_quant="nvfp4",
        gpu_split_gb=30.0,
    ) == [
        "--model_dir",
        "/model",
        "--gpu_split",
        "30",
        "--draft_model_dir",
        "/draft",
        "--cache_size",
        "262400",
        "--cache_quant",
        "nvfp4",
    ]


def test_resident_matrix_reserves_output_inside_context(monkeypatch):
    prompts = []
    monkeypatch.setattr(probe, "_encode_token_ids", lambda *args: [1, 2, 3])
    monkeypatch.setattr(probe, "_to_input_tensor", lambda ids: ids)
    monkeypatch.setattr(probe, "_job_sequence_ids", lambda job: job)

    def run_job(generator, factory, input_ids, **kwargs):
        prompts.append(input_ids)
        observation = JobObservation(
            bucket_tokens=kwargs["bucket_tokens"], phase=kwargs["phase"],
            repetition=kwargs["repetition"], input_tokens=len(input_ids),
            cached_tokens=0, prefill_tokens=len(input_ids) - 1,
            ttft_ms=1, prefill_ms=1, generated_tokens=kwargs["max_new_tokens"],
            page_metrics={},
        )
        return observation, input_ids + [9] * kwargs["max_new_tokens"]

    monkeypatch.setattr(probe, "run_generator_job", run_job)
    config = ResidentProbeConfig(
        context_tokens=(1024,), appended_tokens=128,
        max_new_tokens=64, repetitions=1,
    )
    probe.run_resident_matrix(None, None, config, None)
    assert len(prompts[1]) + config.max_new_tokens == 1024
    assert len(prompts[1]) - len(prompts[0]) - config.max_new_tokens == 128
