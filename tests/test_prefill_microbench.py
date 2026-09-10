from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from qwasar_bench import prefill_microbench as bench


def test_schedule_rechecks_baseline_and_runs_each_candidate_every_round():
    candidates = [{"name": "a"}, {"name": "b"}]
    schedule = bench.schedule(candidates, rounds=3, seed=42)
    assert schedule[0][1] == "baseline"
    assert schedule[-1][1] == "baseline"
    assert sum(name == "a" for _, name in schedule) == 3
    assert sum(name == "b" for _, name in schedule) == 3
    assert all(schedule[index + 1][1] == "baseline"
               for index, (_, name) in enumerate(schedule) if name != "baseline")


@pytest.mark.parametrize("candidate", [
    {"name": "bad", "staging": 2, "kwargs": {}},
    {"name": "bad", "staging": 0, "kwargs": {"block_n": 32}},
    {"name": "bad", "staging": 1, "kwargs": {"q": 32}},
    {"name": "baseline", "staging": 1, "kwargs": {}},
    {"name": "bad", "staging": 1, "kwargs": {"num_splits": 0}},
    {"name": "bad", "staging": 1, "kwargs": {}, "implementation": "unknown"},
    {"name": "bad", "staging": 0, "kwargs": {}, "implementation": "torch_flash"},
    {"name": "bad", "staging": 1, "kwargs": {"block_m": 64}, "implementation": "torch_flash"},
])
def test_candidates_reject_unsafe_or_misleading_overrides(candidate):
    with pytest.raises(ValueError):
        bench.validate_candidates([candidate])


def test_candidate_staging_restored_when_kernel_fails():
    module = SimpleNamespace(_qc_staging=True)
    with pytest.raises(RuntimeError):
        with bench.staging_context(module, 0):
            assert module._qc_staging is False
            raise RuntimeError("compilation failed")
    assert module._qc_staging is True


def test_summary_preserves_failures_and_rejects_partially_failed_candidate():
    records = [
        {"name": "baseline", "status": "ok", "times_ms": [10.0, 12.0]},
        {"name": "fast", "status": "ok", "times_ms": [2.0, 4.0]},
        {"name": "fast", "status": "numerical_failure", "times_ms": [1.0]},
        {"name": "safe", "status": "ok", "times_ms": [5.0, 7.0]},
        {"name": "oom", "status": "oom", "error": "out of memory"},
    ]
    result = bench.summarize(records)
    assert result["winner"] == "safe"
    assert result["candidates"]["fast"]["eligible"] is False
    assert result["candidates"]["safe"]["median_ms"] == 6.0
    assert result["candidates"]["safe"]["p95_ms"] == pytest.approx(6.9)
    assert result["records"] == records


@pytest.mark.parametrize(("message", "expected"), [
    ("CUDA out of memory", "oom"),
    ("out of resource: shared memory", "compile_failure"),
    ("CUDA error: an illegal memory access was encountered", "fatal_cuda"),
    ("CUDA error: device-side assert triggered", "fatal_cuda"),
])
def test_errors_distinguish_recoverable_compilation_from_poisoned_cuda(message, expected):
    assert bench.classify_error(RuntimeError(message)) == expected


def test_default_grid_does_not_claim_direct_block_n_overrides_are_effective():
    candidates = bench.default_candidates("coarse")
    bench.validate_candidates(candidates)
    assert any(candidate["staging"] == 0 for candidate in candidates)
    assert any(candidate["staging"] == 1 for candidate in candidates)
    assert all("block_n" not in candidate["kwargs"] for candidate in candidates
               if candidate["staging"] == 0)


def test_coarse_grid_compares_automatic_splits_and_both_pipeline_and_warp_options():
    candidates = bench.default_candidates("coarse")
    staged = [candidate["kwargs"] for candidate in candidates if candidate["staging"] == 1]
    assert {settings["num_warps"] for settings in staged} == {4, 8}
    assert {settings["num_stages"] for settings in staged} == {1, 2}
    assert {settings.get("num_splits") for settings in staged} == {None, 1, 2, 4}


def test_nonfinite_timings_cannot_win():
    result = bench.summarize([
        {"name": "baseline", "status": "ok", "times_ms": [2.0]},
        {"name": "bad", "status": "ok", "times_ms": [math.nan]},
    ])
    assert result["winner"] is None


def test_slower_candidate_is_reported_but_never_selected_over_baseline():
    result = bench.summarize([
        {"name": "baseline", "status": "ok", "times_ms": [2.0, 4.0]},
        {"name": "slower", "status": "ok", "times_ms": [5.0, 6.0]},
    ])
    assert result["winner"] is None
    assert result["best_candidate"] == "slower"


def test_subset_uses_last_queries_without_changing_bottom_right_cache_alignment():
    torch = pytest.importorskip("torch")
    query = torch.arange(32).reshape(1, 8, 2, 2)
    kwargs = {"q": query, "k": None, "v": None, "qc": (1, 2, 8, 4),
              "causal": True, "pre_appended_len": 8, "out": query.clone()}
    result = bench.prepare_kwargs(kwargs, query_length=3)
    assert result["q"].flatten().tolist() == list(range(20, 32))
    assert result["q"].is_contiguous()
    assert result["pre_appended_len"] == 8
    assert result["out"] is None
    assert kwargs["q"].shape[1] == 8
    with pytest.raises(ValueError):
        bench.prepare_kwargs(kwargs, query_length=9)
    with pytest.raises(ValueError):
        bench.prepare_kwargs({**kwargs, "k": query})


def test_numerical_gate_reports_real_error_and_rejects_nan():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([3., 4.])
    result = bench.numerical_metrics(torch.tensor([3., 5.]), reference, .01, .01)
    assert result["passed"] is False
    assert result["max_abs"] == 1.0
    assert result["rmse"] == pytest.approx(math.sqrt(.5))
    assert result["relative_l2"] == pytest.approx(.2)
    assert bench.numerical_metrics(reference, reference, .01, .01)["passed"] is True
    assert bench.numerical_metrics(torch.tensor([math.nan, 4.]), reference, .01, .01)["finite"] is False


def test_short_query_records_effective_direct_tiles_even_if_staging_requested():
    kwargs = {"q": SimpleNamespace(shape=(1, 128, 24, 256)),
              "block_table": SimpleNamespace(shape=(1, 200)),
              "k_cache": SimpleNamespace(shape=(256, 256)), "qc": (None, None, 8, 4)}
    candidate = {"staging": 1, "kwargs": {"block_n": 16}}
    result = bench.execution_settings(SimpleNamespace(_qc_prefill_two_pass_min_q=256), kwargs, candidate)
    assert result["actual_staging"] == 0
    assert result["effective_block_n"] == 64


def test_unknown_cuda_runtime_error_aborts_instead_of_reusing_poisoned_context():
    assert bench.classify_error(RuntimeError("CUDA error: unknown error")) == "fatal_cuda"


@pytest.mark.parametrize("use_flash", [False, True])
def test_cli_preserves_reference_failure_and_hashes_capture_without_replaying_more(monkeypatch, tmp_path, use_flash):
    from contextlib import nullcontext
    import hashlib
    import json
    import sys

    capture_path = tmp_path / "capture.pt"
    capture_path.write_bytes(b"capture-content")
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps([
        {"name": "flash", "staging": 1, "kwargs": {}, "implementation": "torch_flash"}
    ] if use_flash else []))
    output_path = tmp_path / "output"

    def load(path, *, map_location, weights_only):
        assert path == capture_path
        assert map_location == "cpu"
        assert weights_only is True
        return {"kwargs": {"q": SimpleNamespace(shape=(1, 8, 2, 2)),
                           "qc": (1, 2, 8, 4), "causal": True}, "metadata": {"origin": "test"}}

    def fail_reference(**kwargs):
        raise RuntimeError("CUDA error: illegal memory access")

    torch = SimpleNamespace(
        Tensor=type("Tensor", (), {}), load=load, __version__="cpu-test",
        version=SimpleNamespace(cuda="test"), inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: True, set_device=lambda device: None,
                             synchronize=lambda: None, get_device_name=lambda: "CPU fixture"),
    )
    module = SimpleNamespace(__file__=__file__, _qc_staging=0,
                             paged_attn_triton_prefill=fail_reference)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "exllamav3.modules.attention_fn.triton_paged", module)
    assert bench.main(["--capture", str(capture_path), "--output", str(output_path),
                       "--candidates", str(candidates_path)]) == 1
    summary = json.loads((output_path / "summary.json").read_text())
    records = [json.loads(line) for line in (output_path / "trials.jsonl").read_text().splitlines()]
    assert summary["aborted"] is True
    assert summary["winner"] is None
    assert len(records) == 1
    assert records[0]["status"] == "fatal_cuda"
    assert module._qc_staging == 0
    assert summary["metadata"]["capture_sha256"] == hashlib.sha256(b"capture-content").hexdigest()
    assert len(summary["metadata"]["harness_sha256"]) == 64
    if use_flash:
        from pathlib import Path
        from qwasar_bench import prefill_flash

        assert summary["metadata"]["flash_source_sha256"] == hashlib.sha256(
            Path(prefill_flash.__file__).read_bytes()).hexdigest()
    else:
        assert "flash_source_sha256" not in summary["metadata"]


@pytest.mark.parametrize("implementation", ["triton", "torch_flash"])
def test_measure_dispatches_requested_backend_and_reports_short_query_staging(monkeypatch, implementation):
    torch = pytest.importorskip("torch")
    from qwasar_bench import prefill_flash

    calls = []

    def triton_kernel(**kwargs):
        calls.append("triton")
        return torch.tensor([3., 4.])

    def flash_kernel(**kwargs):
        calls.append("torch_flash")
        return torch.tensor([3., 4.])

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing is True

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 2.

    timer = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
        Event=Event, max_memory_allocated=lambda: 1024, max_memory_reserved=lambda: 2048,
    ))
    module = SimpleNamespace(_qc_staging=0, _qc_prefill_two_pass_min_q=256,
                             paged_attn_triton_prefill=triton_kernel)
    kwargs = {"q": SimpleNamespace(shape=(1, 113, 24, 256)),
              "block_table": SimpleNamespace(shape=(1, 200)),
              "k_cache": SimpleNamespace(shape=(256, 256)), "qc": (None, None, 8, 4)}
    candidate = {"name": "candidate", "staging": 1, "kwargs": {}, "implementation": implementation}
    bench.validate_candidates([candidate])
    monkeypatch.setattr(prefill_flash, "torch_flash_prefill", flash_kernel)
    result = bench.measure(module, kwargs, candidate, torch.tensor([3., 4.]), timer, 2, .01, .01)
    assert calls == [implementation, implementation, implementation]
    assert result["status"] == "ok"
    assert result["times_ms"] == [2., 2.]
    assert result["actual_staging"] == int(implementation == "torch_flash")
    assert result["effective_block_n"] == (None if implementation == "torch_flash" else 64)
    assert result["implementation"] == implementation
    assert module._qc_staging == 0
