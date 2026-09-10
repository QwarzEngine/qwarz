from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import sys
from types import SimpleNamespace

import pytest


def load_probe():
    try:
        return importlib.import_module("qwasar_bench.profile_probe")
    except ModuleNotFoundError:
        pytest.fail("the profiling helper must be importable without torch")


def test_kernel_accounting_excludes_cpu_ranges_and_merges_overlap():
    probe = load_probe()
    report = probe.summarize_trace({"traceEvents": [
        {"ph": "X", "cat": "user_annotation", "name": "prefill", "ts": 0, "dur": 900},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 10, "dur": 800},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 100, "dur": 100},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 150, "dur": 100},
        {"ph": "X", "cat": "kernel", "name": "unknown", "ts": 300, "dur": 50},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH", "ts": 400, "dur": 20},
        {"ph": "M", "cat": "kernel", "name": "metadata"},
    ]})
    assert report["cuda_activities_present"] is True
    assert report["cuda_kernel_events_present"] is True
    assert report["kernel_launch_count"] == 3
    assert report["kernel_duration_sum_ms"] == 0.25
    assert report["kernel_busy_union_ms"] == 0.2
    assert report["top_kernels"][0] == {
        "name": "gemm", "launch_count": 2, "duration_sum_ms": 0.2,
        "mean_duration_us": 100.0, "max_duration_us": 100.0,
    }
    assert report["memory_activity_count"] == 1


def test_cpu_only_trace_is_not_cuda_success():
    probe = load_probe()
    report = probe.summarize_trace({"traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "cudaLaunchKernel", "ts": 0, "dur": 100},
    ]})
    assert report["cuda_activities_present"] is False
    assert report["cuda_kernel_events_present"] is False
    assert report["kernel_launch_count"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_phase_wrappers_preserve_return_values_and_restore_after_failure(fail):
    probe = load_probe()
    observed = []

    @contextmanager
    def record_function(name):
        observed.append(name)
        yield

    class Job:
        def prefill(self, value):
            return value + 1

    class Generator:
        def iterate_gen(self, value):
            return value * 2

    generator = Generator()
    original = Job.prefill
    try:
        with probe.annotate_phases(generator, Job, record_function) as annotations:
            assert Job().prefill(2) == 3
            assert generator.iterate_gen(3) == 6
            assert "Job.prefill" in annotations["annotated_methods"]
            assert "generator.recurrent_checkpoint" in annotations["missing_methods"]
            if fail:
                raise ValueError("decode failed")
    except ValueError:
        assert fail
    assert Job.prefill is original
    assert "iterate_gen" not in vars(generator)
    assert len(observed) == 2
    assert any("target_verification" in name for name in observed)


@pytest.mark.parametrize("has_kernels", [False, True])
def test_profile_artifacts_mark_overhead_and_missing_cuda(monkeypatch, tmp_path, has_kernels):
    probe = load_probe()
    trace = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "test_kernel", "ts": 1, "dur": 10},
    ] if has_kernels else []}

    class Profiler:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def export_chrome_trace(self, path):
            with open(path, "w") as output:
                json.dump(trace, output)

        def key_averages(self):
            return [SimpleNamespace(
                key="aten::test", count=2, self_cpu_time_total=10,
                cpu_time_total=30, self_device_time_total=20, device_time_total=40,
            )]

    @contextmanager
    def record_function(name):
        yield

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, synchronize=lambda: None),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
            supported_activities=lambda: {"CPU", "CUDA"},
            profile=lambda **kwargs: Profiler(), record_function=record_function,
        ),
    ))
    monkeypatch.setitem(sys.modules, "exllamav3", SimpleNamespace(Job=type("Job", (), {})))
    monkeypatch.setattr(probe.decode_probe, "run_sample", lambda *args: {
        "elapsed_ms": 7.0, "prompt_sha256": "hash", "completion": "text",
    })
    result = probe.profile_sample(
        SimpleNamespace(), None, [1, 2], SimpleNamespace(max_new_tokens=32),
        42, tmp_path / "events.jsonl", tmp_path / "profile",
    )
    assert result["sample"]["exclude_from_latency_percentiles"] is True
    assert result["sample"]["profiling_overhead"] is True
    report = result["profile"]
    assert report["status"] == ("ok" if has_kernels else "cuda_trace_missing")
    assert report["cuda_kernel_events_present"] is has_kernels
    assert report["exclude_from_latency_percentiles"] is True
    assert report["top_cpu_operators"][0]["self_cpu_ms"] == 0.01
    assert report["top_device_operators"][0]["self_device_ms"] == 0.02
    assert json.loads((tmp_path / "profile/profile.json").read_text())["status"] == report["status"]
    assert (tmp_path / "profile/chrome_trace.json").exists()
    assert (tmp_path / "profile/operator_times.json").exists()


def test_unsupported_cuda_profiler_fails_before_decoding(monkeypatch, tmp_path):
    probe = load_probe()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        profiler=SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
            supported_activities=lambda: {"CPU"},
        ),
    ))
    with pytest.raises(RuntimeError, match="CUDA"):
        probe.profile_sample(None, None, [], None, 0, tmp_path / "events", tmp_path / "profile")
