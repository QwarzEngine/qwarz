from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import time

from qwasar_bench import decode_probe


def summarize_trace(trace: dict[str, object]) -> dict[str, object]:
    kernels = {}
    intervals = []
    memory_activity_count = 0
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        if category in ("gpu_memcpy", "gpu_memset"):
            memory_activity_count += 1
        if category != "kernel":
            continue
        duration = max(float(event.get("dur", 0)), 0.0)
        started = float(event["ts"])
        intervals.append((started, started + duration))
        name = event.get("name", "<unnamed>")
        aggregate = kernels.setdefault(name, {
            "name": name, "launch_count": 0, "duration_sum_us": 0.0,
            "max_duration_us": 0.0,
        })
        aggregate["launch_count"] += 1
        aggregate["duration_sum_us"] += duration
        aggregate["max_duration_us"] = max(aggregate["max_duration_us"], duration)
    busy_us = 0.0
    end = float("-inf")
    for started, finished in sorted(intervals):
        busy_us += max(finished - max(started, end), 0.0)
        end = max(end, finished)
    ordered = sorted(kernels.values(), key=lambda row: row["duration_sum_us"], reverse=True)
    return {
        "cuda_activities_present": bool(intervals or memory_activity_count),
        "cuda_kernel_events_present": bool(intervals),
        "kernel_launch_count": len(intervals),
        "kernel_duration_sum_ms": sum(row["duration_sum_us"] for row in ordered) / 1000,
        "kernel_busy_union_ms": busy_us / 1000,
        "kernel_time_semantics": (
            "Duration sum counts actual kernel executions and may exceed elapsed time "
            "when streams or devices overlap. Busy union merges kernel intervals across "
            "all streams and devices; it is not device utilization or inclusive operator time."
        ),
        "memory_activity_count": memory_activity_count,
        "distinct_kernel_count": len(kernels),
        "top_kernels": [{
            "name": row["name"], "launch_count": row["launch_count"],
            "duration_sum_ms": row["duration_sum_us"] / 1000,
            "mean_duration_us": row["duration_sum_us"] / row["launch_count"],
            "max_duration_us": row["max_duration_us"],
        } for row in ordered[:50]],
    }


@contextmanager
def annotate_phases(generator: object, job_type: type, record_function: object):
    targets = [
        (job_type, "prefill", "Job.prefill", "prefill"),
        (generator, "iterate_start_jobs", "generator.iterate_start_jobs", "allocation_restore"),
        (generator, "recurrent_checkpoint", "generator.recurrent_checkpoint", "recurrent_checkpoint"),
        (generator, "iterate_draftmodel_mtp_gen", "generator.iterate_draftmodel_mtp_gen", "mtp_draft"),
        (generator, "iterate_gen", "generator.iterate_gen", "target_verification"),
        (generator, "on_queue_drained", "generator.on_queue_drained", "queue_drain"),
    ]
    originals = []
    annotations = {"annotated_methods": [], "missing_methods": []}

    def wrap(original, label):
        @wraps(original)
        def annotated(*args, **kwargs):
            with record_function("qwasar::" + label):
                return original(*args, **kwargs)
        return annotated

    try:
        for target, method, qualified, phase in targets:
            original = getattr(target, method, None)
            if not callable(original):
                annotations["missing_methods"].append(qualified)
                continue
            local = method in vars(target)
            prior = vars(target).get(method)
            setattr(target, method, wrap(original, phase))
            originals.append((target, method, local, prior))
            annotations["annotated_methods"].append(qualified)
        yield annotations
    finally:
        for target, method, local, prior in reversed(originals):
            if local:
                setattr(target, method, prior)
            else:
                delattr(target, method)


def profile_sample(
    generator: object, tokenizer: object, prompt_ids: list[int],
    args: argparse.Namespace, seed: int, event_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    import torch

    activity = torch.profiler.ProfilerActivity
    if not torch.cuda.is_available() or activity.CUDA not in torch.profiler.supported_activities():
        raise RuntimeError("CUDA activity profiling is unavailable; refusing a CPU-only decode profile")
    from exllamav3 import Job

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    trace_path = output_directory / "chrome_trace.json"
    operators_path = output_directory / "operator_times.json"
    report_path = output_directory / "profile.json"
    for path in (trace_path, operators_path, report_path):
        if path.exists():
            raise FileExistsError(path)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with annotate_phases(generator, Job, torch.profiler.record_function) as annotations:
        with torch.profiler.profile(
            activities=[activity.CPU, activity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            sample = decode_probe.run_sample(
                generator, tokenizer, prompt_ids, args, seed, event_path,
            )
            torch.cuda.synchronize()
    capture_wall_ms = (time.perf_counter() - started) * 1000
    sample = {**sample, "profiling_overhead": True, "exclude_from_latency_percentiles": True}
    profiler.export_chrome_trace(str(trace_path))
    with trace_path.open() as source:
        trace_summary = summarize_trace(json.load(source))
    operators = [{
        "name": event.key, "calls": event.count,
        "self_cpu_ms": event.self_cpu_time_total / 1000,
        "inclusive_cpu_ms": event.cpu_time_total / 1000,
        "self_device_ms": event.self_device_time_total / 1000,
        "inclusive_device_ms": event.device_time_total / 1000,
    } for event in profiler.key_averages()]
    with operators_path.open("x") as output:
        json.dump(operators, output, indent=2)
        output.write("\n")
    report = {
        "schema_version": 1, "profiler": "torch.profiler",
        "status": "ok" if trace_summary["cuda_kernel_events_present"] else "cuda_trace_missing",
        "profiling_overhead": True, "exclude_from_latency_percentiles": True,
        "timing_warning": "All profiled latency measurements include instrumentation overhead.",
        "cuda_activity_requested": True,
        "capture_wall_ms": capture_wall_ms,
        "profiled_sample_elapsed_ms": sample["elapsed_ms"],
        "prompt_sha256": sample["prompt_sha256"], "seed": seed,
        "max_new_tokens": args.max_new_tokens,
        "trace_path": str(trace_path), "operators_path": str(operators_path),
        **annotations, **trace_summary,
        "operator_time_semantics": (
            "Operator and phase inclusive CPU/device times overlap and must not be summed "
            "as wall time. Device attribution follows the profiler's launch correlation. "
            "Kernel totals come only from exported CUDA kernel events."
        ),
        "phase_operators": [row for row in operators if row["name"].startswith("qwasar::")],
        "top_cpu_operators": sorted(operators, key=lambda row: row["self_cpu_ms"], reverse=True)[:30],
        "top_device_operators": sorted(operators, key=lambda row: row["self_device_ms"], reverse=True)[:30],
    }
    if not trace_summary["cuda_kernel_events_present"]:
        report["capture_warning"] = (
            "No CUDA kernel events were exported. This capture cannot substantiate GPU "
            "kernel timing; check CUPTI availability and profiler diagnostics."
        )
    with report_path.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    return {"sample": sample, "profile": report}
