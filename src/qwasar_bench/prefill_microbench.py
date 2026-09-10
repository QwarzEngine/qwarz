"""Replay a captured, already-appended paged attention call without loading a model."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics


TUNABLES = {"block_m", "block_n", "num_warps", "num_stages", "num_splits"}
BASELINE = {"name": "baseline", "staging": 1, "kwargs": {}}


def validate_candidates(candidates):
    names = {"baseline"}
    for candidate in candidates:
        name = candidate.get("name")
        settings = candidate.get("kwargs", {})
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Candidate names must be unique and not baseline")
        if candidate.get("staging") not in (0, 1):
            raise ValueError("staging must be 0 or 1")
        if not isinstance(settings, dict) or set(settings) - TUNABLES:
            raise ValueError("Only tile, warp, stage and split overrides are permitted")
        if any(type(value) is not int or value <= 0 for value in settings.values()):
            raise ValueError("Overrides must be positive integers")
        if candidate["staging"] == 0 and "block_n" in settings:
            raise ValueError("Direct quantized prefill overrides block_n internally")
        implementation = candidate.get("implementation", "triton")
        if implementation not in ("triton", "torch_flash"):
            raise ValueError("implementation must be triton or torch_flash")
        if implementation == "torch_flash" and (candidate["staging"] != 1 or settings):
            raise ValueError("torch_flash requires staging=1 and empty tile overrides")
        names.add(name)


def default_candidates(stage):
    if stage == "coarse":
        configurations = itertools.product((64, 128), (32, 64), (4, 8), (1, 2), (None, 1, 2, 4))
    else:
        configurations = itertools.product((32, 64, 128), (32,), (4, 8), (1, 2, 3), (1, 2, 4))
    candidates = []
    for block_m, block_n, warps, stages, splits in configurations:
        settings = dict(block_m=block_m, block_n=block_n, num_warps=warps, num_stages=stages)
        if splits is not None:
            settings["num_splits"] = splits
        candidates.append({
            "name": f"staged_m{block_m}_n{block_n}_w{warps}_s{stages}_split{splits if splits is not None else 'auto'}",
            "staging": 1, "kwargs": settings,
        })
    for splits in (1, 2, 4):
        candidates.append({"name": f"direct_split{splits}", "staging": 0,
                           "kwargs": dict(block_m=64, num_warps=8, num_stages=2,
                                          num_splits=splits)})
    return candidates


def schedule(candidates, rounds, seed):
    generator = random.Random(seed)
    result = [(0, "baseline")]
    for round_index in range(rounds):
        names = [candidate["name"] for candidate in candidates]
        generator.shuffle(names)
        for name in names:
            result.extend(((round_index, name), (round_index, "baseline")))
    return result


@contextmanager
def staging_context(module, staging):
    original = module._qc_staging
    module._qc_staging = bool(staging)
    try:
        yield
    finally:
        module._qc_staging = original


def classify_error(error):
    message = str(error).lower()
    if "out of memory" in message:
        return "oom"
    if any(term in message for term in ("illegal memory", "device-side assert", "misaligned address", "launch failure")):
        return "fatal_cuda"
    if any(term in message for term in ("out of resource", "outofresource", "compilation", "compile", "ptxas")):
        return "compile_failure"
    if "cuda error" in message:
        return "fatal_cuda"
    return "error"


def summarize(records):
    candidates = {}
    for name in dict.fromkeys(record["name"] for record in records):
        matching = [record for record in records if record["name"] == name]
        timings = [value for record in matching for value in record.get("times_ms", [])]
        finite = bool(timings) and all(math.isfinite(value) and value > 0 for value in timings)
        eligible = finite and all(record["status"] == "ok" for record in matching)
        ordered = sorted(timings) if finite else []
        position = (len(ordered) - 1) * .95
        lower = math.floor(position)
        upper = math.ceil(position)
        candidates[name] = {
            "eligible": eligible, "trials": len(matching),
            "median_ms": statistics.median(ordered) if ordered else None,
            "p95_ms": (ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)) if ordered else None,
            "failures": [record for record in matching if record["status"] != "ok"],
        }
    eligible = [name for name, result in candidates.items() if result["eligible"] and name != "baseline"]
    baseline = candidates.get("baseline", {})
    best_candidate = min(eligible, key=lambda name: candidates[name]["median_ms"]) if eligible else None
    winner = best_candidate if (best_candidate and baseline.get("eligible")
                                and candidates[best_candidate]["median_ms"] < baseline["median_ms"]) else None
    baseline_trials = [statistics.median(record["times_ms"]) for record in records
                       if record["name"] == "baseline" and record["status"] == "ok"]
    return {"winner": winner, "best_candidate": best_candidate, "candidates": candidates, "records": records,
            "baseline_drift_ratio": max(baseline_trials) / min(baseline_trials) if baseline_trials else None,
            "quality_note": "Attention-output tolerance is a screening gate, not proof of model quality."}


def to_device(value, torch, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, torch, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, torch, device) for item in value)
    return value


def prepare_kwargs(kwargs, query_length=None):
    if any(kwargs.get(name) is not None for name in ("k", "v", "k_new", "v_new")):
        raise ValueError("Capture must use an already-appended cache without new K/V")
    if kwargs.get("qc") is None or not kwargs.get("causal"):
        raise ValueError("Expected a causal quantized-cache capture")
    result = dict(kwargs)
    result["out"] = None
    if query_length is not None:
        if not 0 < query_length <= kwargs["q"].shape[1]:
            raise ValueError("query_length must fit the captured query")
        result["q"] = kwargs["q"][:, -query_length:].contiguous()
    return result


def execution_settings(module, kwargs, candidate):
    if candidate.get("implementation", "triton") == "torch_flash":
        return {"implementation": "torch_flash", "actual_staging": 1, "effective_block_n": None,
                "effective_block_n_note": "native Flash Attention; Triton tiles do not apply"}
    query = kwargs["q"]
    actual_staging = bool(candidate["staging"] == 1
                          and query.shape[1] >= module._qc_prefill_two_pass_min_q
                          and query.shape[0] * kwargs["block_table"].shape[1] <= kwargs["k_cache"].shape[0])
    block_n = candidate["kwargs"].get("block_n", kwargs.get("block_n"))
    if not actual_staging:
        head_dim = query.shape[-1]
        block_n = max(16, min(128, 16384 // head_dim))
        if head_dim >= 256 and kwargs["qc"][2] + kwargs["qc"][3] >= 13 and block_n > 16:
            block_n //= 2
    return {"implementation": "triton", "actual_staging": int(actual_staging), "effective_block_n": block_n,
            "effective_block_n_note": "default chosen by installed function" if block_n is None else None}


def numerical_metrics(output, reference, atol, rtol):
    import torch

    output = output.detach().float().cpu()
    reference = reference.detach().float().cpu()
    finite = bool(torch.isfinite(output).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "passed": False, "max_abs": None, "rmse": None, "relative_l2": None}
    difference = output - reference
    reference_norm = float(torch.linalg.vector_norm(reference))
    difference_norm = float(torch.linalg.vector_norm(difference))
    return {"finite": True, "passed": bool(torch.allclose(output, reference, atol=atol, rtol=rtol)),
            "max_abs": float(difference.abs().max()), "rmse": float(difference.square().mean().sqrt()),
            "relative_l2": difference_norm / max(reference_norm, 1e-30)}


def measure(module, kwargs, candidate, reference, torch, repetitions, atol, rtol):
    record = {**candidate, **execution_settings(module, kwargs, candidate)}
    kernel = module.paged_attn_triton_prefill
    if candidate.get("implementation", "triton") == "torch_flash":
        from .prefill_flash import torch_flash_prefill

        kernel = torch_flash_prefill
    with staging_context(module, candidate["staging"]):
        arguments = {**kwargs, **candidate["kwargs"]}
        torch.cuda.synchronize()
        kernel(**arguments)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = kernel(**arguments)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        record.update(times_ms=times, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        record["numerical"] = numerical_metrics(output, reference, atol, rtol)
        record["status"] = "ok" if record["numerical"]["passed"] else "numerical_failure"
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--stage", choices=("coarse", "refine"), default="coarse")
    parser.add_argument("--query-length", type=int)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=.01)
    parser.add_argument("--rtol", type=float, default=.01)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.rounds < 1 or not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        parser.error("repetitions/rounds must be positive and tolerances finite and nonnegative")
    candidates = json.loads(args.candidates.read_text()) if args.candidates else default_candidates(args.stage)
    validate_candidates(candidates)
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("trials.jsonl", "summary.json", "metadata.json")):
        parser.error("output directory already contains benchmark results")
    import torch
    import triton

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        parser.error("CUDA is required to replay a captured kernel")
    torch.cuda.set_device(args.device)
    module = importlib.import_module("exllamav3.modules.attention_fn.triton_paged")
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    kwargs = to_device(prepare_kwargs(capture["kwargs"], args.query_length), torch, args.device)
    with args.capture.open("rb") as capture_stream:
        capture_sha256 = hashlib.file_digest(capture_stream, "sha256").hexdigest()
    metadata = {"capture_metadata": capture.get("metadata", {}), "capture_path": str(args.capture.resolve()),
                "capture_sha256": capture_sha256,
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "q_shape": list(kwargs["q"].shape), "torch": torch.__version__, "triton": triton.__version__,
                "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                "source_path": module.__file__, "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                "seed": args.seed, "rounds": args.rounds, "repetitions": args.repetitions,
                "atol": args.atol, "rtol": args.rtol, "candidates": candidates}
    if any(candidate.get("implementation") == "torch_flash" for candidate in candidates):
        from . import prefill_flash

        metadata["flash_source_path"] = prefill_flash.__file__
        metadata["flash_source_sha256"] = hashlib.sha256(Path(prefill_flash.__file__).read_bytes()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    records = []
    failed = set()
    lookup = {candidate["name"]: candidate for candidate in [BASELINE, *candidates]}
    fatal = False
    with (args.output / "trials.jsonl").open("w") as stream, torch.inference_mode():
        try:
            with staging_context(module, 1):
                torch.cuda.synchronize()
                reference = module.paged_attn_triton_prefill(**kwargs).detach().clone()
                torch.cuda.synchronize()
            for round_index, name in schedule(candidates, args.rounds, args.seed):
                if name in failed:
                    continue
                try:
                    record = measure(module, kwargs, lookup[name], reference, torch,
                                     args.repetitions, args.atol, args.rtol)
                except Exception as error:
                    record = {**lookup[name], "status": classify_error(error), "error": str(error)}
                    failed.add(name)
                    if record["status"] == "oom":
                        torch.cuda.empty_cache()
                record["round"] = round_index
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                (args.output / "summary.json").write_text(json.dumps(summarize(records), indent=2, allow_nan=False))
                if record["status"] == "fatal_cuda" or (name == "baseline" and record["status"] != "ok"):
                    fatal = True
                    break
        except Exception as error:
            fatal = True
            record = {"name": "baseline", "status": classify_error(error), "error": str(error)}
            records.append(record)
            stream.write(json.dumps(record) + "\n")
        finally:
            summary = {**summarize(records), "aborted": fatal, "metadata": metadata}
            if fatal:
                summary["winner"] = None
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
