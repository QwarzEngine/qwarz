"""Reproducible, opt-in NVFP4 projection study on the production loader.

Run GPU modes only under the repository's managed service window.
No production settings or installed dependencies are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from . import nvfp4_projections as projections

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/home/rekeyea/models/Qwen3.8-27B-EXL3-5.0bpw")


def write(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def setup():
    from qwasar_runtime import hybrid
    hybrid.prepare_environment()
    import torch
    from qwasar_runtime.engine import configure_gpu
    configure_gpu(torch)
    torch.manual_seed(923)
    return torch


def preflight(out):
    from qwasar_runtime import hybrid
    pin = hybrid.donor_pin()
    records = [projections.verify_file(projections.CHECKPOINT, projections.SHA256)]
    records.extend(projections.verify_file(Path(pin["path"]) / name, value["sha256"])
                   for name, value in pin["shards"].items())
    write(out / "verified-donors.json", records)


def unpack_weights(tensors, torch):
    # Independent decoder of the *checkpoint*, not NativeLinear's swizzled operands.
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       device="cuda", dtype=torch.float32)
    packed = tensors["weight_packed"].cuda()
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    return (lut[codes] * tensors["weight_scale"].cuda().float().repeat_interleave(16, -1)
            / tensors["weight_global_scale"].cuda().float())


def relative_rms(actual, reference):
    return ((actual.float() - reference).square().mean()
            / reference.square().mean().clamp_min(1e-30)).sqrt().item()


def oracle(out):
    torch = setup()
    from safetensors import safe_open
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    with safe_open(str(projections.CHECKPOINT), framework="pt", device="cpu") as handle:
        keys = [key.removesuffix(".weight_packed") for key in handle.keys()
                if key.endswith(".weight_packed")
                and projections.selected(key.removesuffix(".weight_packed"), "all")
                and int(key.split(".")[2]) in (0, 3, 30, 31, 60, 63)]
        for key in sorted(keys):
            tensors = projections.read_tensors(handle, key)
            native = projections.PaddedLinear(key, tensors, "cuda", torch.float32)
            weight = unpack_weights(tensors, torch)
            for m in (1, 7, 128, 2048):
                x = torch.randn((m, native.in_features), device="cuda", dtype=torch.float16)
                reference = x.float() @ weight.T
                y = native.forward(x)
                error = relative_rms(y, reference)
                # Mechanical sanity bound, NOT a quality equivalence threshold.
                passed = bool(torch.isfinite(y).all()) and error < .25
                row = {"key": key, "m": m, "relative_rms_unquantized_activation": error,
                       "passed": passed, "out_dtype": str(y.dtype)}
                rows.append(row)
                write(out / f"check-{len(rows):03}.json", row)
                if not passed:
                    raise RuntimeError(f"independent activation oracle failed: {row}")
            corrupted = dict(tensors)
            corrupted["weight_global_scale"] = tensors["weight_global_scale"] * 1000
            red = projections.PaddedLinear(key, corrupted, "cuda", torch.float32)
            red_error = relative_rms(red.forward(x), reference)
            if not red_error > .9:
                raise RuntimeError(f"oracle did not detect corrupted scale: {key}")
            write(out / f"red-{key}.json", {"relative_rms": red_error, "detected": True})
            del native, red, weight, x, y, reference
            torch.cuda.empty_cache()
            print(json.dumps({"oracle": key, "passed": True}), flush=True)
    if len(keys) != 27:
        raise RuntimeError(f"expected 27 representative projections, got {len(keys)}")
    write(out / "summary.json", {"checks": len(rows), "red_controls": len(keys),
                                "max_relative_rms": max(r["relative_rms_unquantized_activation"]
                                                        for r in rows), "passed": True})


def benchmark(out, profile, contexts, repeats, tokens):
    torch = setup()
    from qwasar_runtime.engine import ExLlamaBackend
    from qwasar_bench.decode_probe import configure_mtp_policy, run_sample
    from qwasar_bench.fidelity_probe import fresh_generator

    with projections.replace(profile) as replaced:
        backend = ExLlamaBackend(MODEL, 262144, prefill="xqa")
    try:
        if len(replaced) != projections.COUNTS[profile]:
            raise RuntimeError(f"projection count: {len(replaced)}")
        cfg = backend.config
        if not (cfg["vision"] and cfg["mtp_head"]["installed"]
                and cfg["rendezvous"]["installed"] and cfg["rendezvous"]["gpu_embedding"]
                and cfg["mlp_graphs"] == 64):
            raise RuntimeError(f"production acceleration missing: {cfg}")
        g, tokenizer = backend.generator, backend.tokenizer
        configure_mtp_policy(g, "fixed6")
        write(out / "configuration.json", {"profile": profile, "production": cfg,
              "replaced": replaced, "torch": torch.__version__, "python": platform.python_version(),
              "command": sys.argv, "contexts": contexts, "repeats": repeats,
              "tokens": tokens, "timing": "unprofiled; bounded output length; not task quality",
              "module_paths": {name: getattr(sys.modules.get(name), "__file__", None)
                               for name in ("flashinfer", "exllamav3", "torch")}})
        write(out / "block-paths.json", [
            {"key": m.key, "class": type(m).__name__,
             "batched_call": getattr(m, "bc", None) is not None,
             "bc_split": bool(getattr(m, "bc_split", False))}
            for m in g.model
            if type(m).__name__ in ("Attention", "GatedDeltaNet")])
        # Same archived tokens isolate implementation cost, NOT held-out quality.
        source = ROOT / "results/20260910-quality-gate/prompts-phase2.json"
        cells = json.loads(source.read_text())
        longest = max(cells, key=lambda cell: len(cell["prompt_ids"]))["prompt_ids"]
        if max(contexts) + tokens + 8 > 262144:
            raise ValueError("context must reserve output and speculative scratch")
        from exllamav3 import Job
        from exllamav3.generator.sampler.presets import GreedySampler
        write(out / "prompt-provenance.json", {"source": str(source),
              "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "quality_evidence": False})
        rows = []
        with backend.tuning():
            job = Job(input_ids=torch.tensor([longest[:8200]]), max_new_tokens=64,
                      sampler=GreedySampler(), seed=923, stop_conditions=[])
            g.enqueue(job)
            while g.num_remaining_jobs():
                g.iterate()
            fresh_generator(g)
            for context in contexts:
                # Keep the actual question at the end even for the short shapes.
                ids = longest[:context - 1024] + longest[-1024:]
                for rep in range(repeats):
                    fresh_generator(g)
                    label = f"c{context}-r{rep}"
                    args = argparse.Namespace(max_new_tokens=tokens, thinking="medium", sampler="recommended")
                    sample = run_sample(g, tokenizer, ids, args, 923 + rep,
                                        out / f"{label}-events.jsonl")
                    torch.cuda.synchronize()
                    sample.update(label=label, context=context, repetition=rep, profile=profile,
                                  peak_allocated=torch.cuda.max_memory_allocated(),
                                  peak_reserved=torch.cuda.max_memory_reserved(),
                                  prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                    write(out / f"{label}-sample.json", sample)
                    row = {key: sample.get(key) for key in (
                        "label", "context", "repetition", "ttft_ms", "elapsed_ms",
                        "decode_tokens_per_second", "generated_tokens", "draft_acceptance",
                        "peak_allocated", "peak_reserved", "cached_tokens", "truncated")}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        write(out / "summary.json", {"profile": profile, "samples": rows, "completed": True})
    finally:
        backend._lifetime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "oracle", "benchmark", "screen"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(projections.COUNTS), default="control")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--profiles", nargs="+", choices=tuple(projections.COUNTS),
                        default=["control", "attention", "attention", "control",
                                 "gdn_input", "gdn_output", "control"])
    args = parser.parse_args()
    if args.repeats < 1 or args.tokens < 1 or any(c < 2048 for c in args.contexts):
        parser.error("positive repeats/tokens and contexts >=2048 required")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "sources.json", {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path(projections.__file__)]
    })
    if args.mode == "preflight":
        preflight(args.output)
    elif args.mode == "oracle":
        oracle(args.output)
    elif args.mode == "screen":
        processes = []
        for index, profile in enumerate(args.profiles):
            target = args.output / f"{index:02}-{profile}"
            command = [sys.executable, "-m", "qwasar_bench.nvfp4_study", "benchmark",
                       "--output", str(target), "--profile", profile,
                       "--contexts", *map(str, args.contexts), "--repeats", str(args.repeats),
                       "--tokens", str(args.tokens)]
            with (args.output / f"{index:02}-{profile}.log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=900)
            processes.append({"profile": profile, "output": str(target),
                              "returncode": result.returncode, "command": command})
            write(args.output / f"process-{index:02}.json", processes[-1])
            if result.returncode:
                raise RuntimeError(f"screen failed: {target}")
        write(args.output / "completed.json", {"processes": processes, "completed": True})
    else:
        benchmark(args.output, args.profile, args.contexts, args.repeats, args.tokens)


if __name__ == "__main__":
    main()
