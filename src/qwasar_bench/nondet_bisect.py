"""Localize the first nondeterministic module in the 32K+ decode stack (2026-09-23).

The 2026-09-22/23 draft-graph campaigns showed the production stack is bit-stable
at 4K (single-chunk prefill, no PRIMS) but diverges run-to-run at 32K+ even within
one greedy arm. This bisect hooks every module of the target and draft models,
fingerprints each forward call's input and output, runs the identical greedy job
twice on a fresh cache, and reports the first module whose OUTPUT differs while its
INPUT matches: that module is the nondeterminism source, everything after it only
propagates the difference.

Fingerprints are order-sensitive: a strided byte hash plus fp64 moments. They are
cheap enough to keep prefill overhead near one minute per pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .mtp_study import backend, warmup
from .nvfp4_study import ROOT, setup, write

STRIDE = 64  # sample every 64th byte for the hash
PREFILL_QLEN = 16  # calls wider than this are prefill chunks


def fingerprint(torch, tensor):
    """Order-sensitive fingerprint of one tensor, or a dtype marker for non-float tensors."""
    if tensor is None:
        return None
    x = tensor.detach()
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        # Integer tensors (token ids): hash the bytes; two id lists with the same
        # shape must not compare equal, or acceptance divergence shows up as a
        # false "source" at the embedding and MTP inputs.
        raw = x.contiguous().view(torch.uint8)
        return {"dtype": str(x.dtype), "shape": list(x.shape),
                "sha": hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()}
    raw = x.contiguous().view(torch.uint8)
    sample = raw[::STRIDE].cpu().numpy().tobytes()
    wide = x.double()
    return {
        "sha": hashlib.sha256(sample).hexdigest(),
        "sum": wide.sum().item(),
        "sq": wide.square().sum().item(),
        "shape": list(x.shape),
    }


class Recorder:
    """Per-instance forward wrappers on every module of the target and draft model.

    exllamav3 modules are plain ABCs (no nn.Module hooks), so wrap the instance
    attribute; Model calls ``module.forward(x, params)``, which resolves through
    the instance. Inputs are fingerprinted BEFORE the call because some blocks
    update the residual in place (``x += y1``).
    """

    def __init__(self, torch):
        self.torch = torch
        self.calls = {}
        self.patched = []
        self.seq = 0

    def attach(self, model, prefix):
        for index, module in enumerate(model.modules):
            key = f"{prefix}[{index:03d}]{getattr(module, 'key', type(module).__name__)}"
            self.calls[key] = []
            original = module.forward
            module.forward = self._wrap(key, original)
            self.patched.append((module, original))

    def _wrap(self, key, original):
        def wrapped(x, *args, **kwargs):
            inp = x if self.torch.is_tensor(x) else next(
                (a for a in args if self.torch.is_tensor(a)), None)
            params = next((a for a in args if isinstance(a, dict)),
                          kwargs.get("params") or {})
            in_fp = fingerprint(self.torch, inp)
            output = original(x, *args, **kwargs)
            out = output if self.torch.is_tensor(output) else (
                output[0] if isinstance(output, (tuple, list)) and output
                and self.torch.is_tensor(output[0]) else None)
            record = {
                "seq": self.seq,
                "call": len(self.calls[key]),
                "qlen": int(inp.shape[-2]) if inp is not None and inp.dim() >= 2 else None,
                "prefill": bool(params.get("prefill", False)) or (
                    inp is not None and inp.dim() >= 2 and inp.shape[-2] > PREFILL_QLEN),
                "in": in_fp,
                "out": fingerprint(self.torch, out),
            }
            self.calls[key].append(record)
            self.seq += 1
            return output
        return wrapped

    def close(self):
        for module, original in self.patched:
            module.forward = original
        self.patched = []


def compare_runs(calls_a, calls_b):
    """First output divergence per module; a source has a matching input there.

    Fail-closed: module sets must match exactly, and any call-count mismatch is
    reported instead of silently truncated.
    """
    only_a = sorted(set(calls_a) - set(calls_b))
    only_b = sorted(set(calls_b) - set(calls_a))
    if only_a or only_b:
        raise ValueError(f"module sets differ: only_a={only_a} only_b={only_b}")
    sources = []
    propagated = []
    length_mismatches = []
    for key in sorted(calls_a):
        a, b = calls_a[key], calls_b[key]
        if len(a) != len(b):
            length_mismatches.append({"module": key, "calls_a": len(a), "calls_b": len(b)})
        for call, (ca, cb) in enumerate(zip(a, b)):
            if ca["out"] != cb["out"]:
                entry = {"module": key, "call": call, "seq": ca["seq"],
                         "prefill": ca["prefill"], "qlen": ca["qlen"]}
                if ca["in"] == cb["in"]:
                    sources.append(entry)
                else:
                    propagated.append(entry)
                break
    sources.sort(key=lambda entry: entry["seq"])
    propagated.sort(key=lambda entry: entry["seq"])
    return {
        "modules": len(calls_a),
        "sources": sources,
        "propagated": propagated,
        "length_mismatches": length_mismatches,
        "stable": not sources and not propagated and not length_mismatches,
    }


def run_pass(torch, b, ids, tokens):
    """One recorded greedy pass over a fresh cache."""
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import GreedySampler
    from .fidelity_probe import fresh_generator
    fresh_generator(b.generator)
    recorder = Recorder(torch)
    recorder.attach(b.generator.model, "t")
    if getattr(b.generator, "draft_model", None) is not None:
        recorder.attach(b.generator.draft_model, "d")
    try:
        b.generator.enqueue(Job(input_ids=ids, max_new_tokens=tokens,
                                sampler=GreedySampler(), seed=20260923, stop_conditions=[]))
        while b.remaining():
            b.iterate()
    finally:
        recorder.close()
    return recorder.calls


def bisect_context(torch, b, context, tokens):
    gen = torch.Generator().manual_seed(20260923 + context)
    ids = torch.randint(10000, 50000, (1, context), generator=gen)
    first = run_pass(torch, b, ids, tokens)
    second = run_pass(torch, b, ids, tokens)
    verdict = compare_runs(first, second)
    verdict["context"] = context
    verdict["decode_tokens"] = tokens
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="4096,32768")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--timed-cell", default=None,
                        help="label of a prompts-phase2 cell to re-run unrecorded, "
                             "for a clean tok/s comparison with the graph-band arms")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    if any(c % 256 for c in contexts):
        raise ValueError("contexts must be multiples of 256")
    args.output.mkdir(parents=True, exist_ok=False)
    torch = setup()
    b = backend(262144)
    try:
        warmup(b, torch)
        for context in sorted(contexts):
            verdict = bisect_context(torch, b, context, args.tokens)
            write(args.output / f"bisect-{context}.json", verdict)
            first = verdict["sources"][:1] or verdict["propagated"][:1] or [None]
            print(f"[{context}] stable={verdict['stable']} "
                  f"sources={len(verdict['sources'])} first={first[0]}")
        if args.timed_cell:
            from .decode_probe import run_sample
            from .fidelity_probe import fresh_generator
            cells = json.loads((ROOT / "results/20260910-quality-gate/prompts-phase2.json").read_text())
            cell = next(c for c in cells if c["label"] == args.timed_cell)
            fresh_generator(b.generator)
            options = argparse.Namespace(max_new_tokens=cell["max_tokens"],
                                         thinking="medium", sampler="greedy")
            stats = run_sample(b.generator, b.tokenizer, cell["prompt_ids"], options,
                               cell["seed"], args.output / "timed-events.jsonl")
            verifies = len(stats.get("draft_stats") or [])
            decode_ms = stats["elapsed_ms"] - stats["ttft_ms"]
            write(args.output / f"timed-{args.timed_cell}.json", {
                "label": args.timed_cell,
                "decode_tokens_per_second": stats["decode_tokens_per_second"],
                "ms_per_verify": decode_ms / verifies if verifies else None,
                "verifies": verifies,
                "generated_tokens": stats["generated_tokens"],
                "completion_sha256": hashlib.sha256(stats["completion"].encode()).hexdigest(),
            })
            print(f"[timed] {args.timed_cell}: {stats['decode_tokens_per_second']:.2f} tok/s")
    finally:
        del b
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
