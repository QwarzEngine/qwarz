"""Record the ExLlama oracle q38 will have to match.

One greedy token at 64, 32,768 and 262,144 positions. Layer taps are the
first Gated DeltaNet, the first attention layer, the first MLP, the
embedding and the vocabulary head. Full tensors are kept only for the
short prompt. Longer calls store a SHA-256 of the raw tensor bytes.
This process does not serve traffic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from taps import select_taps

SEED = 20260928
ID_LOW = 1000
ID_HIGH = 20000


def prompt_ids(torch, length):
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(ID_LOW, ID_HIGH, (1, length), generator=generator)


def tensor_record(torch, tensor, save_path=None):
    if tensor is None or not torch.is_tensor(tensor):
        return None
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    record = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype).removeprefix("torch."),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if save_path is not None:
        torch.save(cpu, save_path)
        record["file"] = save_path.name
    return record


class Tap:
    def __init__(self, torch, module, name, output, save_full):
        self.torch = torch
        self.name = name
        self.output = output
        self.save_full = save_full
        self.records = []
        self.original = module.forward
        module.forward = self._wrapped
        self.module = module

    def close(self):
        self.module.forward = self.original

    def _wrapped(self, x, *args, **kwargs):
        params = next((item for item in args if isinstance(item, dict)), kwargs.get("params") or {})
        qlen = int(x.shape[-2]) if self.torch.is_tensor(x) and x.dim() >= 2 else None
        output = self.original(x, *args, **kwargs)
        if self.torch.cuda.is_current_stream_capturing():
            return output
        out = output[0] if isinstance(output, tuple) and output else output
        save = None
        if self.save_full and qlen is not None and qlen <= 64 and not any(item.get("file") for item in self.records):
            save = self.output / f"{self.name}-q{qlen}.pt"
        self.records.append({
            "qlen": qlen,
            "prefill": bool(params.get("prefill", False)) or (qlen or 0) > 1,
            "in": tensor_record(self.torch, x if self.torch.is_tensor(x) else None),
            "out": tensor_record(self.torch, out if self.torch.is_tensor(out) else None, save),
        })
        return output


def module_list(model):
    # Model.__iter__ yields the inner modules. model.modules is only the blocks.
    return [(getattr(module, "key", type(module).__name__), type(module).__name__) for module in model]


def run(model, donor, output, contexts):
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import GreedySampler
    from qwasar_runtime.engine import ExLlamaBackend
    from qwasar_bench.fidelity_probe import fresh_generator

    output.mkdir(parents=True, exist_ok=True)
    print("loading production ExLlama stack", flush=True)
    backend = ExLlamaBackend(model, 262144, prefill="xqa")
    modules = module_list(backend.generator.model)
    try:
        chosen = select_taps(modules)
    except ValueError:
        from collections import Counter
        print(Counter(kind for _, kind in modules), flush=True)
        raise
    by_key = {getattr(module, "key", type(module).__name__): module for module in backend.generator.model}
    report = {
        "seed": SEED,
        "id_range": [ID_LOW, ID_HIGH],
        "model": str(model),
        "donor": str(donor),
        "taps": chosen,
        "contexts": [],
    }
    try:
        for length in contexts:
            print(f"context {length}", flush=True)
            fresh_generator(backend.generator)
            ids = prompt_ids(torch, length)
            taps = [Tap(torch, by_key[key], name, output, save_full=(length <= 64)) for name, key in chosen.items()]
            token = None
            try:
                with backend.tuning():
                    backend.generator.enqueue(Job(
                        input_ids=ids, max_new_tokens=1, sampler=GreedySampler(), seed=SEED, stop_conditions=[],
                    ))
                    while backend.remaining():
                        for event in backend.iterate():
                            if event.get("token_ids"):
                                token = event["token_ids"][-1]
            finally:
                for tap in taps:
                    tap.close()
            if token is None:
                raise RuntimeError(f"context {length} produced no token")
            report["contexts"].append({
                "length": length,
                "first_ids": ids[0, :8].tolist(),
                "last_ids": ids[0, -8:].tolist(),
                "greedy_token": token,
                "taps": {tap.name: tap.records for tap in taps},
            })
            (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"context {length} token {token}", flush=True)
    finally:
        del backend
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw")
    parser.add_argument("--donor", type=Path, default=Path.home() / "models/nvidia-qwen38-27b-nvfp4/dbb8f445b3145f8a4c18ddc769f032d57d32867c")
    parser.add_argument("--output", type=Path, default=ROOT / "results/q38-oracle")
    parser.add_argument("--contexts", default="64,32768,261888")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run to load ExLlama and occupy the 5090")
    contexts = [int(item) for item in args.contexts.split(",") if item]
    report = run(args.model, args.donor, args.output, contexts)
    summary = {
        "seed": report["seed"],
        "id_range": report["id_range"],
        "taps": report["taps"],
        "contexts": [
            {"length": item["length"], "greedy_token": item["greedy_token"],
             "first_ids": item["first_ids"], "last_ids": item["last_ids"]}
            for item in report["contexts"]
        ],
    }
    (ROOT / "engine/oracle/capture.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
