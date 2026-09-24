"""Bounded MTP pilot orchestration; invoke only under managed.py."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .mtp_adapter import ResidualAdapter, file_hash, install
from .nvfp4_study import write


def quality(out, adapter_path, seed):
    from qwasar_runtime import engine
    from .nvfp4_quality import run

    original = engine.ExLlamaBackend
    lifetime = ExitStack()

    def load(*args, **kwargs):
        b = original(*args, **kwargs)
        if adapter_path:
            adapter = ResidualAdapter.load(adapter_path)
            lifetime.enter_context(install(b.generator, adapter))
            b.config["experimental_mtp_adapter"] = {
                "sha256": file_hash(adapter_path), "rank": adapter.a.shape[1]}
        return b

    engine.ExLlamaBackend = load
    try:
        write(out / "arm.json", {"arm": "candidate" if adapter_path else "control",
              "adapter": str(adapter_path) if adapter_path else None, "seed": seed,
              "projection_profile": "control"})
        run(out, "control", [seed], [32768, 131072, 258048])
    finally:
        lifetime.close()
        engine.ExLlamaBackend = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("campaign", "resume", "quality"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--seed", type=int, default=193)
    parser.add_argument("--budget", type=int, default=4800)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    if args.mode == "resume" and args.source is None:
        parser.error("resume requires --source with the original selected checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "source.json", {p.name: file_hash(p) for p in
          Path(__file__).parent.glob("mtp_*.py")})
    write(args.output / "execution-environment.json", {
        key: os.environ.get(key) for key in
        ("CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")})
    if args.mode == "quality":
        quality(args.output, args.adapter, args.seed)
        return
    started = time.monotonic()
    processes = []

    def execute(label, module, arguments):
        remaining = args.budget - (time.monotonic() - started)
        if remaining < 180:
            raise TimeoutError("pilot budget exhausted; restore service")
        command = [sys.executable, "-m", module, *map(str, arguments)]
        with (args.output / f"{label}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
        entry = {"label": label, "command": command, "returncode": result.returncode,
                 "elapsed_since_start": time.monotonic() - started}
        processes.append(entry)
        write(args.output / f"{label}-process.json", entry)
        if result.returncode:
            raise RuntimeError(f"stage failed: {label}")

    data, training, online = (args.output / name for name in ("data", "training", "online"))
    if args.mode == "campaign":
        execute("capture", "qwasar_bench.mtp_study",
                ["capture", "--output", data, "--limit", args.limit, "--tokens", 1024])
        execute("train", "qwasar_bench.mtp_study",
                ["train", "--output", training, "--source", data, "--steps", 800])
    else:
        training = args.source / "training"
        write(args.output / "origin.json", {"source": str(args.source.resolve()),
              "training_hashes": {name: file_hash(training / name) for name in
                                  ("selection.json", "heldout.json", "baseline.json")},
              "reason": "Resume evaluation after OOM, retaining the original validation-selected checkpoint."})
    selection = json.loads((training / "selection.json").read_text())
    adapter = Path(selection["path"])
    if file_hash(adapter) != selection["sha256"]:
        raise ValueError("selected adapter hash mismatch")
    execute("online", "qwasar_bench.mtp_study",
            ["online", "--output", online, "--adapter", adapter, "--tokens", 1024,
             "--contexts", 4096, 32768, "--repeats", 4])
    for i, (arm, seed) in enumerate((("control", 193), ("candidate", 193),
                                     ("candidate", 827), ("control", 827))):
        label = f"quality-{i:02}-{arm}-s{seed}"
        arguments = ["quality", "--output", args.output / label, "--seed", seed]
        if arm == "candidate":
            arguments += ["--adapter", adapter]
        execute(label, "qwasar_bench.mtp_campaign", arguments)
    write(args.output / "completed.json", {"completed": True, "processes": processes,
          "elapsed_seconds": time.monotonic() - started, "automatic_promotion": False})


if __name__ == "__main__":
    main()
