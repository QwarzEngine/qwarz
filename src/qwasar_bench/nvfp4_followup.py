"""Alternated held-out pilot, with one fresh process per arm and seed."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .nvfp4_study import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072, 258048])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = [("control", 193), ("attention", 193), ("attention", 827), ("control", 827)]
    for index, (profile, seed) in enumerate(sequence):
        target = args.output / f"{index:02}-{profile}-s{seed}"
        command = [sys.executable, "-m", "qwasar_bench.nvfp4_quality", "run",
                   "--output", str(target), "--profile", profile, "--seeds", str(seed),
                   "--contexts", *map(str, args.contexts)]
        with (args.output / f"{target.name}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        write(args.output / f"process-{index:02}.json",
              {"command": command, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"held-out pilot failed: {target}")
    write(args.output / "completed.json", {"completed": True, "sequence": sequence})


if __name__ == "__main__":
    main()
