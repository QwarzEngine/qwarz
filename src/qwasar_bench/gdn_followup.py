"""Evaluate the 2026-09-23 GDN-determinism follow-up: acceptance A/B + bit-exact graph recert.

With the GDN patch the stack is bit-reproducible, so the draft-graph promotion gate
returns to its strict form: G and E must produce IDENTICAL draft ids at every verify
(zero tolerance), and the shadow-vs-venv A/B settles whether the patch's rounding
shift moves acceptance/cycle beyond trajectory luck. Fail-closed throughout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

from .graph_band import first_divergence

INF = float("inf")


def load_cells(run_dir: Path):
    """{(cell, arm): [run dicts in rep order]} from one graph_runner output dir."""
    run_dir = Path(run_dir)
    samples = sorted(run_dir.glob("*-sample.json"))
    if not samples:
        raise ValueError(f"no samples in {run_dir}")
    out = {}
    for path in samples:
        if path.name.startswith("jit-"):
            continue
        sample = json.loads(path.read_text())
        parts = path.name.removesuffix("-sample.json").rsplit("-", 2)
        if len(parts) != 3 or not parts[1].startswith("r"):
            raise ValueError(f"unparseable run label: {path.name}")
        cell, rep_text, arm = parts
        ids_path = run_dir / f"{cell}-{rep_text}-{arm}-draft-ids.json"
        if not ids_path.exists():
            raise ValueError(f"missing draft ids log: {ids_path}")
        out.setdefault((cell, arm), []).append({
            "rep": int(rep_text[1:]), "ids": json.loads(ids_path.read_text()),
            "sample": sample})
    for key in out:
        out[key].sort(key=lambda r: r["rep"])
        reps = [r["rep"] for r in out[key]]
        # The runner's rep counter is global across cells; only uniqueness matters.
        # Pairing uses the sorted position, so reps need not start at 0.
        if len(set(reps)) != len(reps):
            raise ValueError(f"duplicate reps for {key}: {reps}")
    return out


def bitexact_gate(cells_map, cell, need_pairs=2):
    """Zero-tolerance G==E gate: identical draft ids at every verify of every pair,
    identical completions, and zero inline validation mismatches in G samples."""
    e_runs = cells_map.get((cell, "E"), [])
    g_runs = cells_map.get((cell, "G"), [])
    if len(e_runs) < need_pairs or len(g_runs) < need_pairs:
        raise ValueError(f"{cell}: need {need_pairs} E/G pairs, got {len(e_runs)}/{len(g_runs)}")
    pairs = []
    for i in range(need_pairs):
        div = first_divergence(e_runs[i]["ids"], g_runs[i]["ids"])
        same_completion = (e_runs[i]["sample"].get("completion_sha256")
                           == g_runs[i]["sample"].get("completion_sha256"))
        mismatches = g_runs[i]["sample"]["draft_graph"]["validation"]["mismatches"]
        pairs.append({"rep": i, "first_divergence": None if div == INF else div,
                      "same_completion": same_completion, "validation_mismatches": mismatches})
    gate = all(p["first_divergence"] is None and p["same_completion"]
               and p["validation_mismatches"] == 0 for p in pairs)
    return {"cell": cell, "pairs": pairs, "gate_pass": gate}


def cycle_gate(cells_map, cell, lo_ms, hi_ms, need_pairs=2):
    """Median (E-G) ms/verify across rep-matched pairs must sit in [lo_ms, hi_ms]."""
    e_runs = cells_map.get((cell, "E"), [])
    g_runs = cells_map.get((cell, "G"), [])
    if len(e_runs) < need_pairs or len(g_runs) < need_pairs:
        raise ValueError(f"{cell}: need {need_pairs} E/G pairs")
    deltas = [e_runs[i]["sample"]["ms_per_verify"] - g_runs[i]["sample"]["ms_per_verify"]
              for i in range(need_pairs)]
    med = statistics.median(deltas)
    return {"cell": cell, "pair_deltas_ms": deltas, "median_ms": med,
            "gate_pass": lo_ms <= med <= hi_ms}


def ab_gate(shadow_map, venv_map, cell, acc_margin, cycle_margin):
    """Shadow (deterministic, 1 rep) vs venv (noisy, n reps) on acceptance and cycle time."""
    shadow = shadow_map.get((cell, "P"), [])
    venv = venv_map.get((cell, "P"), [])
    if len(shadow) != 1 or len(venv) < 2:
        raise ValueError(f"{cell}: need 1 shadow P run and >=2 venv P runs")
    s_acc = shadow[0]["sample"]["draft_acceptance"]
    v_accs = [r["sample"]["draft_acceptance"] for r in venv]
    s_cycle = shadow[0]["sample"]["ms_per_verify"]
    v_cycle = statistics.median(r["sample"]["ms_per_verify"] for r in venv)
    return {
        "cell": cell,
        "shadow_acceptance": s_acc, "venv_acceptance_runs": v_accs,
        "shadow_ms_per_verify": s_cycle, "venv_ms_per_verify_median": v_cycle,
        "acceptance_pass": min(v_accs) - acc_margin <= s_acc <= max(v_accs) + acc_margin,
        "cycle_pass": s_cycle <= v_cycle * (1 + cycle_margin),
    }


def evaluate(root: Path):
    root = Path(root)
    prediction = json.loads((root / "prediction.json").read_text())
    if not prediction.get("frozen_before_measurement"):
        raise ValueError("prediction was not frozen")
    gates = prediction["gates"]
    out = {"prediction_sha256": hashlib.sha256(
        (root / "prediction.json").read_bytes()).hexdigest()}

    bx = load_cells(root / "bx-shadow")
    out["bitexact"] = [bitexact_gate(bx, cell) for cell in gates["bitexact_cells"]]
    out["cycle"] = [cycle_gate(bx, cell, gates["cycle_ms"]["lo"], gates["cycle_ms"]["hi"])
                    for cell in gates["cycle_cells"]]

    shadow = load_cells(root / "ab-shadow")
    venv = load_cells(root / "ab-venv")
    out["ab"] = [ab_gate(shadow, venv, cell,
                         gates["ab"]["acceptance_margin"], gates["ab"]["cycle_margin"])
                 for cell in gates["ab"]["cells"]]

    out["all_pass"] = (
        all(g["gate_pass"] for g in out["bitexact"])
        and all(g["gate_pass"] for g in out["cycle"])
        and all(g["acceptance_pass"] and g["cycle_pass"] for g in out["ab"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    verdict = evaluate(args.root)
    (args.root / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "prediction_sha256"},
                     indent=2, default=str)[:3000])
    print("ALL PASS" if verdict["all_pass"] else "GATES FAILED")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
