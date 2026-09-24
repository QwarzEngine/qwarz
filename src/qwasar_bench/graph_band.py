"""Offline analysis for the 2026-09-23 draft-graph statistical-equivalence campaign.

Fail-closed: missing runs/arms or a missing frozen gate raise instead of
producing a partial verdict. Pure CPU; reads only the campaign output dirs.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import statistics

INF = float("inf")


def load_run(run_dir: Path):
    """Return {rep: {"arm": str, "ids": [...], "sample": dict}} for one runner output."""
    samples = sorted(run_dir.glob("*-r*-?-sample.json"))
    if not samples:
        raise ValueError(f"no samples in {run_dir}")
    runs = {}
    for path in samples:
        sample = json.loads(path.read_text())
        # The run label lives in the FILENAME; sample["label"] is overwritten
        # by the cell's own label upstream in graph_runner.
        label = path.name.removesuffix("-sample.json")
        stem = label.rsplit("-r", 1)[1]
        rep_text, arm = stem.rsplit("-", 1)
        rep = int(rep_text)
        ids_path = run_dir / f"{label}-draft-ids.json"
        if not ids_path.exists():
            raise ValueError(f"missing draft ids log: {ids_path}")
        ids = json.loads(ids_path.read_text())
        if rep in runs:
            raise ValueError(f"duplicate rep {rep} in {run_dir}")
        runs[rep] = {"arm": arm, "ids": ids, "sample": sample}
    return runs


def first_divergence(a, b):
    """First verify index whose six draft ids differ. Runs legitimately end at
    different verifies once trajectories diverge (EOS is target-driven), so
    lengths need not match: an identical prefix with different lengths counts
    as divergence at the shorter run's end. inf only if fully identical."""
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra != rb:
            return i
    return INF if len(a) == len(b) else min(len(a), len(b))


def band(runs, need_e=6, need_g=6):
    """E-E and G-E first-divergence distributions over all run pairs."""
    e_reps = sorted(r for r, v in runs.items() if v["arm"] == "E")
    g_reps = sorted(r for r, v in runs.items() if v["arm"] == "G")
    if len(e_reps) != need_e or len(g_reps) != need_g:
        raise ValueError(f"expected {need_e} E and {need_g} G runs, got {len(e_reps)}/{len(g_reps)}")

    def pairs(left, right):
        out = []
        for ra, rb in combinations(left, 2) if right is None else ((x, y) for x in left for y in right):
            div = first_divergence(runs[ra]["ids"], runs[rb]["ids"])
            same_completion = (runs[ra]["sample"].get("completion_sha256")
                               == runs[rb]["sample"].get("completion_sha256"))
            out.append({"reps": [ra, rb], "first_divergence": None if div == INF else div,
                        "diverged": div != INF, "same_completion": same_completion})
        return out

    ee, ge = pairs(e_reps, None), pairs(e_reps, g_reps)

    def stats(rows):
        divs = [r["first_divergence"] if r["diverged"] else INF for r in rows]
        return {"n": len(rows), "diverged": sum(r["diverged"] for r in rows),
                "identical_fraction": 1 - sum(r["diverged"] for r in rows) / len(rows),
                "same_completion_fraction": sum(r["same_completion"] for r in rows) / len(rows),
                "first_divergence_median": None if statistics.median(divs) == INF else statistics.median(divs),
                "first_divergence_min": None if min(divs) == INF else min(divs)}

    ee_s, ge_s = stats(ee), stats(ge)
    # Frozen gate BAND-*: G-E median no earlier than E-E median, and G-E
    # identical fraction within 0.15 of E-E. Empty band (all E-E identical)
    # requires all G-E identical.
    med = lambda s: INF if s["first_divergence_median"] is None else s["first_divergence_median"]
    if ee_s["identical_fraction"] == 1.0:
        passed = ge_s["identical_fraction"] == 1.0
    else:
        passed = med(ge_s) >= med(ee_s) and ge_s["identical_fraction"] >= ee_s["identical_fraction"] - 0.15
    return {"ee": ee_s, "ge": ge_s, "ee_pairs": ee, "ge_pairs": ge, "gate_pass": passed}


def gain(runs, per_arm=4):
    """Cycle effect (E-G ms/verify) and net effect (P-G tok/s), rep-matched."""
    by_arm = {}
    for arm in ("P", "G", "E"):
        rows = sorted((r for r, v in runs.items() if v["arm"] == arm), key=lambda r: r)
        if len(rows) != per_arm:
            raise ValueError(f"expected {per_arm} {arm} runs, got {len(rows)}")
        by_arm[arm] = [runs[r] for r in rows]
    med = lambda arm, key: statistics.median(v["sample"][key] for v in by_arm[arm])
    cycle = med("E", "ms_per_verify") - med("G", "ms_per_verify")
    net_pairs = [g["sample"]["decode_tokens_per_second"] / p["sample"]["decode_tokens_per_second"] - 1
                 for p, g in zip(by_arm["P"], by_arm["G"])]
    mem = med("G", "peak_allocated") - med("P", "peak_allocated")
    return {"cycle_ms": cycle,
            "ms_per_verify": {arm: med(arm, "ms_per_verify") for arm in ("P", "G", "E")},
            "net_tok_s_pairs": net_pairs, "net_tok_s_median": statistics.median(net_pairs),
            "acceptance": {arm: med(arm, "draft_acceptance") for arm in ("P", "G", "E")},
            "mem_delta_gib": mem / 2**30}


def evaluate(root: Path):
    root = Path(root)
    prediction = json.loads((root / "prediction.json").read_text())
    if not prediction.get("frozen_before_measurement"):
        raise ValueError("prediction was not frozen")
    out = {"prediction_sha256": __import__("hashlib").sha256(
        (root / "prediction.json").read_bytes()).hexdigest()}
    gates = {}
    for tag, cell in (("32K", "32"), ("256K", "256")):
        band_dir, ab_dir = root / f"band{cell}", root / f"ab{cell}"
        b = band(load_run(band_dir))
        g = gain(load_run(ab_dir))
        gates[f"BAND-{tag}"] = b["gate_pass"]
        gates[f"CYCLE-{tag}"] = ((0.25 if tag == "32K" else 0.40) <= g["cycle_ms"] <= 1.50)
        gates[f"NET-{tag}"] = g["net_tok_s_median"] > 0
        out[tag] = {"band": {k: b[k] for k in ("ee", "ge", "gate_pass")}, "gain": g}
    mem = max(abs(out[t]["gain"]["mem_delta_gib"]) for t in ("32K", "256K"))
    gates["MEM"] = mem <= 0.10
    out["gates"] = gates
    out["all_gates_pass"] = all(gates.values())
    out["automatic_promotion"] = False
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate(a.input)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"gates": result["gates"], "all_gates_pass": result["all_gates_pass"]}, indent=2))


if __name__ == "__main__":
    main()
