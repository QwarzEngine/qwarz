"""Fail-closed aggregation of the NVFP4 screen and held-out pilot."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import statistics

from .nvfp4_study import write


def retrieval_semantic(sample):
    from .nvfp4_quality import json_value
    if sample["status"] != "completed":
        return False
    try:
        actual = json_value(sample["message"]["content"])
    except (KeyError, TypeError, ValueError):
        return False
    expected = sample["expected"]
    expanded = {f"AUDIT_MARKER_{i}": value for i, value in enumerate(expected["values"])}
    expanded["sum"] = expected["sum"]
    return actual == expected or actual == expanded


def paired_counts(control, candidate):
    if not control or control.keys() != candidate.keys():
        raise ValueError("missing or unpaired quality results")
    counts = collections.Counter()
    by_kind = {}
    clusters = collections.defaultdict(list)
    for label, a in control.items():
        b = candidate[label]
        if type(a.get("passed")) is not bool or type(b.get("passed")) is not bool:
            raise ValueError("missing quality verdict")
        if a["kind"] != b["kind"]:
            raise ValueError("different task kinds")
        state = ("both_pass" if a["passed"] and b["passed"] else
                 "regression" if a["passed"] else "improvement" if b["passed"] else "both_fail")
        counts[state] += 1
        kind = a["kind"]
        group = by_kind.setdefault(kind, {"n": 0, "control_pass": 0, "candidate_pass": 0,
                                         "regressions": [], "improvements": []})
        group["n"] += 1
        group["control_pass"] += a["passed"]
        group["candidate_pass"] += b["passed"]
        if state in ("regression", "improvement"):
            group[state + "s"].append(label)
        cluster = label.split("-s")[0] if kind == "code" else (
            "retrieval" if kind == "append" else kind)
        clusters[cluster].append(int(b["passed"]) - int(a["passed"]))
    # Resample task families, not repeated seeds as independent tasks.
    rng = random.Random(923)
    values = list(clusters.values())
    boot = []
    for _ in range(10000):
        draw = [item for _ in values for item in rng.choice(values)]
        boot.append(statistics.fmean(draw) * 100)
    boot.sort()
    interval = [boot[249], boot[9749]]
    if interval[0] == interval[1]:
        interval = None  # A degenerate bootstrap is not evidence of zero uncertainty.
    return {"pairs": len(control), "counts": dict(counts), "by_kind": by_kind,
            "delta_success_pp": 100 * (counts["improvement"] - counts["regression"]) / len(control),
            "descriptive_cluster_bootstrap_95_pp": interval,
            "task_families": len(clusters),
            "warning": "Small pilot with correlated tasks; this interval is not a quality certificate."}


def screen_summary(root):
    manifest = json.loads((root / "completed.json").read_text())
    if manifest.get("completed") is not True:
        raise ValueError("screen incomplete")
    runs = {}
    for process in manifest["processes"]:
        if process["returncode"]:
            raise ValueError("screen process failed")
        target = Path(process["output"])
        data = json.loads((target / "summary.json").read_text())
        if data.get("completed") is not True:
            raise ValueError("screen arm incomplete")
        runs[target.name] = data
    summaries = {}
    for name, data in runs.items():
        by_context = {}
        for context in sorted({s["context"] for s in data["samples"]}):
            rows = [s for s in data["samples"] if s["context"] == context]
            by_context[context] = {
                "n": len(rows), **{
                    key: statistics.median(s[key] for s in rows)
                    for key in ("ttft_ms", "decode_tokens_per_second", "draft_acceptance",
                                "peak_allocated", "peak_reserved")},
                "truncated": sum(s["truncated"] for s in rows)}
        summaries[name] = by_context
    return summaries


def quality_summary(root, grades):
    manifest = json.loads((root / "completed.json").read_text())
    if manifest.get("completed") is not True:
        raise ValueError("quality pilot incomplete")
    arms = {"control": {}, "attention": {}}
    semantic = {"control": {}, "attention": {}}
    performance = collections.defaultdict(lambda: collections.defaultdict(list))
    prompts = {}
    for index, (profile, seed) in enumerate(manifest["sequence"]):
        name = f"{index:02}-{profile}-s{seed}"
        run = root / name
        summary = json.loads((run / "summary.json").read_text())
        if summary.get("completed") is not True:
            raise ValueError("quality arm incomplete")
        rows = json.loads((grades / name / "summary.json").read_text())
        expected_labels = {s["label"] for s in summary["samples"]}
        if {r["label"] for r in rows} != expected_labels:
            raise ValueError("missing grades")
        for row in rows:
            label = row["label"]
            if label in arms[profile]:
                raise ValueError("duplicate quality label")
            sample_bytes = (run / f"{label}-sample.json").read_bytes()
            if hashlib.sha256(sample_bytes).hexdigest() != row["source_sha256"]:
                raise ValueError("stale grade: sample hash mismatch")
            sample = json.loads(sample_bytes)
            if row["kind"] != sample["kind"]:
                raise ValueError("grade kind mismatch")
            arms[profile][label] = row
            semantic[profile][label] = dict(row)
            if sample["kind"] in ("retrieval", "append"):
                semantic[profile][label]["passed"] = retrieval_semantic(sample)
            if sample.get("request_sha256"):
                other = prompts.setdefault(label, sample["request_sha256"])
                if other != sample["request_sha256"]:
                    raise ValueError("prompt mismatch between arms")
            performance[profile][label.split("-s")[0]].append({
                "label": label, "passed": row["passed"], "status": sample["status"],
                "metrics": sample["metrics"], "usage": sample["usage"],
                "peak_allocated": sample["peak_allocated"]})
    return {"quality_strict": paired_counts(arms["control"], arms["attention"]),
            "quality_semantic_retrieval": paired_counts(semantic["control"], semantic["attention"]),
            "performance": {k: dict(v) for k, v in performance.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--grades", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.quality) != bool(args.grades):
        parser.error("--quality and --grades must be supplied together")
    report = {"screen": screen_summary(args.screen), "automatic_promotion": False}
    if args.quality:
        report.update(quality_summary(args.quality, args.grades))
    write(args.output, report)


if __name__ == "__main__":
    main()
