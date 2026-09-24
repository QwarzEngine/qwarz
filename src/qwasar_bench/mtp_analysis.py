"""Summarize adapter training, closed-loop timing and graded task outcomes."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics

from .mtp_adapter import file_hash
from .nvfp4_analysis import paired_counts, retrieval_semantic
from .nvfp4_study import write


def online_summary(rows):
    result = {}
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        groups[row["context"]][row["arm"]].append(row)
    if not groups:
        raise ValueError("empty online comparison")
    for context, arms in groups.items():
        if set(arms) != {"control", "candidate"}:
            raise ValueError("missing comparison arm")
        pairs = {}
        for rep in {r["rep"] for r in arms["control"] + arms["candidate"]}:
            a = [r for r in arms["control"] if r["rep"] == rep]
            b = [r for r in arms["candidate"] if r["rep"] == rep]
            if len(a) != 2 or len(b) != 2 or len({r["prompt_sha256"] for r in a + b}) != 1:
                raise ValueError("unpaired online samples")
            pairs[rep] = {
                key: {"control": statistics.median(r[key] for r in a),
                      "candidate": statistics.median(r[key] for r in b)}
                for key in ("decode_tokens_per_second", "draft_acceptance",
                            "decode_ms_per_verify_estimate", "ttft_ms", "elapsed_ms")}
        summary = {}
        for key in ("decode_tokens_per_second", "draft_acceptance",
                    "decode_ms_per_verify_estimate", "ttft_ms", "elapsed_ms"):
            a = statistics.median(r[key] for r in arms["control"])
            b = statistics.median(r[key] for r in arms["candidate"])
            summary[key] = {"control": a, "candidate": b, "relative_change_pct": (b / a - 1) * 100}
        result[context] = {"control_n": len(arms["control"]), "candidate_n": len(arms["candidate"]),
                           "medians": summary, "per_task": pairs}
    return result


def quality_summary(root, grades, adapter_sha256):
    arms = {"control": {}, "candidate": {}}
    semantic = {"control": {}, "candidate": {}}
    hashes = {}
    configuration = None
    runs = [run for run in sorted(root.glob("quality-*-s*")) if run.is_dir()]
    if len(runs) != 4:
        raise ValueError("expected four quality runs")
    for run in runs:
        arm = json.loads((run / "arm.json").read_text())["arm"]
        config = json.loads((run / "configuration.json").read_text())["config"]
        adapter = config.pop("experimental_mtp_adapter", None)
        if arm == "candidate":
            if not adapter or adapter.get("sha256") != adapter_sha256:
                raise ValueError("quality checkpoint differs from selection")
        elif arm != "control" or adapter is not None:
            raise ValueError("invalid quality control")
        if configuration is None:
            configuration = config
        elif config != configuration:
            raise ValueError("different target configurations")
        summary = json.loads((run / "summary.json").read_text())
        if not summary["completed"]:
            raise ValueError("incomplete quality run")
        rows = json.loads((grades / run.name / "summary.json").read_text())
        if {r["label"] for r in rows} != {s["label"] for s in summary["samples"]}:
            raise ValueError("missing grades")
        for row in rows:
            label = row["label"]
            path = run / f"{label}-sample.json"
            if file_hash(path) != row["source_sha256"] or label in arms[arm]:
                raise ValueError("stale or duplicate grade")
            sample = json.loads(path.read_text())
            if sample.get("request_sha256"):
                if hashes.setdefault(label, sample["request_sha256"]) != sample["request_sha256"]:
                    raise ValueError("different quality prompts")
            arms[arm][label] = row
            semantic[arm][label] = dict(row)
            if sample["kind"] in ("retrieval", "append"):
                semantic[arm][label]["passed"] = retrieval_semantic(sample)
    return {"strict": paired_counts(arms["control"], arms["candidate"]),
            "semantic_retrieval": paired_counts(semantic["control"], semantic["candidate"])}


def training_source(root):
    origin = root / "origin.json"
    if not origin.exists():
        return root / "training"
    provenance = json.loads(origin.read_text())
    training = Path(provenance["source"]) / "training"
    for name in ("selection.json", "heldout.json", "baseline.json"):
        if file_hash(training / name) != provenance["training_hashes"][name]:
            raise ValueError("original training evidence changed")
    return training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    completion = json.loads((root / "completed.json").read_text())
    if not completion["completed"] or any(p["returncode"] for p in completion["processes"]):
        raise ValueError("campaign incomplete")
    online = json.loads((root / "online/summary.json").read_text())
    if not online["completed"]:
        raise ValueError("online comparison incomplete")
    rows = online["samples"]
    training = training_source(root)
    selection = json.loads((training / "selection.json").read_text())
    if file_hash(selection["path"]) != selection["sha256"]:
        raise ValueError("selected checkpoint changed")
    config = json.loads((root / "online/configuration.json").read_text())
    if config["adapter_sha256"] != selection["sha256"]:
        raise ValueError("online checkpoint differs from selection")
    result = {"online": online_summary(rows),
              "quality": quality_summary(root, args.grades, selection["sha256"]),
              "selection": selection,
              "offline_heldout": json.loads((training / "heldout.json").read_text()),
              "evidence_sha256": {
                  str(path): file_hash(path) for path in
                  [root / "completed.json", root / "online/summary.json",
                   *root.glob("quality-*-s*/*-sample.json"),
                   *args.grades.glob("*/summary.json")]},
              "automatic_promotion": False,
              "caution": "Offline argmax agreement is not online acceptance; task pilot is small."}
    write(args.output, result)


if __name__ == "__main__":
    main()
