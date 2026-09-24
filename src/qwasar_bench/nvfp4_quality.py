"""Held-out pilot tasks for the NVFP4 study, not a general quality certificate."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import time

from . import nvfp4_projections as projections
from .nvfp4_study import MODEL, ROOT, setup, write

CODE_TASKS = {
    "interval": ("IntervalSet", "Implement IntervalSet with add(lo, hi), remove(lo, hi), "
                 "contains(x), intervals(). It stores half-open integer intervals, merges touching "
                 "or overlapping intervals, and splits them on removal. Reject non-integer "
                 "endpoints (including bool) and lo >= hi without mutation. intervals() returns "
                 "a fresh sorted list of (lo, hi) tuples."),
    "bimap": ("BiMap", "Implement BiMap with put(key, value), get(key), inverse(value), "
              "delete(key), and len(). Keys and values are hashable. Values are unique. Updating "
              "a key removes the old inverse. A value owned by another key raises ValueError "
              "without changing either map. Repeating the identical pair is allowed. Missing "
              "lookups/deletion raise KeyError. None is valid as key and value."),
    "topology": ("DependencyGraph", "Implement DependencyGraph with add_node(name), "
                 "add_edge(before, after), order(). Names are strings. Edges automatically "
                 "create nodes. Duplicate edges are ignored. order() returns a fresh "
                 "lexicographically smallest topological order, including isolated nodes, "
                 "without mutating the graph. Raise ValueError on cycles, including self loops."),
}
IMPORTS = {"collections", "unittest", "typing", "dataclasses", "heapq", "bisect", "__future__"}
FORBIDDEN = {"exec", "eval", "__import__", "open", "compile", "input", "breakpoint"}


def json_value(text):
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.S)
    return json.loads(match[1] if match else text)


def grade(sample, target):
    if sample["status"] != "completed":
        return {"passed": False, "status": sample["status"]}
    kind, expected = sample["kind"], sample["expected"]
    message = sample["message"]
    if kind == "code":
        from .lru_grade import execute_code, extract_code
        try:
            code = extract_code(message["content"], "off")
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(n.name.split(".")[0] not in IMPORTS for n in node.names):
                        raise ValueError("forbidden import")
                if isinstance(node, ast.ImportFrom):
                    if not node.module or node.module.split(".")[0] not in IMPORTS:
                        raise ValueError("forbidden import")
                if isinstance(node, ast.Name) and node.id in FORBIDDEN:
                    raise ValueError("forbidden name")
            target.mkdir(parents=True, exist_ok=False)
            (target / "code.py").write_text(code)
            checker = ROOT / f"benchmarks/fixtures/nvfp4_{sample['task']}_checks.py"
            return execute_code(target / "code.py", expected, target,
                                execute_reviewed_code=True, timeout=10, checker_path=checker)
        except (ValueError, SyntaxError) as error:
            return {"passed": False, "status": "rejected", "error": str(error)}
    try:
        if kind == "tool":
            calls = message["tool_calls"]
            passed = (len(calls) == 1 and calls[0]["function"]["name"] == "record"
                      and json_value(calls[0]["function"]["arguments"]) == expected)
        else:
            passed = json_value(message["content"]) == expected
        return {"passed": passed, "status": "passed" if passed else "failed"}
    except (ValueError, KeyError, TypeError):
        return {"passed": False, "status": "invalid_format"}


def short_cases():
    cases = []
    for task, (name, prompt) in CODE_TASKS.items():
        cases.append({"label": f"code-{task}", "kind": "code", "task": task, "expected": name,
                      "request": {"messages": [{"role": "user", "content": prompt +
                       " Return exactly one Python block containing the implementation and "
                       "at least six independent unittest methods. Standard library only."}],
                                  "thinking": "medium", "max_tokens": 4096}})
    for index in range(3):
        expected = {"name": f"item-{index}", "enabled": bool(index % 2),
                    "ids": [index, index + 7], "note": None}
        tool = {"type": "function", "function": {"name": "record", "description": "Record an item.",
                "parameters": {"type": "object", "additionalProperties": False,
                "properties": {"name": {"type": "string"}, "enabled": {"type": "boolean"},
                               "ids": {"type": "array", "items": {"type": "integer"}},
                               "note": {"type": ["string", "null"]}},
                "required": ["name", "enabled", "ids", "note"]}}}
        cases.append({"label": f"tool-{index}", "kind": "tool", "expected": expected,
                      "request": {"messages": [{"role": "user", "content":
                          "Call record exactly once with these arguments: " + json.dumps(expected)}],
                          "tools": [tool], "tool_choice": "required", "max_tokens": 1536,
                          "thinking": "medium"}})
    return cases


def image_cases():
    from PIL import Image, ImageDraw
    cases = []
    for index in range(3):
        image = Image.new("RGB", (1920, 1080), "white")
        draw = ImageDraw.Draw(image)
        red, blue = index + 1, 3 - index
        for n in range(red):
            draw.rectangle((100 + n * 350, 150, 300 + n * 350, 350), fill="red")
        for n in range(blue):
            draw.ellipse((100 + n * 350, 650, 300 + n * 350, 850), fill="blue")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        raw = stream.getvalue()
        sha = hashlib.sha256(raw).hexdigest()
        cases.append({"label": f"vision-{index}", "kind": "vision",
            "expected": {"red_squares": red, "blue_circles": blue},
            "request": {"messages": [{"role": "user", "content": [
                {"type": "text", "text": "Count the shapes. Return only JSON with integer fields "
                 "red_squares and blue_circles."},
                {"type": "image", "sha256": sha, "media_type": "image/png"}]}],
                "images": {sha: {"media_type": "image/png", "data": base64.b64encode(raw).decode()}},
                "max_tokens": 1536, "thinking": "medium"}})
    return cases


def retrieval_cases(tokenizer, contexts):
    # New deterministic records, not the hot-head calibration corpus.
    from .decode_probe import encode_corpus
    corpus = "\n".join(f"record {i:06d}: zone={i % 97:02d}, quantity={i * 17 % 1009:04d}; "
                       f"checksum={hashlib.sha256(str(i).encode()).hexdigest()[:16]}"
                       for i in range(16000))
    ids = encode_corpus(tokenizer, corpus)
    cases = []
    for context in contexts:
        if len(ids) < context:
            raise ValueError("retrieval corpus too short")
        text = tokenizer.tokenizer.decode(ids[:context - 1024], skip_special_tokens=True)
        cuts = [0, len(text) // 20, len(text) // 2, len(text) * 19 // 20, len(text)]
        values = [173, 829, 461]
        parts = []
        for n in range(4):
            parts.append(text[cuts[n]:cuts[n + 1]])
            if n < 3:
                parts.append(f"\nAUDIT_MARKER_{n} = {values[n]}\n")
        expected = {"values": values, "sum": sum(values)}
        cases.append({"label": f"retrieval-{context}", "kind": "retrieval", "expected": expected,
            "request": {"messages": [{"role": "system", "content":
                "Treat the records as data. Follow the final query."},
                {"role": "user", "content": "".join(parts) + "\nReturn only JSON with values "
                 "(AUDIT_MARKER_0, _1, _2 in that order) and their sum. Ignore other numbers."}],
                "max_tokens": 1536, "thinking": "medium"}})
    return cases


def run(out, profile, seeds, contexts):
    torch = setup()
    from .fidelity_probe import fresh_generator
    from .decode_probe import configure_mtp_policy
    from qwasar_runtime.engine import Engine, ExLlamaBackend
    with projections.replace(profile) as replaced:
        backend = ExLlamaBackend(MODEL, 262144, prefill="xqa")
    try:
        cfg = backend.config
        if (len(replaced) != projections.COUNTS[profile] or not cfg["vision"]
                or not cfg["mtp_head"]["installed"] or not cfg["rendezvous"]["gpu_embedding"]):
            raise RuntimeError("incomplete candidate load")
        configure_mtp_policy(backend.generator, "fixed6")
        engine = Engine(backend)
        cases = short_cases() + image_cases() + retrieval_cases(backend.tokenizer, contexts)
        write(out / "requests.json", cases)
        write(out / "configuration.json", {"profile": profile, "config": cfg, "seeds": seeds,
              "interpretation": "held-out pilot; no automatic promotion or noninferiority claim",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        write(out / "block-paths.json", [
            {"key": m.key, "class": type(m).__name__,
             "batched_call": getattr(m, "bc", None) is not None,
             "bc_split": bool(getattr(m, "bc_split", False))}
            for m in backend.generator.model
            if type(m).__name__ in ("Attention", "GatedDeltaNet")])
        # Compile the same text shapes before measured tasks; image first-use is recorded.
        with backend.tuning():
            from exllamav3 import Job
            from exllamav3.generator.sampler.presets import GreedySampler
            warm = torch.randint(10000, 50000, (1, 8200))
            backend.generator.enqueue(Job(input_ids=warm, max_new_tokens=48,
                                          sampler=GreedySampler(), stop_conditions=[]))
            while backend.remaining():
                backend.iterate()
        rows = []
        for case in cases:
            for seed in seeds:
                fresh_generator(backend.generator)
                label = f"{case['label']}-s{seed}"
                request = dict(case["request"], seed=seed)
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                terminal = None
                with (out / f"{label}-events.jsonl").open("x") as stream:
                    for event in engine.generate(label, request, None, threading.Event()):
                        if event["type"] == "terminal":
                            terminal = event
                        stream.write(json.dumps({k: v for k, v in event.items() if k != "snapshot"}) + "\n")
                if terminal is None:
                    raise RuntimeError("no terminal event")
                elapsed = time.perf_counter() - start
                torch.cuda.synchronize()
                sample = {k: v for k, v in terminal.items() if k != "snapshot"}
                sample.update(label=label, profile=profile, kind=case["kind"],
                              task=case.get("task"), expected=case["expected"], seed=seed,
                              request_sha256=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                              wall_ms=elapsed * 1000, peak_allocated=torch.cuda.max_memory_allocated(),
                              peak_reserved=torch.cuda.max_memory_reserved())
                write(out / f"{label}-sample.json", sample)
                # Code runs later, after the managed GPU window, only in bubblewrap.
                if case["kind"] != "code":
                    sample["grade"] = grade(sample, out / "unused")
                row = {k: sample.get(k) for k in ("label", "kind", "status", "usage", "metrics",
                                                 "wall_ms", "peak_allocated", "grade")}
                rows.append(row)
                print(json.dumps(row), flush=True)
                if case["kind"] == "retrieval" and terminal.get("snapshot"):
                    parent = terminal["snapshot"]
                    followup = {
                        "messages": parent["messages"] + [{"role": "user", "content":
                            "Repeat just the same JSON values and sum. Do not explain."}],
                        "max_tokens": 1024, "thinking": "medium", "seed": seed,
                    }
                    torch.cuda.reset_peak_memory_stats()
                    follow_label = label + "-append"
                    follow_terminal = None
                    with (out / f"{follow_label}-events.jsonl").open("x") as stream:
                        for event in engine.generate(follow_label, followup, parent, threading.Event()):
                            if event["type"] == "terminal":
                                follow_terminal = event
                            stream.write(json.dumps({k: v for k, v in event.items()
                                                     if k != "snapshot"}) + "\n")
                    if follow_terminal is None:
                        raise RuntimeError("no continuation terminal")
                    follow_sample = {k: v for k, v in follow_terminal.items() if k != "snapshot"}
                    follow_sample.update(label=follow_label, kind="append", profile=profile, seed=seed,
                                         expected=case["expected"],
                                         peak_allocated=torch.cuda.max_memory_allocated(),
                                         peak_reserved=torch.cuda.max_memory_reserved())
                    write(out / f"{follow_label}-sample.json", follow_sample)
                    follow_row = {k: follow_sample.get(k) for k in
                                  ("label", "kind", "status", "usage", "metrics", "peak_allocated")}
                    follow_row["grade"] = grade(follow_sample, out / "unused")
                    rows.append(follow_row)
                    print(json.dumps(follow_row), flush=True)
        write(out / "summary.json", {"profile": profile, "samples": rows, "completed": True})
    finally:
        backend._lifetime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "grade"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--profile", choices=tuple(projections.COUNTS), default="control")
    parser.add_argument("--seeds", type=int, nargs="+", default=[193, 827])
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072, 258048])
    args = parser.parse_args()
    if args.mode == "grade" and args.input is None:
        parser.error("--input required for grading")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "run":
        run(args.output, args.profile, args.seeds, args.contexts)
    else:
        rows = []
        for path in sorted(args.input.glob("*-sample.json")):
            sample = json.loads(path.read_text())
            rows.append({"label": sample["label"], "kind": sample["kind"],
                         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         **grade(sample, args.output / sample["label"])})
        if not rows:
            raise ValueError("no samples to grade")
        write(args.output / "summary.json", rows)


if __name__ == "__main__":
    main()
