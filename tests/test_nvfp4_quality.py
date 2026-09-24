import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from qwasar_bench.nvfp4_quality import CODE_TASKS, grade, json_value, short_cases
from qwasar_bench.nvfp4_analysis import paired_counts, quality_summary, retrieval_semantic


def oracle(task):
    path = Path(__file__).resolve().parents[1] / f"benchmarks/fixtures/nvfp4_{task}_checks.py"
    spec = importlib.util.spec_from_file_location(f"check_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_lru


def test_pilot_cases_are_distinct_from_calibration_families():
    cases = short_cases()
    assert len(cases) == len({c["label"] for c in cases}) == 6
    assert set(CODE_TASKS) == {"interval", "bimap", "topology"}
    assert not {"lru", "ring", "bucket", "ttl"} & set(CODE_TASKS)
    assert all(c["request"]["thinking"] == "medium" for c in cases)


def test_missing_quality_never_passes():
    with pytest.raises(ValueError, match="missing"):
        paired_counts({"x": {"kind": "code", "passed": True}}, {})
    with pytest.raises(ValueError, match="verdict"):
        paired_counts({"x": {"kind": "code", "passed": True}}, {"x": {"kind": "code"}})


def test_paired_outcomes_not_just_aggregate_score():
    a = {str(i): {"kind": "vision", "passed": bool(i & 1)} for i in range(4)}
    b = {str(i): {"kind": "vision", "passed": bool(i & 2)} for i in range(4)}
    result = paired_counts(a, b)
    assert result["counts"] == {"both_fail": 1, "regression": 1, "improvement": 1, "both_pass": 1}
    assert result["delta_success_pp"] == 0


def test_aggregation_rejects_stale_grades(tmp_path):
    root, grades = tmp_path / "quality", tmp_path / "grades"
    root.mkdir()
    sequence = [("control", 193), ("attention", 193)]
    (root / "completed.json").write_text(json.dumps({"completed": True, "sequence": sequence}))
    label = "vision-0-s193"
    for index, (profile, seed) in enumerate(sequence):
        name = f"{index:02}-{profile}-s{seed}"
        run, grade_dir = root / name, grades / name
        run.mkdir()
        grade_dir.mkdir(parents=True)
        sample = {"label": label, "kind": "vision", "status": "completed", "request_sha256": "same",
                  "metrics": {}, "usage": {}, "peak_allocated": 123}
        raw = json.dumps(sample).encode()
        (run / f"{label}-sample.json").write_bytes(raw)
        (run / "summary.json").write_text(json.dumps({"completed": True, "samples": [{"label": label}]}))
        row = {"label": label, "kind": "vision", "passed": True,
               "source_sha256": hashlib.sha256(raw).hexdigest()}
        (grade_dir / "summary.json").write_text(json.dumps([row]))
    assert quality_summary(root, grades)["quality_strict"]["pairs"] == 1
    (root / "01-attention-s193" / f"{label}-sample.json").write_text("{}")
    with pytest.raises(ValueError, match="stale grade"):
        quality_summary(root, grades)


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
def test_incomplete_is_never_a_pass(status, tmp_path):
    assert not grade({"status": status}, tmp_path)["passed"]


def test_json_grader_is_exact_and_missing_is_failure(tmp_path):
    base = {"status": "completed", "kind": "vision", "expected": {"red_squares": 2}}
    assert grade({**base, "message": {"content": '{"red_squares":2}'}}, tmp_path)["passed"]
    assert not grade({**base, "message": {"content": '{"red_squares":3}'}}, tmp_path)["passed"]
    assert not grade({**base, "message": {"content": 'text before {"red_squares":2}'}}, tmp_path)["passed"]
    assert json_value('```json\n{"a":1}\n```') == {"a": 1}


def test_tool_grader_checks_name_arguments_and_count(tmp_path):
    base = {"status": "completed", "kind": "tool", "expected": {"x": None}}
    call = {"function": {"name": "record", "arguments": '{"x":null}'}}
    assert grade({**base, "message": {"tool_calls": [call]}}, tmp_path)["passed"]
    assert not grade({**base, "message": {"tool_calls": [call, call]}}, tmp_path)["passed"]
    assert not grade({**base, "message": {"tool_calls": []}}, tmp_path)["passed"]


def test_retrieval_semantic_does_not_repair_wrong_numbers():
    sample = {"status": "completed", "expected": {"values": [173, 829, 461], "sum": 1463}}
    for text in ('{"values":[173,829,461],"sum":1463}',
                 '{"AUDIT_MARKER_0":173,"AUDIT_MARKER_1":829,"AUDIT_MARKER_2":461,"sum":1463}'):
        assert retrieval_semantic({**sample, "message": {"content": text}})
    assert not retrieval_semantic({**sample, "message": {"content":
        '{"AUDIT_MARKER_0":173,"AUDIT_MARKER_1":829,"AUDIT_MARKER_2":462,"sum":1464}'}})
    assert not retrieval_semantic({**sample, "message": {"content": '{"sum":1463}'}})


def test_code_is_not_executed_when_static_screen_rejects(tmp_path):
    sample = {"status": "completed", "kind": "code", "expected": "BiMap", "task": "bimap",
              "message": {"content": "```python\nimport os\n```"}}
    assert grade(sample, tmp_path / "grade")["status"] == "rejected"
    assert not (tmp_path / "grade").exists()


def test_interval_oracle_and_red_control():
    class IntervalSet:
        def __init__(self):
            self.values = set()

        def validate(self, lo, hi):
            if type(lo) is not int or type(hi) is not int or lo >= hi:
                raise ValueError()

        def add(self, lo, hi):
            self.validate(lo, hi)
            self.values.update(range(lo, hi))

        def remove(self, lo, hi):
            self.validate(lo, hi)
            self.values.difference_update(range(lo, hi))

        def contains(self, value):
            return value in self.values

        def intervals(self):
            result = []
            for value in sorted(self.values):
                if result and result[-1][1] == value:
                    result[-1] = (result[-1][0], value + 1)
                else:
                    result.append((value, value + 1))
            return result

    check = oracle("interval")
    assert check(IntervalSet) == 7

    class Broken(IntervalSet):
        def remove(self, lo, hi):
            pass

    with pytest.raises(AssertionError):
        check(Broken)


def test_bimap_oracle_and_red_control():
    class BiMap:
        def __init__(self):
            self.data = {}

        def __len__(self):
            return len(self.data)

        def get(self, key):
            return self.data[key]

        def inverse(self, value):
            for key, stored in self.data.items():
                if stored == value:
                    return key
            raise KeyError(value)

        def put(self, key, value):
            if any(k != key and v == value for k, v in self.data.items()):
                raise ValueError()
            self.data[key] = value

        def delete(self, key):
            del self.data[key]

    check = oracle("bimap")
    assert check(BiMap) == 7

    class Broken(BiMap):
        def put(self, key, value):
            self.data[key] = value

    with pytest.raises(AssertionError):
        check(Broken)


def test_topology_oracle_and_red_control():
    class Graph:
        def __init__(self):
            self.edges = {}

        def add_node(self, name):
            self.edges.setdefault(name, set())

        def add_edge(self, before, after):
            self.add_node(before)
            self.add_node(after)
            self.edges[before].add(after)

        def order(self):
            pending = set(self.edges)
            out = []
            while pending:
                free = sorted(n for n in pending if not any(n in self.edges[m] for m in pending))
                if not free:
                    raise ValueError()
                out.append(free[0])
                pending.remove(free[0])
            return out

    check = oracle("topology")
    assert check(Graph) == 7

    class Broken(Graph):
        def order(self):
            return sorted(self.edges)

    with pytest.raises(AssertionError):
        check(Broken)
