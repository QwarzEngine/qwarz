from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from qwasar_bench import lru_grade as grade


VALID_CODE = '''import collections
import unittest

class LRUCache:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity")
        self.capacity = capacity
        self.data = collections.OrderedDict()

    def get(self, key):
        value = self.data[key]
        self.data.move_to_end(key)
        return value

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)

    def delete(self, key):
        del self.data[key]

    def __len__(self):
        return len(self.data)

class GeneratedChecks(unittest.TestCase):
    def test_invalid(self):
        with self.assertRaises(ValueError):
            LRUCache(0)

    def test_missing(self):
        with self.assertRaises(KeyError):
            LRUCache(1).get("missing")

    def test_put(self):
        cache = LRUCache(1)
        cache.put("key", 3)
        self.assertEqual(cache.get("key"), 3)

    def test_none(self):
        cache = LRUCache(1)
        cache.put("key", None)
        self.assertIsNone(cache.get("key"))

    def test_delete(self):
        cache = LRUCache(1)
        cache.put("key", 3)
        cache.delete("key")
        self.assertEqual(len(cache), 0)

    def test_update(self):
        cache = LRUCache(1)
        cache.put("key", 3)
        cache.put("key", 4)
        self.assertEqual(cache.get("key"), 4)
        self.assertEqual(len(cache), 1)
'''


def make_run(tmp_path, code=VALID_CODE, **sample_fields):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"arguments": {"thinking": "off"}}))
    (run / "sample-32768-1.json").write_text(json.dumps({
        "completion": code, "truncated": False, "finish_reason": "eos", **sample_fields,
    }))
    return run


def test_extraction_preserves_exact_fenced_or_plain_code_and_thinking_boundary():
    code = "\nclass Example:\n    pass\n"
    assert grade.extract_code(code, "off") == code
    assert grade.extract_code("Reasoning</think>\nHere:\n```python\n" + code + "```\n", "medium") == code


@pytest.mark.parametrize(("text", "thinking"), [
    ("still reasoning\n```python\npass\n```", "medium"),
    ("```python\npass\n```\n```python\npass\n```", "off"),
    ("```javascript\npass\n```", "off"),
    ("```python\npass", "off"),
    ("Here is code:\npass", "off"),
    ("reason</think>answer</think>pass", "medium"),
])
def test_ambiguous_or_incomplete_extraction_is_rejected(text, thinking):
    with pytest.raises(ValueError):
        grade.extract_code(text, thinking)


def test_ast_discovers_behavioral_class_or_inherited_lru_name_and_rejects_ambiguity():
    assert grade.discover_cache_class(VALID_CODE) == "LRUCache"
    assert grade.discover_cache_class("class LRUCache(ExternalBase):\n    pass\n") == "LRUCache"
    with pytest.raises(ValueError):
        grade.discover_cache_class(VALID_CODE + "\nclass Other:\n    get=put=delete=__len__=None\n\nclass LRUCache2:\n    def get(self): pass\n    def put(self): pass\n    def delete(self): pass\n    def __len__(self): pass\n")


def test_default_mode_only_extracts_and_never_executes_code(tmp_path):
    run = make_run(tmp_path, "raise RuntimeError('must not execute')\n" + VALID_CODE)
    result = grade.grade_run(run, tmp_path / "review")
    assert result["samples"][0]["status"] == "extracted"
    assert result["samples"][0]["passed"] is False
    assert not (tmp_path / "review/sample-32768-1/execution.log").exists()
    assert (tmp_path / "review/sample-32768-1/code.py").read_text() == "raise RuntimeError('must not execute')\n" + VALID_CODE


def test_execution_api_requires_explicit_review_opt_in(tmp_path):
    with pytest.raises(PermissionError):
        grade.execute_code(tmp_path / "missing.py", "LRUCache", tmp_path, timeout=1)


def test_truncated_sample_never_enters_execution_even_with_flag(tmp_path):
    run = make_run(tmp_path, truncated=True)
    result = grade.grade_run(run, tmp_path / "graded", execute_reviewed_code=True)
    assert result["samples"][0]["status"] == "truncated"
    assert result["samples"][0]["passed"] is False
    assert not (tmp_path / "graded/sample-32768-1/execution.log").exists()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap unavailable")
@pytest.mark.parametrize(("code", "expected_status"), [
    (VALID_CODE, "passed"),
    (VALID_CODE.replace("self.data.move_to_end(key)", "pass", 1), "failed"),
    ("raise RuntimeError('benign fixture error')\n" + VALID_CODE, "execution_error"),
    ("import time\ntime.sleep(5)\n" + VALID_CODE, "timeout"),
    (VALID_CODE.split("class GeneratedChecks")[0], "insufficient_tests"),
    (VALID_CODE.replace("class GeneratedChecks", "@unittest.skip('benign fixture')\nclass GeneratedChecks"), "insufficient_tests"),
    ("import os\nos.write(1, b'not-json\\n')\n" + VALID_CODE, "execution_error"),
])
def test_reviewed_benign_fixtures_execute_only_in_sandbox(tmp_path, code, expected_status):
    import hashlib

    run = make_run(tmp_path, code)
    result = grade.grade_run(run, tmp_path / "graded", execute_reviewed_code=True, timeout=.3 if expected_status == "timeout" else 5)
    sample = result["samples"][0]
    assert sample["status"] == expected_status
    assert sample["passed"] is (expected_status == "passed")
    assert sample["code_sha256"] == hashlib.sha256(code.encode()).hexdigest()
    assert sample["sample_sha256"] == hashlib.sha256((run / "sample-32768-1.json").read_bytes()).hexdigest()
    assert result["checker_sha256"] == hashlib.sha256(grade.CHECKER.read_bytes()).hexdigest()
    if expected_status in ("passed", "failed", "insufficient_tests"):
        assert sample["execution"]["generated"]["tests_run"] == (0 if "class GeneratedChecks" not in code else 6)
    if expected_status == "passed":
        assert sample["execution"]["oracle"]["checks"] == 7


def test_grading_output_directory_cannot_be_reused(tmp_path):
    run = make_run(tmp_path)
    grade.grade_run(run, tmp_path / "review")
    with pytest.raises(FileExistsError):
        grade.grade_run(run, tmp_path / "review")


@pytest.mark.parametrize("checks", [None, True, 6, 8])
def test_success_requires_all_seven_independent_checks(checks):
    report = {"generated": {"discovered": 6, "tests_run": 6, "failures": 0, "errors": 0,
                            "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
                            "successful": True}, "oracle": {"checks": checks}}
    assert grade.execution_status(report) == "failed"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap unavailable")
def test_sandbox_hides_host_files_and_environment_and_applies_resource_limits(tmp_path, monkeypatch):
    marker = tmp_path / "host-only"
    marker.write_text("not visible in sandbox")
    monkeypatch.setenv("QWASAR_GRADER_TEST_VALUE", "host-only")
    guards = f'''import os
from pathlib import Path
import resource
assert not Path({str(marker)!r}).exists()
assert not Path('/home').exists()
assert os.environ.get('QWASAR_GRADER_TEST_VALUE') is None
assert resource.getrlimit(resource.RLIMIT_CPU) == (5, 5)
assert resource.getrlimit(resource.RLIMIT_AS) == (536870912, 536870912)
assert resource.getrlimit(resource.RLIMIT_NPROC) == (32, 32)
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (1048576, 1048576)
try:
    Path('/work/code.py').open('a')
except OSError:
    pass
else:
    raise AssertionError('submission mount is writable')
print('submission stdout belongs only in logs')
'''
    run = make_run(tmp_path, guards + VALID_CODE)
    output = tmp_path / "graded"
    result = grade.grade_run(run, output, execute_reviewed_code=True, timeout=5)
    assert result["samples"][0]["passed"] is True
    assert marker.read_text() == "not visible in sandbox"
    assert "submission stdout belongs only in logs" in (output / "sample-32768-1/execution.log").read_text()
    assert "submission stdout belongs only in logs" not in (output / "sample-32768-1/result-channel.json").read_text()


def test_medium_without_closing_think_is_not_treated_as_complete_code(tmp_path):
    run = make_run(tmp_path)
    (run / "run.json").write_text(json.dumps({"arguments": {"thinking": "medium"}}))
    result = grade.grade_run(run, tmp_path / "review", execute_reviewed_code=True)
    assert result["samples"][0]["status"] == "error"
    assert result["samples"][0]["passed"] is False
    assert not (tmp_path / "review/sample-32768-1/execution.log").exists()
