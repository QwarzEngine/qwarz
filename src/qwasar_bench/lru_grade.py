"""Extract LRU submissions; execute reviewed code only inside bubblewrap."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys


CHECKER = Path(__file__).resolve().parents[2] / "benchmarks/fixtures/lru_checks.py"
RUNNER = '''import contextlib
import importlib.util
import json
import resource
import sys
import traceback
import unittest

resource.setrlimit(resource.RLIMIT_CPU, (int(sys.argv[2]), int(sys.argv[2])))
resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def failure():
    traceback.print_exc(file=sys.stderr)
    return {"error": traceback.format_exc()}

report = {"python": sys.version, "generated": None, "oracle": None}
with contextlib.redirect_stdout(sys.stderr):
    try:
        checker = load("frozen_checker", "/work/checker.py")
        submission = load("generated_lru", "/work/code.py")
    except BaseException:
        report["import_error"] = failure()
    else:
        try:
            suite = unittest.defaultTestLoader.loadTestsFromModule(submission)
            discovered = suite.countTestCases()
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
            report["generated"] = {
                "discovered": discovered, "tests_run": result.testsRun,
                "failures": len(result.failures), "errors": len(result.errors),
                "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
                "unexpected_successes": len(result.unexpectedSuccesses),
                "successful": result.wasSuccessful(),
            }
        except BaseException:
            report["generated"] = failure()
        try:
            report["oracle"] = {"checks": checker.check_lru(getattr(submission, sys.argv[1]))}
        except BaseException:
            report["oracle"] = failure()
json.dump(report, sys.stdout, allow_nan=False)
sys.stdout.write("\\n")
'''


def digest(content):
    return hashlib.sha256(content).hexdigest()


def extract_code(completion, thinking):
    if not isinstance(completion, str):
        raise ValueError("completion must be text")
    if thinking not in ("off", "medium", "low", "xhigh"):
        raise ValueError("run.json must identify the thinking mode")
    if thinking != "off":
        if completion.count("</think>") != 1:
            raise ValueError("thinking output must have exactly one closing </think> marker")
        completion = completion.split("</think>", 1)[1]
    if "```" in completion:
        match = re.search(r"(?m)^```python[ \t]*\r?\n([\s\S]*?)^```[ \t]*(?:\r?\n|$)", completion)
        if completion.count("```") != 2 or match is None:
            raise ValueError("expected exactly one complete python fence")
        code = match.group(1)
    else:
        code = completion
    if not code.strip():
        raise ValueError("empty code")
    try:
        ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f"code syntax is invalid: {error}") from error
    return code


def discover_cache_class(code):
    candidates = []
    required = {"get", "put", "delete", "__len__"}
    for node in ast.parse(code).body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {member.name for member in node.body if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if required <= methods or (node.name == "LRUCache" and node.bases):
            candidates.append(node.name)
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous top-level LRU implementation class")
    return candidates[0]


def execution_status(report):
    if report.get("import_error"):
        return "execution_error"
    generated, oracle = report.get("generated"), report.get("oracle")
    if not isinstance(generated, dict) or not isinstance(oracle, dict) or generated.get("error"):
        return "execution_error"
    fields = ("discovered", "tests_run", "failures", "errors", "skipped", "expected_failures", "unexpected_successes")
    if any(type(generated.get(key)) is not int or generated[key] < 0 for key in fields):
        return "execution_error"
    effective = generated["tests_run"] - generated["skipped"] - generated["expected_failures"]
    if generated["discovered"] < 6 or effective < 6:
        return "insufficient_tests"
    passed = (generated.get("successful") is True
              and not any(generated[key] for key in ("failures", "errors", "unexpected_successes"))
              and type(oracle.get("checks")) is int and oracle["checks"] == 7 and not oracle.get("error"))
    return "passed" if passed else "failed"


def execute_code(code_path, cache_class, output_dir, *, execute_reviewed_code=False,
                 timeout=10., checker_path=None, runner_path=None):
    if not execute_reviewed_code:
        raise PermissionError("execution requires explicit --execute-reviewed-code")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        raise RuntimeError("bubblewrap is required; host execution is prohibited")
    output_dir = Path(output_dir)
    checker_path = Path(checker_path or CHECKER)
    if runner_path is None:
        runner_path = output_dir / "runner.py"
        runner_path.write_text(RUNNER)
    command = [
        bubblewrap, "--unshare-all", "--new-session", "--die-with-parent", "--cap-drop", "ALL",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--size", "16777216", "--tmpfs", "/tmp",
        "--dir", "/work", "--ro-bind", str(Path(code_path).resolve()), "/work/code.py",
        "--ro-bind", str(checker_path.resolve()), "/work/checker.py",
        "--ro-bind", str(Path(runner_path).resolve()), "/work/runner.py",
        "--clearenv", "--setenv", "PATH", "/usr/bin", "--chdir", "/work",
        "/usr/bin/python", "-I", "/work/runner.py", cache_class, str(max(1, math.ceil(timeout))),
    ]
    channel_path = output_dir / "result-channel.json"
    with channel_path.open("xb") as channel, (output_dir / "execution.log").open("xb") as log:
        try:
            process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=channel, stderr=log, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "passed": False, "timeout_seconds": timeout, "command": command}
    record = {"returncode": process.returncode, "command": command, "passed": False}
    try:
        report = json.loads(channel_path.read_text())
        if not isinstance(report, dict):
            raise ValueError("runner result must be an object")
        status = execution_status(report)
    except (ValueError, UnicodeError) as error:
        record.update(status="execution_error", error=f"invalid result channel: {error}")
        return record
    record.update(execution=report, status=status if process.returncode == 0 else "execution_error")
    record["passed"] = record["status"] == "passed"
    return record


def grade_run(run_dir, output_dir, *, execute_reviewed_code=False, timeout=10.):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    metadata_bytes = (run_dir / "run.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    thinking = metadata.get("arguments", {}).get("thinking")
    if thinking not in ("off", "medium", "low", "xhigh"):
        raise ValueError("run.json must identify the thinking mode")
    samples = sorted(run_dir.glob("sample-*.json"))
    if not samples:
        raise ValueError("run has no sample-*.json files")
    output_dir.mkdir(parents=True, exist_ok=False)
    checker = CHECKER.read_bytes()
    (output_dir / "checker.py").write_bytes(checker)
    (output_dir / "runner.py").write_text(RUNNER)
    result = {"run": str(run_dir.resolve()), "run_metadata_sha256": digest(metadata_bytes),
              "checker_sha256": digest(checker), "runner_sha256": digest(RUNNER.encode()),
              "grader_sha256": digest(Path(__file__).read_bytes()), "host_python": sys.version,
              "execute_reviewed_code": bool(execute_reviewed_code), "thinking": thinking,
              "scope": "Generated unittest suite and frozen independent LRU checker; code is never repaired.",
              "samples": []}
    for sample_path in samples:
        sample_dir = output_dir / sample_path.stem
        sample_dir.mkdir()
        sample_bytes = sample_path.read_bytes()
        (sample_dir / "sample.json").write_bytes(sample_bytes)
        record = {"sample": sample_path.name, "sample_sha256": digest(sample_bytes), "passed": False}
        try:
            sample = json.loads(sample_bytes)
            if sample.get("truncated") is not False or sample.get("finish_reason") == "max_new_tokens":
                record["status"] = "truncated"
            else:
                code = extract_code(sample.get("completion"), thinking)
                (sample_dir / "code.py").write_bytes(code.encode())
                record["code_sha256"] = digest(code.encode())
                record["cache_class"] = discover_cache_class(code)
                record["status"] = "extracted"
                if execute_reviewed_code:
                    record.update(execute_code(
                        sample_dir / "code.py", record["cache_class"], sample_dir,
                        execute_reviewed_code=True, timeout=timeout,
                        checker_path=output_dir / "checker.py", runner_path=output_dir / "runner.py",
                    ))
        except Exception as error:
            record.update(status="error", error_type=type(error).__name__, error=str(error))
        (sample_dir / "grade.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        result["samples"].append(record)
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-reviewed-code", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.)
    args = parser.parse_args(argv)
    result = grade_run(args.run, args.output, execute_reviewed_code=args.execute_reviewed_code, timeout=args.timeout)
    print(json.dumps({"output": str(args.output), "samples": len(result["samples"]),
                      "passed": sum(sample["passed"] for sample in result["samples"])}))
    expected = "passed" if args.execute_reviewed_code else "extracted"
    return 0 if all(sample["status"] == expected for sample in result["samples"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
