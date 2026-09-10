#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PI = Path.home() / ".local/share/mise/installs/pi/0.84.4/pi/pi"


def main():
    parser = argparse.ArgumentParser(description="Exercise the real Pi client against the running Qwasar service")
    parser.add_argument("--pi", type=Path, default=DEFAULT_PI)
    parser.add_argument("--output", type=Path, default=ROOT / "results/pi-v1")
    parser.add_argument("--thinking", choices=["off", "medium"], default="medium")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwasar-pi-") as temporary:
        directory = Path(temporary)
        config = directory / "config"
        fixture = directory / "fixture"
        config.mkdir()
        fixture.mkdir()
        (config / "models.json").write_bytes((ROOT / "integrations/pi/models.json").read_bytes())
        (fixture / "ledger.py").write_text("def add(total, amount):\n    return total - amount\n")
        original_test = "import unittest\nfrom ledger import add\n\nclass LedgerTest(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(3, 2), 5)\n"
        (fixture / "test_ledger.py").write_text(original_test)
        prompt = "Read ledger.py and test_ledger.py using read. Fix the addition bug in ledger.py using edit. Write a new test_negative.py using write, with a unittest asserting add(3, -2) equals 1. Run python3 -m unittest -v using bash, then briefly report the test results. Use each of read, edit, write and bash. Do not change the existing test assertion."
        environment = dict(os.environ, PI_CODING_AGENT_DIR=str(config), PI_OFFLINE="1")
        command = [str(arguments.pi), "--provider", "qwasar", "--model", "qwasar-qwen38-27b",
                   "--thinking", arguments.thinking, "--mode", "json", "--no-session",
                   "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes",
                   "--no-context-files", "--no-approve", "--tools", "read,write,edit,bash", prompt]
        started = time.monotonic()
        with (output / "pi-events.jsonl").open("w") as events, (output / "pi-stderr.log").open("w") as errors:
            result = subprocess.run(command, cwd=fixture, env=environment, stdout=events, stderr=errors, timeout=300)
        elapsed = time.monotonic() - started
        records = []
        for line in (output / "pi-events.jsonl").read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        tools = [event for event in records if event.get("type") == "tool_execution_end"]
        names = {event.get("toolName") for event in tools}
        check = subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=fixture, text=True, capture_output=True)
        oracle = subprocess.run([sys.executable, "-c", "from ledger import add; assert add(3, 2) == 5; assert add(3, -2) == 1; assert add(0, 0) == 0"], cwd=fixture, text=True, capture_output=True)
        (output / "tests.log").write_text(check.stdout + check.stderr)
        (output / "oracle.log").write_text(oracle.stdout + oracle.stderr)
        sources = {source.name: source.read_text() for source in fixture.glob("*.py")}
        (output / "fixture.json").write_text(json.dumps(sources, indent=2))
        messages = [event.get("message", {}) for event in records if event.get("type") == "message_end"]
        passed = result.returncode == 0 and check.returncode == 0 and oracle.returncode == 0 and sources["test_ledger.py"] == original_test and "test_negative.py" in sources and {"read", "write", "edit", "bash"} <= names and not any(event.get("isError") for event in tools) and not any(message.get("stopReason") == "error" for message in messages)
        summary = {"passed": passed, "pi_version": subprocess.check_output([str(arguments.pi), "--version"], text=True).strip(),
                   "thinking": arguments.thinking, "elapsed_seconds": elapsed, "pi_exit": result.returncode,
                   "tool_names": sorted(names), "tool_steps": len(tools), "test_exit": check.returncode, "oracle_exit": oracle.returncode,
                   "message_count": len(messages), "isolated_configuration": True}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
