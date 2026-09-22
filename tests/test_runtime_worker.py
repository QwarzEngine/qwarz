import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class WorkerTests(unittest.TestCase):
    def start(self, context_size="262144"):
        process = subprocess.Popen([sys.executable, "-m", "qwasar_runtime.worker", "--fake", "--context-size", context_size],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
        self.addCleanup(self.cleanup, process)
        ready = json.loads(process.stdout.readline())
        self.assertEqual(ready["type"], "ready")
        self.assertTrue(ready["config"]["fake"])
        return process

    @staticmethod
    def cleanup(process):
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)

    @staticmethod
    def send(process, value):
        process.stdin.write(json.dumps(value) + "\n")
        process.stdin.flush()

    def test_jsonl_cancel_during_generation_and_followup(self):
        process = self.start()
        self.send(process, {"op": "generate", "id": "resp_slow", "request": {
            "messages": [{"role": "user", "content": "[fake:slow]"}]}})
        self.assertEqual(json.loads(process.stdout.readline())["type"], "started")
        while json.loads(process.stdout.readline())["type"] != "delta":
            pass
        self.send(process, {"op": "cancel", "id": "resp_slow"})
        while True:
            event = json.loads(process.stdout.readline())
            if event["type"] == "terminal":
                break
        self.assertEqual(event["status"], "cancelled")
        self.send(process, {"op": "generate", "id": "resp_next", "request": {
            "messages": [{"role": "user", "content": "hello"}]}})
        while True:
            event = json.loads(process.stdout.readline())
            if event["type"] == "terminal":
                break
        self.assertEqual(event["status"], "completed")
        self.assertEqual(event["id"], "resp_next")
        self.send(process, {"op": "shutdown"})
        self.assertEqual(process.wait(timeout=5), 0)

    def test_invalid_budget_emits_error_without_started(self):
        process = self.start("8192")
        self.send(process, {"op": "generate", "id": "resp_bad", "request": {
            "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8192}})
        event = json.loads(process.stdout.readline())
        self.assertEqual(event["type"], "terminal")
        self.assertEqual(event["error"]["http_status"], 400)
        self.assertEqual(event["error"]["code"], "context_length_exceeded")

    def test_eof_cancels_active_generation(self):
        process = self.start()
        self.send(process, {"op": "generate", "id": "resp_eof", "request": {
            "messages": [{"role": "user", "content": "[fake:slow]"}]}})
        while json.loads(process.stdout.readline())["type"] != "delta":
            pass
        process.stdin.close()
        process.stdin = None
        events = [json.loads(line) for line in process.stdout]
        self.assertEqual(events[-1]["status"], "cancelled")
        self.assertEqual(process.wait(timeout=5), 0)

    def test_malformed_nested_requests_fail_before_started_then_worker_recovers(self):
        cases = [
            {"messages": [{"role": "user", "content": [1]}]},
            {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "tool_calls": [1]}]},
            {"messages": [{"role": "user", "content": "go"}], "tool_choice": {"type": "function", "function": None}},
            {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "tool_calls": [{"function": None}]}]},
        ]
        for request in cases:
            with self.subTest(request=request):
                process = self.start()
                self.send(process, {"op": "generate", "id": "resp_bad", "request": request})
                event = json.loads(process.stdout.readline())
                self.assertEqual(event["type"], "terminal")
                self.assertEqual(event["status"], "failed")
                self.assertEqual(event["error"]["http_status"], 400)
                self.send(process, {"op": "generate", "id": "resp_ok", "request": {
                    "messages": [{"role": "user", "content": "hello"}]}})
                while True:
                    event = json.loads(process.stdout.readline())
                    if event["type"] == "terminal":
                        break
                self.assertEqual(event["status"], "completed")
