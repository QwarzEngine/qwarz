from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

import pytest


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/debug/qwasar-server"


def request(base, path, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(base + path, data=data,
        headers={"Content-Type": "application/json", **(headers or {})}), timeout=15)


@contextmanager
def server(tmp_path):
    if not BINARY.exists():
        pytest.skip("build qwasar-server with cargo build before HTTP tests")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    python = ROOT / ".venv/bin/python"
    log = (tmp_path / "server.log").open("ab")
    process = subprocess.Popen([str(BINARY), "--port", str(port), "--database", str(tmp_path / "sessions.db"),
        "--python", str(python), "--fake-worker", "--context-size", "1024"], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=log)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        ready = False
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with request(base, "/health") as response:
                    ready = json.load(response).get("worker", {}).get("status") == "ready"
                if ready:
                    break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(.05)
        assert ready, (tmp_path / "server.log").read_text()
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def body(content="hello", **options):
    return {"model": "qwasar-qwen38-27b", "messages": [{"role": "user", "content": content}],
            "max_tokens": 64, **options}


def test_pi_extension_pattern_properties_is_accepted_before_generation(tmp_path):
    tools = [{"type": "function", "function": {"name": "mcporter_call", "parameters": {
        "type": "object", "properties": {"args": {"type": "object", "patternProperties": {"^.*$": {}}}}}}}]
    with server(tmp_path) as base:
        with request(base, "/v1/chat/completions", body("Hola", tools=tools, stream=True)) as response:
            chunks = list(payloads(response))
        assert any(chunk.get("choices") and chunk["choices"][0].get("finish_reason") == "stop" for chunk in chunks)


def payloads(response):
    for line in response:
        if line.startswith(b"data: "):
            data = line[6:].strip()
            if data == b"[DONE]":
                return
            yield json.loads(data)


def test_streaming_models_validation_and_idempotent_replay(tmp_path):
    with server(tmp_path) as base:
        with request(base, "/v1/models") as response:
            assert json.load(response)["data"][0]["id"] == "qwasar-qwen38-27b"
        with request(base, "/v1/chat/completions", body(stream=True)) as response:
            chunks = list(payloads(response))
        assert any(chunk.get("choices", [{}])[0].get("delta", {}).get("content") for chunk in chunks if chunk.get("choices"))
        assert any(chunk.get("choices", [{}])[0].get("finish_reason") == "stop" for chunk in chunks if chunk.get("choices"))
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base, "/v1/chat/completions", body(response_format={"type": "json_object"}))
        assert error.value.code == 400
        headers = {"Idempotency-Key": "same-request"}
        with request(base, "/v1/chat/completions", body(), headers) as response:
            first = json.load(response)
        with request(base, "/v1/chat/completions", body(), headers) as response:
            second = json.load(response)
        assert first["id"] == second["id"]
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base, "/v1/chat/completions", body("different"), headers)
        assert error.value.code == 409


def test_cancel_busy_recovery_and_budget_rejection(tmp_path):
    with server(tmp_path) as base:
        response = request(base, "/v1/chat/completions", body("[fake:slow]", stream=True))
        stream = payloads(response)
        initial = next(stream)
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base, "/v1/chat/completions", body())
        assert error.value.code == 409
        with request(base, f"/v1/responses/{initial['id']}/cancel", {}) as cancelled:
            assert json.load(cancelled)["id"] == initial["id"]
        list(stream)
        response.close()
        with request(base, "/v1/chat/completions", body()) as response:
            assert json.load(response)["choices"][0]["finish_reason"] == "stop"
        with pytest.raises(urllib.error.HTTPError) as error:
            request(base, "/v1/chat/completions", body("hello " * 3000))
        assert error.value.code in (400, 422)


def test_responses_durable_parent_and_restart(tmp_path):
    with server(tmp_path) as base:
        with request(base, "/v1/responses", {"model": "qwasar-qwen38-27b", "input": "hello", "max_output_tokens": 64}) as response:
            first = json.load(response)
        assert first["status"] == "completed"
    with server(tmp_path) as base:
        with request(base, "/v1/responses/" + first["id"]) as response:
            fetched = json.load(response)
            assert fetched == first
            assert fetched["object"] == "response"
        with request(base, "/v1/responses", {"model": "qwasar-qwen38-27b", "input": "next", "previous_response_id": first["id"], "max_output_tokens": 64}) as response:
            assert json.load(response)["status"] == "completed"


def test_responses_stream_declares_items_before_deltas_with_matching_indices(tmp_path):
    with server(tmp_path) as base:
        with request(base, "/v1/responses", {"model": "qwasar-qwen38-27b", "input": "hello",
                     "max_output_tokens": 64, "stream": True}) as response:
            events = list(payloads(response))
        items = {}
        for event in events:
            if event["type"] == "response.output_item.added":
                items[event["output_index"]] = event["item"]["id"]
            if event["type"].endswith(".delta"):
                assert event["output_index"] in items, event
                assert items[event["output_index"]] == event["item_id"]
            if event["type"] == "response.output_item.done":
                assert items[event["output_index"]] == event["item"]["id"]
        assert items
        assert events[-1]["type"] == "response.completed"
        assert [event["sequence_number"] for event in events] == list(range(len(events)))


def test_failed_idempotent_retry_does_not_turn_into_success(tmp_path):
    with server(tmp_path) as base:
        for _ in range(2):
            with pytest.raises(urllib.error.HTTPError) as error:
                request(base, "/v1/chat/completions", body("hello " * 3000),
                        {"Idempotency-Key": "budget-failure"})
            assert error.value.code in (400, 422)


def test_responses_function_call_lifecycle_and_parent_result(tmp_path):
    with server(tmp_path) as base:
        arguments = {"model": "qwasar-qwen38-27b", "input": "[fake:tool]",
                     "max_output_tokens": 256, "stream": True,
                     "tools": [{"type": "function", "name": "read", "parameters": {
                         "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]}
        headers = {"Idempotency-Key": "responses-tool-replay"}
        for _ in range(2):
            with request(base, "/v1/responses", arguments, headers) as response:
                events = list(payloads(response))
            slots = {}
            for event in events:
                if event["type"] == "response.output_item.added":
                    slots[event["output_index"]] = event["item"]
                if event["type"] == "response.function_call_arguments.delta":
                    assert slots[event["output_index"]]["type"] == "function_call"
                    assert slots[event["output_index"]]["id"] == event["item_id"]
                    assert json.loads(event["delta"]) == {"path": "fake"}
            final = events[-1]["response"]
            assert final["status"] == "completed"
            assert [slots[index]["id"] for index in sorted(slots)] == [item["id"] for item in final["output"]]
            call = next(item for item in final["output"] if item["type"] == "function_call")
        with request(base, "/v1/responses", {"model": "qwasar-qwen38-27b", "previous_response_id": final["id"],
                     "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "file content"}],
                     "max_output_tokens": 64}) as response:
            assert json.load(response)["status"] == "completed"


def test_responses_without_reasoning_uses_message_index_zero(tmp_path):
    with server(tmp_path) as base:
        with request(base, "/v1/responses", {"model": "qwasar-qwen38-27b", "input": "hello",
                     "max_output_tokens": 64, "stream": True, "reasoning": {"effort": "off"}}) as response:
            events = list(payloads(response))
        for event in events:
            if "output_index" in event:
                assert event["output_index"] == 0
        assert events[-1]["response"]["output"][0]["type"] == "message"


def test_stream_usage_opt_out_is_preserved_on_idempotent_replay(tmp_path):
    with server(tmp_path) as base:
        for _ in range(2):
            with request(base, "/v1/chat/completions", body(stream=True, stream_options={"include_usage": False}),
                         {"Idempotency-Key": "no-usage"}) as response:
                chunks = list(payloads(response))
            assert all(chunk.get("choices") for chunk in chunks)


@pytest.mark.parametrize("same_port", [True, False])
def test_duplicate_server_cannot_recover_live_owners_database(tmp_path, same_port):
    with server(tmp_path) as base:
        response = request(base, "/v1/chat/completions", body("[fake:slow]", stream=True))
        stream = payloads(response)
        response_id = next(stream)["id"]
        port = int(base.rsplit(":", 1)[1])
        if not same_port:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        duplicate = subprocess.Popen([str(BINARY), "--port", str(port), "--database", str(tmp_path / "sessions.db"),
            "--python", str(ROOT / ".venv/bin/python"), "--fake-worker", "--context-size", "1024"], cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(.1)
            assert duplicate.poll() == 1, "duplicate process must refuse database ownership"
            with request(base, "/v1/responses/" + response_id) as stored:
                assert json.load(stored)["status"] == "in_progress"
        finally:
            if duplicate.poll() is None:
                duplicate.terminate()
            duplicate.wait(timeout=5)
            response.close()
