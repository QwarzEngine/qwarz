from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from qwasar_bench.openai_client import ChatCompletionsClient, ResponseStreamError


@contextmanager
def chat_server() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []
    counters = {"prompt_tokens_total": 0, "completion_tokens_total": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/health"
            body = json.dumps({"ok": True, **counters}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            assert self.path == "/v1/chat/completions"
            content_length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(content_length))
            requests.append(request)
            response_index = len(requests)
            response_id = f"chatcmpl_{response_index}"
            prompt_delta = 5 if response_index == 1 else 9
            counters["prompt_tokens_total"] += prompt_delta
            counters["completion_tokens_total"] += 2

            chunks = [
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": f"think {response_index}"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"answer {response_index}"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_chat_client_reconstructs_persistent_history_and_health_usage() -> None:
    with chat_server() as (base_url, requests):
        client = ChatCompletionsClient(base_url)
        first = client.create_stream(
            model="qwen38",
            input_items=[{"role": "user", "content": "turn one"}],
            max_output_tokens=16,
        )
        second = client.create_stream(
            model="qwen38",
            input_items=[{"role": "user", "content": "turn two"}],
            max_output_tokens=16,
            previous_response_id=first.response_id,
        )

    assert first.response_id == "chatcmpl_1"
    assert first.text == "answer 1"
    assert first.input_tokens == 5
    assert first.output_tokens == 2
    assert second.input_tokens == 9
    assert second.output_tokens == 2
    assert second.first_delta_ns is not None
    assert requests[1]["messages"] == [
        {"role": "user", "content": "turn one"},
        {
            "role": "assistant",
            "content": "answer 1",
            "reasoning_content": "think 1",
        },
        {"role": "user", "content": "turn two"},
    ]


def test_chat_client_rejects_unknown_parent_before_request() -> None:
    with chat_server() as (base_url, requests):
        client = ChatCompletionsClient(base_url)
        with pytest.raises(ResponseStreamError, match="unknown previous_response_id"):
            client.create_stream(
                model="qwen38",
                input_items=[{"role": "user", "content": "turn"}],
                max_output_tokens=16,
                previous_response_id="missing",
            )

    assert requests == []
