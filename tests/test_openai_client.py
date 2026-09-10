from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from qwasar_bench.openai_client import ResponseStreamError, ResponsesClient, parse_sse


@contextmanager
def responses_server(events: list[dict[str, object] | str]) -> Iterator[tuple[str, list[dict[str, object]]]]:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(content_length)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in events:
                if isinstance(event, str):
                    body = event
                else:
                    event_type = str(event["type"])
                    body = f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
                self.wfile.write(body.encode("utf-8"))
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


def test_stream_response_collects_usage_and_first_delta() -> None:
    events: list[dict[str, object] | str] = [
        {"type": "response.created", "response": {"id": "resp_123"}},
        {"type": "response.output_text.delta", "delta": "hello "},
        {"type": "response.output_text.delta", "delta": "world"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        },
        "data: [DONE]\n\n",
    ]
    with responses_server(events) as (base_url, requests):
        client = ResponsesClient(base_url)
        result = client.create_stream(
            model="qwen38",
            input_items=[{"role": "user", "content": "hello"}],
            max_output_tokens=16,
            previous_response_id="resp_parent",
        )

    assert result.response_id == "resp_123"
    assert result.text == "hello world"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.first_delta_ns is not None
    assert result.first_delta_ns >= result.request_started_ns
    assert result.completed_ns >= result.first_delta_ns
    assert requests[0]["previous_response_id"] == "resp_parent"
    assert requests[0]["stream"] is True


def test_parse_sse_supports_multiline_data_and_comments() -> None:
    lines = iter(
        [
            b": heartbeat\n",
            b"event: custom\n",
            b'data: {"part":\n',
            b'data: "value"}\n',
            b"\n",
        ]
    )

    assert list(parse_sse(lines)) == [
        ("custom", '{"part":\n"value"}'),
    ]


def test_failed_terminal_event_raises() -> None:
    events: list[dict[str, object] | str] = [
        {"type": "response.created", "response": {"id": "resp_bad"}},
        {
            "type": "response.failed",
            "response": {"id": "resp_bad", "error": {"message": "worker crashed"}},
        },
    ]
    with responses_server(events) as (base_url, _):
        with pytest.raises(ResponseStreamError, match="worker crashed"):
            ResponsesClient(base_url).create_stream(
                model="qwen38",
                input_items=[{"role": "user", "content": "hello"}],
                max_output_tokens=16,
            )
