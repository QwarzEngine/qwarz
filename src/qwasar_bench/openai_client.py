from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator


class ResponseStreamError(RuntimeError):
    """Raised when a Responses API stream cannot produce a completed response."""


@dataclass(frozen=True, slots=True)
class StreamResult:
    response_id: str
    text: str
    input_tokens: int
    output_tokens: int
    request_started_ns: int
    headers_received_ns: int
    first_delta_ns: int | None
    completed_ns: int
    events: tuple[dict[str, Any], ...]


def parse_sse(
    lines: Iterable[bytes], *, max_event_bytes: int = 4 * 1024 * 1024
) -> Iterator[tuple[str, str]]:
    event_name = "message"
    data_lines: list[str] = []
    event_bytes = 0

    def dispatch() -> tuple[str, str] | None:
        nonlocal event_name, data_lines, event_bytes
        if not data_lines:
            event_name = "message"
            event_bytes = 0
            return None
        event = (event_name, "\n".join(data_lines))
        event_name = "message"
        data_lines = []
        event_bytes = 0
        return event

    for raw_line in lines:
        event_bytes += len(raw_line)
        if event_bytes > max_event_bytes:
            raise ResponseStreamError(
                f"SSE event exceeded maximum size of {max_event_bytes} bytes"
            )
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ResponseStreamError(f"SSE stream is not valid UTF-8: {error}") from error
        if not line:
            event = dispatch()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    event = dispatch()
    if event is not None:
        yield event


def _terminal_error(payload: dict[str, Any], event_type: str) -> str:
    response = payload.get("response")
    if isinstance(response, dict):
        error = response.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return event_type


class ResponsesClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 300.0) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def create_stream(
        self,
        *,
        model: str,
        input_items: list[dict[str, Any]],
        max_output_tokens: int,
        previous_response_id: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> StreamResult:
        payload: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": True,
        }
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        request = urllib.request.Request(
            f"{self.base_url}/v1/responses",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request_started_ns = time.monotonic_ns()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                headers_received_ns = time.monotonic_ns()
                return self._consume_stream(
                    response,
                    request_started_ns=request_started_ns,
                    headers_received_ns=headers_received_ns,
                )
        except ResponseStreamError:
            raise
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise ResponseStreamError(f"Responses API HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ResponseStreamError(f"Responses API request failed: {error}") from error

    def _consume_stream(
        self,
        response: Iterable[bytes],
        *,
        request_started_ns: int,
        headers_received_ns: int,
    ) -> StreamResult:
        response_id: str | None = None
        text_parts: list[str] = []
        input_tokens: int | None = None
        output_tokens: int | None = None
        first_delta_ns: int | None = None
        completed_ns: int | None = None
        terminal_seen = False
        raw_events: list[dict[str, Any]] = []

        for sse_event, data in parse_sse(response):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as error:
                raise ResponseStreamError(f"invalid JSON in SSE event {sse_event}: {error}") from error
            if not isinstance(payload, dict):
                raise ResponseStreamError(f"SSE event {sse_event} payload must be an object")
            raw_events.append(payload)
            event_type = payload.get("type", sse_event)
            if not isinstance(event_type, str):
                raise ResponseStreamError("SSE event type must be a string")
            response_object = payload.get("response")
            if isinstance(response_object, dict) and isinstance(response_object.get("id"), str):
                response_id = response_object["id"]
            if event_type == "response.output_text.delta":
                delta = payload.get("delta")
                if not isinstance(delta, str):
                    raise ResponseStreamError("response.output_text.delta is missing a string delta")
                if first_delta_ns is None:
                    first_delta_ns = time.monotonic_ns()
                text_parts.append(delta)
            elif event_type == "response.completed":
                if not isinstance(response_object, dict):
                    raise ResponseStreamError("response.completed is missing its response object")
                usage = response_object.get("usage")
                if not isinstance(usage, dict):
                    raise ResponseStreamError("response.completed is missing usage")
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
                    raise ResponseStreamError("response.completed usage counters must be integers")
                completed_ns = time.monotonic_ns()
                terminal_seen = True
            elif event_type in ("response.failed", "response.cancelled"):
                raise ResponseStreamError(_terminal_error(payload, event_type))

        if not terminal_seen or completed_ns is None:
            raise ResponseStreamError("stream ended without a terminal response.completed event")
        if response_id is None:
            raise ResponseStreamError("completed stream did not provide a response id")
        assert input_tokens is not None
        assert output_tokens is not None
        return StreamResult(
            response_id=response_id,
            text="".join(text_parts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_started_ns=request_started_ns,
            headers_received_ns=headers_received_ns,
            first_delta_ns=first_delta_ns,
            completed_ns=completed_ns,
            events=tuple(raw_events),
        )


@dataclass(frozen=True, slots=True)
class _HealthCounters:
    prompt_tokens_total: int
    completion_tokens_total: int


def _message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponseStreamError("input item content must be a string or content array")
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in ("input_text", "text"):
            raise ResponseStreamError("unsupported Responses content item")
        text = part.get("text")
        if not isinstance(text, str):
            raise ResponseStreamError("Responses text content must contain a string")
        text_parts.append(text)
    return "".join(text_parts)


def _chat_messages(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            raise ResponseStreamError("input items must be objects")
        role = item.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ResponseStreamError(f"unsupported input role: {role}")
        message: dict[str, Any] = {
            "role": role,
            "content": _message_content(item.get("content")),
        }
        if role == "tool":
            tool_call_id = item.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ResponseStreamError("tool messages require tool_call_id")
            message["tool_call_id"] = tool_call_id
        messages.append(message)
    return messages


def _merge_tool_call_delta(
    calls: dict[int, dict[str, Any]], delta_calls: Any
) -> None:
    if not isinstance(delta_calls, list):
        raise ResponseStreamError("chat tool_calls delta must be an array")
    for raw_call in delta_calls:
        if not isinstance(raw_call, dict):
            raise ResponseStreamError("chat tool call delta must be an object")
        index = raw_call.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ResponseStreamError("chat tool call delta requires a non-negative index")
        call = calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        for key in ("id", "type"):
            value = raw_call.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise ResponseStreamError(f"chat tool call {key} must be a string")
                call[key] += value if key == "id" else ""
                if key == "type":
                    call[key] = value
        function = raw_call.get("function")
        if function is not None:
            if not isinstance(function, dict):
                raise ResponseStreamError("chat tool call function must be an object")
            for key in ("name", "arguments"):
                value = function.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ResponseStreamError(f"chat tool function {key} must be a string")
                    call["function"][key] += value


class ChatCompletionsClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 300.0,
        max_history_entries: int = 128,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_history_entries < 1:
            raise ValueError("max_history_entries must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_history_entries = max_history_entries
        self._histories: OrderedDict[str, tuple[dict[str, Any], ...]] = OrderedDict()
        self._request_lock = threading.Lock()

    def _health(self) -> _HealthCounters:
        request = urllib.request.Request(
            f"{self.base_url}/health",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise ResponseStreamError(f"health endpoint HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise ResponseStreamError(f"health endpoint failed: {error}") from error
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ResponseStreamError("health endpoint did not report ok=true")
        prompt_tokens = payload.get("prompt_tokens_total")
        completion_tokens = payload.get("completion_tokens_total")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens < 0
        ):
            raise ResponseStreamError("health endpoint token counters are invalid")
        return _HealthCounters(prompt_tokens, completion_tokens)

    def _parent_messages(self, previous_response_id: str | None) -> list[dict[str, Any]]:
        if previous_response_id is None:
            return []
        try:
            history = self._histories.pop(previous_response_id)
        except KeyError as error:
            raise ResponseStreamError(
                f"unknown previous_response_id: {previous_response_id}"
            ) from error
        self._histories[previous_response_id] = history
        return [dict(message) for message in history]

    def _remember(self, response_id: str, messages: list[dict[str, Any]]) -> None:
        if response_id in self._histories:
            raise ResponseStreamError(f"duplicate chat response id: {response_id}")
        self._histories[response_id] = tuple(dict(message) for message in messages)
        while len(self._histories) > self.max_history_entries:
            self._histories.popitem(last=False)

    def create_stream(
        self,
        *,
        model: str,
        input_items: list[dict[str, Any]],
        max_output_tokens: int,
        previous_response_id: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> StreamResult:
        with self._request_lock:
            messages = self._parent_messages(previous_response_id) + _chat_messages(input_items)
            before = self._health()
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "stream": True,
            }
            request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            request_started_ns = time.monotonic_ns()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    headers_received_ns = time.monotonic_ns()
                    stream = self._consume_chat_stream(
                        response,
                        request_started_ns=request_started_ns,
                        headers_received_ns=headers_received_ns,
                    )
            except ResponseStreamError:
                raise
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace").strip()
                raise ResponseStreamError(
                    f"Chat Completions HTTP {error.code}: {detail}"
                ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                raise ResponseStreamError(f"Chat Completions request failed: {error}") from error
            after = self._health()
            input_tokens = after.prompt_tokens_total - before.prompt_tokens_total
            output_tokens = after.completion_tokens_total - before.completion_tokens_total
            if input_tokens < 0 or output_tokens < 0:
                raise ResponseStreamError("health endpoint token counters moved backwards")
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": stream["text"],
            }
            if stream["reasoning"]:
                assistant["reasoning_content"] = stream["reasoning"]
            if stream["tool_calls"]:
                assistant["tool_calls"] = stream["tool_calls"]
            self._remember(stream["response_id"], messages + [assistant])
            metric_event = {
                "type": "qwasar.metrics",
                "usage_source": "health_counter_delta",
            }
            return StreamResult(
                response_id=stream["response_id"],
                text=stream["text"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_started_ns=request_started_ns,
                headers_received_ns=headers_received_ns,
                first_delta_ns=stream["first_delta_ns"],
                completed_ns=stream["completed_ns"],
                events=tuple(stream["events"] + [metric_event]),
            )

    def _consume_chat_stream(
        self,
        response: Iterable[bytes],
        *,
        request_started_ns: int,
        headers_received_ns: int,
    ) -> dict[str, Any]:
        del request_started_ns, headers_received_ns
        response_id: str | None = None
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        first_delta_ns: int | None = None
        completed_ns: int | None = None
        terminal_seen = False
        raw_events: list[dict[str, Any]] = []
        for sse_event, data in parse_sse(response):
            if data == "[DONE]":
                completed_ns = time.monotonic_ns()
                terminal_seen = True
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as error:
                raise ResponseStreamError(
                    f"invalid JSON in chat SSE event {sse_event}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ResponseStreamError("chat SSE payload must be an object")
            if "error" in payload:
                raise ResponseStreamError(_terminal_error(payload, "chat stream failed"))
            raw_events.append(payload)
            chunk_id = payload.get("id")
            if isinstance(chunk_id, str):
                if response_id is not None and response_id != chunk_id:
                    raise ResponseStreamError("chat stream changed response id")
                response_id = chunk_id
            choices = payload.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ResponseStreamError("chat stream must contain exactly one choice")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ResponseStreamError("chat choice must be an object")
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise ResponseStreamError("chat choice delta must be an object")
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if content is not None:
                if not isinstance(content, str):
                    raise ResponseStreamError("chat content delta must be a string")
                text_parts.append(content)
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise ResponseStreamError("chat reasoning delta must be a string")
                reasoning_parts.append(reasoning)
            if "tool_calls" in delta:
                _merge_tool_call_delta(tool_calls, delta["tool_calls"])
            if first_delta_ns is None and (content or reasoning or "tool_calls" in delta):
                first_delta_ns = time.monotonic_ns()
        if not terminal_seen or completed_ns is None:
            raise ResponseStreamError("chat stream ended without [DONE]")
        if response_id is None:
            raise ResponseStreamError("chat stream did not provide a response id")
        return {
            "response_id": response_id,
            "text": "".join(text_parts),
            "reasoning": "".join(reasoning_parts),
            "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
            "first_delta_ns": first_delta_ns,
            "completed_ns": completed_ns,
            "events": raw_events,
        }
