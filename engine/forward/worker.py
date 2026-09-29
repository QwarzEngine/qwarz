"""JSONL worker for the resident session.

This is the engine the supervisor can attach later. q38-worker still answers
engine_not_linked, and qwasar.service is left on ExLlama. One generate is
active; a second returns 409. Cancel restores the last commit. A generate
returns the greedy token ids from the resident MTP6 windows.
"""
from __future__ import annotations

import json

from engine.forward.gate import promotion_allowed
from engine.forward.session import SessionBusy

MAX_LINE_BYTES = 32 * 1024 * 1024


def _terminal(response_id, status, message, usage, error=None):
    return {
        "type": "terminal",
        "id": response_id,
        "status": status,
        "message": message,
        "usage": usage,
        "metrics": {},
        "snapshot": None,
        "error": error,
    }


class Worker:
    def __init__(self, session):
        self.session = session
        self.active = None

    def ready(self):
        return {
            "type": "ready",
            "protocol": 1,
            "config": {
                "model": "qwasar-qwen38-27b",
                "context_size": self.session.context,
                "vision": True,
                "streams": 1,
                "engine": "q38",
                "linked": False,
                "draft_method": "mtp",
                "draft_tokens": 6,
                "promotion_allowed": False,
            },
        }

    def push(self, line):
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            return [{"stop": True}, _terminal("", "failed", {}, {}, {
                "code": "invalid_request", "message": "worker protocol line exceeds 32 MiB", "http_status": 400,
            })]
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return [_terminal("", "failed", {}, {}, {
                "code": "invalid_request", "message": "protocol command is not JSON", "http_status": 400,
            })]
        if not isinstance(payload, dict):
            return [_terminal("", "failed", {}, {}, {
                "code": "invalid_request", "message": "protocol command must be an object", "http_status": 400,
            })]
        response_id = payload.get("id", "")
        if not isinstance(response_id, str):
            return [_terminal("", "failed", {}, {}, {
                "code": "invalid_request", "message": "request id must be a string", "http_status": 400,
            })]
        operation = payload.get("op")
        if operation == "shutdown":
            self.active = None
            return [{"stop": True}]
        if operation == "cancel":
            if self.active == response_id:
                self.session.cancel()
                self.active = None
            return []
        if operation == "promote":
            allowed = promotion_allowed(payload.get("report"))
            return [{"type": "promote", "allowed": allowed}]
        if operation != "generate" or not response_id or len(response_id) > 256:
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "invalid_request",
                "message": "generate needs a nonempty id of at most 256 characters",
                "http_status": 400,
            })]
        if self.active is not None:
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "runtime_busy", "message": "one generation is already active", "http_status": 409,
            })]
        request = payload.get("request")
        if not isinstance(request, dict) or not isinstance(request.get("ids"), list):
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "invalid_request", "message": "generate request needs an ids list", "http_status": 400,
            })]
        max_new = request.get("max_new", 1)
        if type(max_new) is not int or max_new < 1:
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "invalid_request", "message": "generate needs a positive max_new", "http_status": 400,
            })]
        self.active = response_id
        try:
            try:
                result = self.session.generate(request["ids"], max_new)
            except SessionBusy as error:
                return [_terminal(response_id, "failed", {}, {}, {
                    "code": error.code, "message": str(error), "http_status": error.http_status,
                })]
            except ValueError as error:
                return [_terminal(response_id, "failed", {}, {}, {
                    "code": "invalid_request", "message": str(error), "http_status": 400,
                })]
            cached = result["cached_tokens"]
            tokens = list(result["tokens"])
            prompt = len(request["ids"])
            event = _terminal(
                response_id,
                result["status"],
                {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []},
                {"prompt_tokens": prompt, "completion_tokens": len(tokens), "total_tokens": prompt + len(tokens),
                 "prompt_tokens_details": {"cached_tokens": cached}},
            )
            event["token_ids"] = tokens
            return [event]
        finally:
            self.active = None
