"""JSONL worker for the resident session.

This is the process qwarz starts. One generate is active; a second returns
409. Cancel restores the last commit. A generate of raw ids returns the
greedy token ids from the resident MTP6 windows. A chat completion, the
shape Droid posts, is rendered with the artifact tokenizer and returned as
text and tool calls. promotion_allowed stays false.
"""
from __future__ import annotations

import json
import queue
import threading

from engine.forward.gate import promotion_allowed
from engine.forward.session import SessionBusy
from engine.forward.vision import MM_TOKEN_BASE, canonical_tape

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
    def __init__(self, session, tokenizer=None):
        self.session = session
        self.tokenizer = tokenizer
        self.active = None
        self._cancel = None
        self._pending_cancel = None
        self._lock = threading.Lock()
        self._images = {}

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

    def push(self, line, emit=None):
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
            self.request_stop()
            return [{"stop": True}]
        if operation == "cancel":
            self.note_cancel(response_id)
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
        if not isinstance(request, dict):
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "invalid_request", "message": "generate request needs an ids list", "http_status": 400,
            })]
        if not isinstance(request.get("ids"), list):
            if isinstance(request.get("messages"), list):
                return self._chat(response_id, request, payload.get("parent"), emit)
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "invalid_request", "message": "generate request needs an ids list", "http_status": 400,
            })]
        max_new = request.get("max_new", 1)
        if type(max_new) is not int or max_new < 1:
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "invalid_request", "message": "generate needs a positive max_new", "http_status": 400,
            })]
        self._begin(response_id)
        try:
            try:
                result = self.session.generate(request["ids"], max_new, cancel=self._cancel)
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
            message = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
            event = _terminal(
                response_id,
                result["status"],
                message,
                {"prompt_tokens": prompt, "completion_tokens": len(tokens), "total_tokens": prompt + len(tokens),
                 "prompt_tokens_details": {"cached_tokens": cached}},
            )
            event["token_ids"] = tokens
            if result["status"] == "completed":
                prior = request.get("messages")
                event["snapshot"] = {
                    "version": 1,
                    "messages": [*(prior if isinstance(prior, list) else []), message],
                    "tape": list(self.session.tape),
                }
            return [event]
        finally:
            self._end()

    def _begin(self, response_id):
        with self._lock:
            self._cancel = threading.Event()
            if self._pending_cancel == response_id:
                self._cancel.set()
                self._pending_cancel = None
            self.active = response_id

    def _end(self):
        with self._lock:
            self.active = None
            self._cancel = None

    def note_cancel(self, response_id):
        with self._lock:
            if self.active == response_id and self._cancel is not None:
                self._cancel.set()
            else:
                self._pending_cancel = response_id

    def request_stop(self):
        with self._lock:
            if self._cancel is not None:
                self._cancel.set()

    def _embed(self, sha, _media_type, raw):
        cached = self._images.get(sha)
        if cached is not None:
            return cached
        if getattr(self.session.runner, "_table", None) is not None:
            record = self._embed_gpu(sha, raw)
        else:
            record = self._embed_cpu(sha, raw)
        self._images[sha] = record
        return record

    def _embed_cpu(self, sha, raw):
        from qwasar_runtime.vision import fake_tokens, max_pixels_setting, png_size

        width, height = png_size(raw)
        special = getattr(self.tokenizer, "special", None) or {}
        start = int(special.get("<|vision_start|>", 248053))
        end = int(special.get("<|vision_end|>", 248054))
        tokens = list(fake_tokens(sha, width, height, max_pixels_setting(), start, end))
        return {
            "tokens": tokens,
            "dynamic_ids": [token for token in tokens if token >= MM_TOKEN_BASE],
            "sha256": sha,
        }

    def _embed_gpu(self, sha, raw):
        import io

        from PIL import Image

        from engine.forward.vision_tower import embed_image

        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 64_000_000
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except Exception as error:
            raise ValueError(f"image {sha} could not be decoded: {error}") from error
        finally:
            Image.MAX_IMAGE_PIXELS = previous
        if getattr(image, "is_animated", False):
            image.seek(0)
        record = dict(embed_image(self.session.runner.model_dir, image))
        record["sha256"] = sha
        return record

    def _vision_hooks(self, prepared):
        records = prepared.get("images") or []
        if not records:
            return None, None
        runner = self.session.runner
        if getattr(runner, "_table", None) is None:
            def embed_ids(ids):
                if not any(int(token) >= MM_TOKEN_BASE for token in ids):
                    return None
                return [
                    ("vision", int(token)) if int(token) >= MM_TOKEN_BASE else ("text", int(token))
                    for token in ids
                ]

            return embed_ids, "mrope"
        import torch

        from engine.forward.vision_tower import mix_rows, mrope_table

        rows = torch.cat([record["rows"] for record in records])
        dynamic = []
        for record in records:
            dynamic.extend(record["dynamic_ids"])
        freqs = mrope_table(
            prepared["tokens"], records, len(prepared["tokens"]) + prepared["max_new"] + 16,
        )

        def embed_ids(ids):
            if not any(int(token) >= MM_TOKEN_BASE for token in ids):
                return None
            return mix_rows(runner._table, list(ids), rows, dynamic)

        return embed_ids, freqs

    def _chat(self, response_id, request, parent, emit):
        from engine.forward.chat import prepare_chat, run_turn

        if self.tokenizer is None:
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "invalid_request", "message": "chat rendering is not available", "http_status": 400,
            })]
        try:
            prepared = prepare_chat(
                self.tokenizer, request, parent, self.session.context, embedder=self._embed,
            )
        except (ValueError, TypeError, KeyError, IndexError) as error:
            return [_terminal(response_id, "failed", {}, {}, {
                "code": "invalid_request", "message": str(error), "http_status": 400,
            })]
        events = []

        def publish(event):
            if emit is not None:
                emit(event)
            else:
                events.append(event)

        self._begin(response_id)
        try:
            publish({"type": "started", "id": response_id, "prompt_tokens": len(prepared["tokens"])})
            try:
                embed_ids, inv_freq = self._vision_hooks(prepared)
                turn = run_turn(
                    self.tokenizer, self.session, prepared, response_id, publish,
                    cancel=self._cancel, embed_ids=embed_ids, inv_freq=inv_freq,
                )
            except SessionBusy as error:
                return [_terminal(response_id, "failed", {}, {}, {
                    "code": error.code, "message": str(error), "http_status": error.http_status,
                })]
            except ValueError as error:
                return [_terminal(response_id, "failed", {}, {}, {
                    "code": "invalid_request", "message": str(error), "http_status": 400,
                })]
            message = turn["message"]
            prompt = len(prepared["tokens"])
            produced = list(turn["tokens"])
            cached = turn["cached_tokens"]
            event = _terminal(
                response_id,
                turn["status"],
                message,
                {"prompt_tokens": prompt, "completion_tokens": len(produced),
                 "total_tokens": prompt + len(produced),
                 "prompt_tokens_details": {"cached_tokens": cached}},
            )
            event["metrics"] = turn["metrics"]
            event["token_ids"] = produced
            if turn["status"] == "completed":
                segment = {
                    "message": message,
                    "tokens": list(prepared["tokens"][prepared["assistant_start"]:]) + produced,
                }
                event["snapshot"] = {
                    "version": 1,
                    "header": prepared["header"],
                    "messages": [*prepared["messages"], message],
                    "tape": canonical_tape(self.session.tape),
                    "segments": [*prepared["segments"], segment],
                }
            if emit is not None:
                return [event]
            return [*events, event]
        finally:
            self._end()


class _CpuDraft:
    """Draft stand-in for `--fake`. It never loads weights."""

    def __init__(self):
        self.position = 0
        self._next = 10

    def prefill(self, token_ids, _hidden):
        self.position += len(list(token_ids))

    def step(self, _token, _state):
        drafted = self._next
        self._next += 1
        self.position += 1
        return drafted, ("step", drafted)

    def truncate(self, position):
        if position < 0 or position > self.position:
            raise RuntimeError(f"draft cache cannot rewind to {position}")
        self.position = position


class _CpuRunner:
    """One MTP window on the CPU. The first sample mismatches the draft."""

    def __init__(self):
        self.calls = []
        self.state = 0
        self.hidden = None
        self.samples = [42, 0, 0, 0, 0, 0, 0]
        self.draft = _CpuDraft()

    def forward(self, token_ids, cache_len, embedded=None, inv_freq=None):
        if cache_len != self.state:
            raise RuntimeError(f"forward at {cache_len} with resident cursor {self.state}")
        self.calls.append((cache_len, list(token_ids)))
        self.state += len(list(token_ids))
        self.hidden = [("row", cache_len + index) for index in range(len(token_ids))]

    def score_window(self):
        return list(self.samples)

    def capture(self):
        return self.state

    def restore(self, snapshot):
        self.state = snapshot

    def reset(self):
        self.state = 0
        self.hidden = None


def _cpu_session(context):
    from engine.forward.cycle import DraftCycle
    from engine.forward.session import Session

    runner = _CpuRunner()
    session = Session(runner, context=context)
    session.cycle = DraftCycle(runner, draft=runner.draft)
    return session


def _load_tokenizer(model):
    from exllamav3.model.config import Config
    from exllamav3.tokenizer import Tokenizer
    from qwasar_runtime.rendering import render

    tokenizer = Tokenizer.from_config(Config.from_directory(model))
    render(tokenizer, [{"role": "user", "content": "ok"}], [], "off", "none", [])
    return tokenizer


def _gpu_session(model, context):
    from engine.forward.resident import ModelRunner
    from engine.forward.session import Session

    pages = max(1, context // 256)
    runner = ModelRunner(pages=pages, model_dir=model, retain=True)
    runner._ensure()
    return Session(runner, context=context), _load_tokenizer(model)


def _serve(worker, incoming, outgoing):
    jobs = queue.Queue()
    write_lock = threading.Lock()

    def emit(event):
        if event.get("stop"):
            return
        with write_lock:
            outgoing.write(json.dumps(event, ensure_ascii=False) + "\n")
            outgoing.flush()

    def read_input():
        try:
            while True:
                line = incoming.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                payload = None
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict) and payload.get("op") == "cancel":
                    response_id = payload.get("id", "")
                    if isinstance(response_id, str):
                        worker.note_cancel(response_id)
                    continue
                if isinstance(payload, dict) and payload.get("op") == "shutdown":
                    worker.request_stop()
                    break
                jobs.put(line)
        finally:
            worker.request_stop()
            jobs.put(None)

    reader = threading.Thread(target=read_input, name="qwasar-jsonl-input", daemon=True)
    reader.start()
    with write_lock:
        outgoing.write(json.dumps(worker.ready(), ensure_ascii=False) + "\n")
        outgoing.flush()
    while True:
        line = jobs.get()
        if line is None:
            return 0
        for event in worker.push(line, emit=emit):
            emit(event)


def main(argv=None):
    import argparse
    import os
    import sys
    import traceback

    parser = argparse.ArgumentParser(description="Resident Qwarz JSONL worker")
    parser.add_argument("--model")
    parser.add_argument("--context-size", type=int, default=262144)
    parser.add_argument("--prefill", choices=("baseline", "flash", "xqa"), default="xqa")
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args(argv)
    if not args.fake and not args.model:
        parser.error("--model is required unless --fake is set")
    if args.context_size <= 0 or args.context_size > 262144 or args.context_size % 256 != 0:
        parser.error("context size must be a positive multiple of 256 up to 262144")
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        if args.fake:
            session, tokenizer = _cpu_session(args.context_size), None
        else:
            session, tokenizer = _gpu_session(args.model, args.context_size)
        print("qwarz resident engine ready", file=sys.stderr)
        return _serve(Worker(session, tokenizer), sys.stdin, protocol)
    except BrokenPipeError:
        return 1
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        protocol.close()


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
