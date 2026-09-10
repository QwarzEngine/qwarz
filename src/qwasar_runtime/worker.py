from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import sys
import threading
import traceback

from .engine import Engine, ExLlamaBackend, FakeBackend, FatalRuntimeError, RequestError, terminal_error
from .parsing import parse_json


MAX_LINE_BYTES = 32 * 1024 * 1024


def serve(engine, incoming, outgoing):
    jobs = queue.Queue(maxsize=1)
    active = {}
    state_lock = threading.Lock()
    output_lock = threading.Lock()

    def emit(event):
        with output_lock:
            outgoing.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            outgoing.flush()

    def read_input():
        try:
            while True:
                line = incoming.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                response_id = ""
                try:
                    if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                        raise RequestError("worker protocol line exceeds 32 MiB")
                    payload = parse_json(line)
                    if not isinstance(payload, dict):
                        raise RequestError("protocol command must be an object")
                    response_id = payload.get("id", "")
                    if not isinstance(response_id, str):
                        response_id = ""
                        raise RequestError("request id must be a string")
                    operation = payload.get("op")
                    if operation == "shutdown":
                        with state_lock:
                            for cancellation in active.values():
                                cancellation.set()
                        break
                    if operation == "cancel":
                        with state_lock:
                            if response_id in active:
                                active[response_id].set()
                        continue
                    if operation != "generate" or not response_id or len(response_id) > 256:
                        raise RequestError("generate needs a nonempty id of at most 256 characters")
                    with state_lock:
                        if active:
                            raise RequestError("one generation is already active", "runtime_busy", 409)
                        cancellation = threading.Event()
                        active[response_id] = cancellation
                        jobs.put_nowait((response_id, payload.get("request"), payload.get("parent"), cancellation))
                except (ValueError, TypeError, KeyError) as error:
                    emit(terminal_error(response_id, error))
        finally:
            with state_lock:
                for cancellation in active.values():
                    cancellation.set()
            jobs.put(None)

    emit({"type": "ready", "protocol": 1, "config": engine.config})
    reader = threading.Thread(target=read_input, name="qwasar-jsonl-input", daemon=True)
    reader.start()
    while True:
        job = jobs.get()
        if job is None:
            return
        response_id, request, parent, cancellation = job
        try:
            for event in engine.generate(response_id, request, parent, cancellation):
                if event["type"] == "terminal":
                    with state_lock:
                        active.pop(response_id, None)
                emit(event)
        finally:
            with state_lock:
                active.pop(response_id, None)


def main():
    parser = argparse.ArgumentParser(description="Qwasar version 1 JSONL worker")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--context-size", type=int, default=262144)
    parser.add_argument("--prefill", choices=("baseline", "flash"), default="flash")
    parser.add_argument("--gpu-split-gb", type=float, default=30.0)
    parser.add_argument("--fake", action="store_true", help="Explicit deterministic CPU fixture runtime; never loads CUDA")
    args = parser.parse_args()
    if not args.fake and args.model is None:
        parser.error("--model is required unless --fake is explicitly selected")
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        backend = FakeBackend() if args.fake else ExLlamaBackend(args.model, args.context_size, args.prefill, args.gpu_split_gb)
        diagnostic_directory = os.environ.get("QWASAR_TOOL_DIAGNOSTICS_DIR", "state/tool-errors")
        engine = Engine(backend, args.context_size,
                        diagnostic_directory=diagnostic_directory if diagnostic_directory and not args.fake else None)
        serve(engine, sys.stdin, protocol)
    except (BrokenPipeError, KeyboardInterrupt):
        return 1
    except FatalRuntimeError:
        traceback.print_exc(file=sys.stderr)
        return 70
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        protocol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
