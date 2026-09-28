"""One resident generation.

The Gated DeltaNet state and the NVFP4 pages stay on the device. A second
turn forwards only the suffix after the committed cursor. Cancel restores
that commit. A second generate while one is open is refused.
"""
from __future__ import annotations

from engine.forward.schedule import suffix_bounds


class SessionBusy(Exception):
    code = "runtime_busy"
    http_status = 409

    def __init__(self):
        super().__init__("one generation is already active")


class Session:
    def __init__(self, runner, context=262144):
        self.runner = runner
        self.context = context
        self.tape = []
        self.cursor = 0
        self.committed = 0
        self.snapshot = None
        self.busy = False

    def _rewind_to_commit_or_zero(self, shared):
        if shared >= self.cursor:
            return
        if self.snapshot is not None and shared >= self.committed:
            self.runner.restore(self.snapshot)
            self.cursor = self.committed
            self.tape = self.tape[:self.committed]
            return
        self.runner.reset()
        self.snapshot = None
        self.tape = []
        self.cursor = 0
        self.committed = 0

    def prefill(self, ids):
        ids = list(ids)
        if len(ids) > self.context:
            raise ValueError("prompt exceeds the resident context")
        shared = 0
        while shared < len(self.tape) and shared < len(ids) and self.tape[shared] == ids[shared]:
            shared += 1
        self._rewind_to_commit_or_zero(shared)
        if ids[:self.cursor] != self.tape[:self.cursor]:
            raise RuntimeError("resident tape does not match the prompt prefix")
        forwarded = []
        for start, end in suffix_bounds(self.cursor, len(ids)):
            self.runner.forward(ids[start:end], start)
            self.cursor = end
            self.tape = ids[:end]
            forwarded.append((start, end))
        return forwarded

    def turn(self, ids, cancelled=None):
        if self.busy:
            raise SessionBusy()
        self.busy = True
        ids = list(ids)
        forwarded = []
        try:
            shared = 0
            while shared < len(self.tape) and shared < len(ids) and self.tape[shared] == ids[shared]:
                shared += 1
            self._rewind_to_commit_or_zero(shared)
            if ids[:self.cursor] != self.tape[:self.cursor]:
                raise RuntimeError("resident tape does not match the prompt prefix")
            cached_tokens = self.cursor
            spans = list(suffix_bounds(self.cursor, len(ids)))
            if self.cursor < len(ids) and (not spans or spans[-1][1] < len(ids)):
                spans.append((len(ids) - 1, len(ids)))
            for start, end in spans:
                if cancelled is not None and cancelled():
                    self.cancel()
                    return {"status": "cancelled", "forwarded": forwarded, "cached_tokens": self.committed}
                self.runner.forward(ids[start:end], start)
                self.cursor = end
                self.tape = ids[:end]
                forwarded.append((start, end))
            self.commit()
            return {
                "status": "completed",
                "forwarded": forwarded,
                "cached_tokens": cached_tokens,
                "cursor": self.cursor,
            }
        except SessionBusy:
            raise
        except Exception:
            self.cancel()
            raise

    def commit(self):
        self.committed = self.cursor
        self.tape = self.tape[:self.cursor]
        self.snapshot = self.runner.capture()
        self.busy = False

    def cancel(self):
        if self.snapshot is None:
            self.runner.reset()
            self.tape = []
            self.cursor = 0
            self.committed = 0
        else:
            self.runner.restore(self.snapshot)
            self.cursor = self.committed
            self.tape = self.tape[:self.committed]
        self.busy = False


class MemoryRunner:
    """Records forwards and keeps an integer cursor a snapshot can restore."""

    def __init__(self):
        self.calls = []
        self.state = 0

    def forward(self, token_ids, cache_len):
        if cache_len != self.state:
            raise RuntimeError(f"forward at {cache_len} with resident cursor {self.state}")
        self.calls.append((cache_len, list(token_ids)))
        self.state += len(token_ids)

    def capture(self):
        return self.state

    def restore(self, snapshot):
        self.state = snapshot

    def reset(self):
        self.state = 0
