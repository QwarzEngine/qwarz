"""One resident generation.

The Gated DeltaNet state and the NVFP4 pages stay on the device. A second
turn forwards only the suffix after the committed cursor. Cancel restores
that commit. A second generate while one is open is refused.

Vision rows and the MRoPE table align with the full prompt. Each span
receives its own rows. The frequency table stays whole so a later span
indexes it at the resident cursor. Both are omitted on a text turn, and
the runner is then called with the two arguments it already had.

generate() emits MTP6 tokens from that same resident cache. The prompt
prefix uses the production cuts and leaves the last prompt id held out.
"""
from __future__ import annotations

from engine.forward.schedule import suffix_bounds


def _forward_span(runner, ids, start, end, embedded, inv_freq):
    span = ids[start:end]
    if embedded is None and inv_freq is None:
        runner.forward(span, start)
        return
    kwargs = {}
    if embedded is not None:
        kwargs["embedded"] = embedded[:, start:end] if embedded.dim() == 3 else embedded[start:end]
    if inv_freq is not None:
        kwargs["inv_freq"] = inv_freq
    runner.forward(span, start, **kwargs)


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
        self.cycle = None

    def _rewind_to_commit_or_zero(self, shared):
        if shared >= self.cursor:
            return
        if self.snapshot is not None and shared >= self.committed:
            self.runner.restore(self.snapshot)
            self.cursor = self.committed
            self.tape = self.tape[:self.committed]
            if self.cycle is not None and (
                self.cycle.draft is None or self.cycle.draft.position != self.cursor
            ):
                self.cycle = None
            return
        self.runner.reset()
        self.snapshot = None
        self.tape = []
        self.cursor = 0
        self.committed = 0
        self.cycle = None

    def prefill(self, ids, embedded=None, inv_freq=None):
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
            _forward_span(self.runner, ids, start, end, embedded, inv_freq)
            self.cursor = end
            self.tape = ids[:end]
            forwarded.append((start, end))
        return forwarded

    def generate(self, ids, max_new):
        """Emit greedy MTP6 tokens. The last prompt id stays out of the prefill."""
        if self.busy:
            raise SessionBusy()
        if type(max_new) is not int or max_new < 1:
            raise ValueError("generate needs a positive max_new")
        self.busy = True
        ids = list(ids)
        try:
            if len(ids) > self.context:
                raise ValueError("prompt exceeds the resident context")
            shared = 0
            while shared < len(self.tape) and shared < len(ids) and self.tape[shared] == ids[shared]:
                shared += 1
            self._rewind_to_commit_or_zero(shared)
            if self.cycle is not None and self.cycle.draft is not None and self.cycle.draft.position != self.cursor:
                self.cycle = None
            if self.cycle is None and self.cursor:
                self.runner.reset()
                self.snapshot = None
                self.tape = []
                self.cursor = 0
                self.committed = 0
            if ids[:self.cursor] != self.tape[:self.cursor]:
                raise RuntimeError("resident tape does not match the prompt prefix")
            cached_tokens = self.cursor
            from engine.forward.cycle import DraftCycle, continue_drafted

            if self.cycle is None:
                self.cycle = DraftCycle(self.runner)
            tokens, self.cycle = continue_drafted(
                self.runner, ids, max_new, cycle=self.cycle, start=self.cursor,
            )
            if self.cycle.cache_len != len(ids) + len(self.cycle.tokens) - 1:
                raise RuntimeError("drafted cache diverged from the emitted tokens")
            self.tape = (ids + list(self.cycle.tokens))[:self.cycle.cache_len]
            self.cursor = self.cycle.cache_len
            self.commit()
            return {
                "status": "completed",
                "tokens": list(tokens),
                "cached_tokens": cached_tokens,
                "cursor": self.cursor,
            }
        except SessionBusy:
            raise
        except Exception:
            self.cancel()
            raise

    def turn(self, ids, cancelled=None, embedded=None, inv_freq=None):
        if self.busy:
            raise SessionBusy()
        self.cycle = None
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
                _forward_span(self.runner, ids, start, end, embedded, inv_freq)
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
            self.cycle = None
        else:
            self.runner.restore(self.snapshot)
            self.cursor = self.committed
            self.tape = self.tape[:self.committed]
            if self.cycle is not None and (
                self.cycle.draft is None or self.cycle.draft.position != self.cursor
            ):
                self.cycle = None
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
