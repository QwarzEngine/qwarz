"""One MTP6 window on the resident target.

The target cache keeps every token except the one about to be scored. The
prompt prefix uses the production page cuts. Each cut exports its post-norm
state, and that cut's last row is the hidden paired with the first token of
the next cut. The draft proposes six ids from the held token. One target
forward scores the held token and the six drafts. The matching prefix is
written again and the first mismatch, or the bonus token when all six match,
stays held for the next window.
"""
from __future__ import annotations

from engine.forward.mtp import DRAFT_TOKENS, acceptance, emitted_tokens


def _is_tensor(value):
    return callable(getattr(value, "dim", None))


def shift_rows(carry, state):
    """Pair each prefill token with the previous token's post-norm row.

    A missing carry is a blank row, so the first prompt token has no previous
    state. The last row of this cut is the carry into the next cut.
    """
    if _is_tensor(state):
        import torch

        fp16 = state if state.dtype == torch.float16 else state.to(dtype=torch.float16)
        blank = torch.zeros_like(fp16[:, :1]) if carry is None else carry
        return torch.cat((blank, fp16[:, :-1]), dim=1), fp16[:, -1:].contiguous()
    rows = list(state)
    if not rows:
        raise ValueError("each prefill cut exports a post-norm state")
    return [None if carry is None else carry, *rows[:-1]], rows[-1]


def _snapshot(hidden):
    if _is_tensor(hidden):
        return hidden.detach().clone()
    return list(hidden)


def _greedy_samples(runner, width):
    score = getattr(runner, "score_window", None)
    if score is None:
        from engine.forward.token import logits

        samples = [int(token) for token in logits(runner._model, runner.last).argmax(dim=-1).view(-1).tolist()]
    else:
        samples = [int(token) for token in score()]
    if len(samples) != width:
        raise RuntimeError(f"the target returned {len(samples)} samples for a window of {width}")
    return samples


class DraftCycle:
    def __init__(self, runner, draft=None):
        self.runner = runner
        self.draft = draft
        self.carry = None
        self.primed = 0
        self.accepted = 0
        self.rejected = 0

    def _model(self):
        if self.draft is None:
            self.runner._ensure()
            from engine.forward.draft import MTPDraft

            self.draft = MTPDraft(self.runner._model, self.runner._table)
        return self.draft

    def absorb_prefill(self, token_ids, hidden):
        draft = self._model()
        paired, self.carry = shift_rows(self.carry, hidden)
        draft.prefill(list(token_ids), paired)

    def propose(self, token_id, steps=DRAFT_TOKENS):
        draft = self._model()
        if self.carry is None:
            raise RuntimeError("draft has no target state")
        self.primed = draft.position
        token = int(token_id)
        state = self.carry
        drafted = []
        for _ in range(steps):
            token, state = draft.step(token, state)
            drafted.append(int(token))
        return drafted

    def repair(self, verify_hidden, drafted, matched):
        draft = self._model()
        draft.truncate(self.primed + 1)
        if _is_tensor(verify_hidden):
            import torch

            state = verify_hidden if verify_hidden.dtype == torch.float16 else verify_hidden.to(dtype=torch.float16)
            if matched:
                draft.prefill(list(drafted[:matched]), state[:, :matched])
            self.carry = state[:, matched:matched + 1].contiguous()
        else:
            rows = list(verify_hidden)
            if matched:
                draft.prefill(list(drafted[:matched]), rows[:matched])
            self.carry = rows[matched]
        self.accepted += matched
        self.rejected += len(drafted) - matched

    def rate(self):
        return acceptance(self.accepted, self.rejected)


def prefill_held(runner, cycle, prompt, start=0):
    """Forward a prompt prefix on the production cuts and absorb each post-norm state.

    The last prompt id is not forwarded. It stays held for the first MTP window.
    A later call starts at the resident cursor and carries the previous row in.
    """
    from engine.forward.schedule import suffix_bounds

    prompt = list(prompt)
    spans = suffix_bounds(start, len(prompt))
    if not spans:
        if len(prompt) >= 2 and start == len(prompt) - 1:
            if cycle.draft is not None and cycle.draft.position != start:
                raise RuntimeError(
                    f"draft cache {cycle.draft.position} diverged from the target cursor {start}"
                )
            return start, prompt[start]
        raise ValueError("MTP prefill needs a held token after at least one token")
    for begin, end in spans:
        runner.forward(prompt[begin:end], begin)
        cycle.absorb_prefill(prompt[begin:end], runner.hidden)
    cache_len = spans[-1][1]
    if cycle.draft.position != cache_len:
        raise RuntimeError(
            f"draft cache {cycle.draft.position} diverged from the target cursor {cache_len}"
        )
    return cache_len, prompt[cache_len]


def continue_drafted(runner, prompt, max_new, steps=DRAFT_TOKENS, cycle=None, start=0):
    """Greedy tokens from MTP6 windows. The runner keeps the last window's hold.

    The prompt prefix uses the same page cuts as a resident turn. Each cut
    exports its post-norm state, and the last row is the hidden paired with
    the first token of the next cut. The final prompt id stays out of the cache.
    """
    cycle = cycle or DraftCycle(runner)
    cache_len, held = prefill_held(runner, cycle, prompt, start)
    produced = []
    while len(produced) < max_new:
        drafted = cycle.propose(held, steps)
        before = runner.capture()
        runner.forward([held, *drafted], cache_len)
        samples = _greedy_samples(runner, len(drafted) + 1)
        verify_hidden = _snapshot(runner.hidden)
        emitted, matched = emitted_tokens(drafted, samples)
        runner.restore(before)
        body = [held, *emitted[:-1]]
        runner.forward(body, cache_len)
        cycle.repair(verify_hidden, drafted, matched)
        cache_len += len(body)
        if cycle.draft.position != cache_len:
            raise RuntimeError(
                f"draft cache {cycle.draft.position} diverged from the target cursor {cache_len}"
            )
        produced.extend(emitted)
        held = emitted[-1]
    cycle.tokens = produced
    cycle.cache_len = cache_len
    return produced[:max_new], cycle
