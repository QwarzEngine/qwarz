"""One MTP6 window on the resident target.

The target cache keeps every token except the one about to be scored. The
draft proposes six ids from that held token. One target forward scores the
held token and the six drafts. The matching prefix is written again and the
first mismatch, or the bonus token when all six match, stays held for the
next window.
"""
from __future__ import annotations

import torch

from engine.forward.mtp import DRAFT_TOKENS, acceptance, emitted_tokens


def _fp16(hidden):
    return hidden if hidden.dtype == torch.float16 else hidden.to(torch.float16)


class DraftCycle:
    def __init__(self, runner):
        self.runner = runner
        self.draft = None
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
        state = _fp16(hidden)
        blank = torch.zeros_like(state[:, :1]) if self.carry is None else self.carry
        draft.prefill(list(token_ids), torch.cat((blank, state[:, :-1]), dim=1))
        self.carry = state[:, -1:].contiguous()

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
        state = _fp16(verify_hidden)
        if matched:
            draft.prefill(list(drafted[:matched]), state[:, :matched])
        self.carry = state[:, matched:matched + 1].contiguous()
        self.accepted += matched
        self.rejected += len(drafted) - matched

    def rate(self):
        return acceptance(self.accepted, self.rejected)


def continue_drafted(runner, prompt, max_new, steps=DRAFT_TOKENS):
    """Greedy tokens from MTP6 windows. The runner keeps the last window's hold."""
    from engine.forward.token import logits

    prompt = list(prompt)
    if len(prompt) < 2:
        raise ValueError("MTP prefill needs a held token after at least one token")
    prefill, held = prompt[:-1], prompt[-1]
    runner.forward(prefill, 0)
    cycle = DraftCycle(runner)
    cycle.absorb_prefill(prefill, runner.hidden)
    cache_len = len(prefill)
    produced = []
    while len(produced) < max_new:
        drafted = cycle.propose(held, steps)
        before = runner.capture()
        runner.forward([held, *drafted], cache_len)
        samples = [int(token) for token in logits(runner._model, runner.last).argmax(dim=-1).view(-1).tolist()]
        verify_hidden = runner.hidden.detach().clone()
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
    return produced[:max_new], cycle
