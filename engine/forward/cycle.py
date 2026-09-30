"""One MTP6 window on the resident target.

The target cache keeps every token except the one about to be scored. The
prompt prefix uses the production page cuts. Each cut exports its post-norm
state, and that cut's last row is the hidden paired with the first token of
the next cut. The draft proposes six ids from the held token. One target
forward scores the held token and the six drafts. When the runner can cut
Gated DeltaNet, that forward's accepted prefix stays in the cache. Otherwise
the prefix is written again. The first mismatch, or the bonus token when all
six match, stays held for the next window.
"""
from __future__ import annotations

import os
import time

from engine.forward.mtp import DRAFT_TOKENS, acceptance, emitted_tokens
from engine.forward.vision import MM_TOKEN_BASE


# Suffix drafting: when the last NGRAM_N tokens occurred earlier in the
# resident history, copy the tokens that followed instead of rolling the MTP
# head. The copy is a deterministic proposal, so the verifier's ratio test
# (q = 1 on the copied id) still keeps the target's law. 0 disables.
NGRAM = os.environ.get("QWARZ_NGRAM", "0") != "0"
NGRAM_N = int(os.environ.get("QWARZ_NGRAM_N", "5"))
NGRAM_SPAN = 32768


class SuffixIndex:
    """Last position after each n-gram of the recent history."""

    def __init__(self, tokens, n=NGRAM_N, span=NGRAM_SPAN):
        self.n = n
        self.tokens = list(tokens)
        self.after = {}
        start = max(n, len(self.tokens) - span)
        for end in range(start, len(self.tokens)):
            self.after[tuple(self.tokens[end - n:end])] = end

    def extend(self, tokens):
        for token in tokens:
            end = len(self.tokens)
            if end >= self.n:
                self.after[tuple(self.tokens[end - self.n:end])] = end
            self.tokens.append(int(token))

    def propose(self, held, steps):
        """``steps`` ids that followed the last occurrence of history+held, or None."""
        tail = tuple(self.tokens[-(self.n - 1):] + [int(held)]) if self.n > 1 else (int(held),)
        start = self.after.get(tail)
        if start is None:
            return None
        proposal = self.tokens[start:start + steps]
        return proposal if len(proposal) == steps else None


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
        # Host-observed phase seconds per MTP window, accumulated per turn by
        # the caller as deltas: draft (propose), verify (7-token forward),
        # sample (target scoring), replay (rewind or replay forward + repair).
        self.phase_s = [0.0, 0.0, 0.0, 0.0]
        self.windows = 0
        self.window_s = []

    def _model(self):
        if self.draft is None:
            self.runner._ensure()
            from engine.forward.draft import MTPDraft

            pages = getattr(self.runner, "pages", None)
            capacity = pages * 256 if isinstance(pages, int) else None
            self.draft = MTPDraft(self.runner._model, self.runner._table, capacity=capacity)
        return self.draft

    def absorb_prefill(self, token_ids, hidden):
        draft = self._model()
        paired, self.carry = shift_rows(self.carry, hidden)
        # The draft table is text-only. Image ids stay on the target rows.
        safe = [0 if int(token) >= MM_TOKEN_BASE else int(token) for token in token_ids]
        draft.prefill(safe, paired)

    def propose(self, token_id, steps=DRAFT_TOKENS, sampling=None):
        draft = self._model()
        if self.carry is None:
            raise RuntimeError("draft has no target state")
        self.primed = draft.position
        self.proposals = None
        roll = getattr(draft, "roll", None)
        if roll is None:
            token = int(token_id)
            state = self.carry
            drafted = []
            for _ in range(steps):
                token, state = draft.step(token, state)
                drafted.append(int(token))
            return drafted
        if sampling is None:
            drafted, _state = roll(int(token_id), self.carry, steps)
        else:
            drafted, _state = roll(int(token_id), self.carry, steps, sampling=sampling)
            self.proposals = draft.proposals
        return drafted

    def repair_copied(self, verify_hidden, held, drafted, matched):
        """Repair after a window the MTP head did not roll: the held id enters too."""
        draft = self._model()
        import torch

        state = verify_hidden if verify_hidden.dtype == torch.float16 else verify_hidden.to(dtype=torch.float16)
        rows = torch.cat((self.carry.to(torch.float16), state[:, :matched]), dim=1)
        draft.prefill([int(held), *drafted[:matched]], rows)
        self.carry = state[:, matched:matched + 1].contiguous()
        self.accepted += matched
        self.rejected += len(drafted) - matched

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


def _stop_requested(cancel):
    if cancel is None:
        return False
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    if callable(cancel):
        return bool(cancel())
    return False


def _forward(runner, ids, cache_len, embed_ids, inv_freq):
    kwargs = {}
    if embed_ids is not None:
        rows = embed_ids(ids)
        if rows is not None:
            kwargs["embedded"] = rows
    if inv_freq is not None:
        kwargs["inv_freq"] = inv_freq
    if kwargs:
        runner.forward(list(ids), cache_len, **kwargs)
    else:
        runner.forward(list(ids), cache_len)


def _samples(runner, width, choose):
    if choose is None:
        return _greedy_samples(runner, width)
    window = getattr(choose, "window", None)
    if window is not None and getattr(runner, "score_logits", None) is None:
        from engine.forward.token import logits

        tensor = logits(runner._model, runner.last)
        flat = tensor.reshape(-1, tensor.shape[-1])
        if flat.shape[0] != width:
            raise RuntimeError(
                f"the target returned {flat.shape[0]} logit rows for a window of {width}"
            )
        return window(flat)
    score = getattr(runner, "score_logits", None)
    if score is not None:
        rows = list(score())
    else:
        from engine.forward.token import logits

        tensor = logits(runner._model, runner.last)
        vocab = tensor.shape[-1]
        flat = tensor.reshape(-1, vocab)
        if flat.shape[0] != width:
            raise RuntimeError(
                f"the target returned {flat.shape[0]} logit rows for a window of {width}"
            )
        rows = [flat[index] for index in range(width)]
    if len(rows) != width:
        raise RuntimeError(f"the target returned {len(rows)} logit rows for a window of {width}")
    return [int(choose(row, index)) for index, row in enumerate(rows)]


def _verify_sampled(runner, width, verify, drafted, proposals):
    from engine.forward.token import logits

    tensor = logits(runner._model, runner.last)
    flat = tensor.reshape(-1, tensor.shape[-1])
    if flat.shape[0] != width:
        raise RuntimeError(f"the target returned {flat.shape[0]} logit rows for a window of {width}")
    return verify(flat, list(drafted), proposals)


class CancelledTurn(Exception):
    def __init__(self, produced):
        super().__init__("generation cancelled")
        self.produced = list(produced)


def _apply_inject(runner, cycle, cache_len, held, produced, directive, embed_ids, inv_freq, suffix=None):
    inject = [int(token) for token in directive.get("inject") or []]
    if directive.get("drop_last"):
        if not produced:
            raise RuntimeError("cannot drop a token that was not produced")
        produced.pop()
        body = inject[:-1]
        new_hold = inject[-1] if inject else held
    else:
        if not inject:
            return cache_len, held
        body = [held, *inject[:-1]]
        new_hold = inject[-1]
    if body:
        _forward(runner, body, cache_len, embed_ids, inv_freq)
        cycle.absorb_prefill(body, runner.hidden)
        cache_len += len(body)
        if cycle.draft.position != cache_len:
            raise RuntimeError(
                f"draft cache {cycle.draft.position} diverged from the target cursor {cache_len}"
            )
        if suffix is not None:
            suffix.extend(body)
    produced.extend(inject)
    return cache_len, new_hold


def split_spans(spans, cuts):
    """Split prefill spans at extra page-aligned cuts, so a mark can land there."""
    out = []
    for begin, end in spans:
        inner = sorted(cut for cut in set(cuts or ()) if begin < cut < end)
        for cut in inner:
            out.append((begin, cut))
            begin = cut
        out.append((begin, end))
    return out


def prefill_held(runner, cycle, prompt, start=0, embed_ids=None, inv_freq=None, on_span=None, cuts=None):
    """Forward a prompt prefix on the production cuts and absorb each post-norm state.

    The last prompt id is not forwarded. It stays held for the first MTP window.
    A later call starts at the resident cursor and carries the previous row in.
    ``on_span(end, carry)`` sees each cut once its state is complete, so the
    session can mark a prefix it may return to.
    """
    from engine.forward.schedule import suffix_bounds

    prompt = list(prompt)
    spans = split_spans(suffix_bounds(start, len(prompt)), cuts)
    if not spans:
        if len(prompt) >= 2 and start == len(prompt) - 1:
            if cycle.draft is not None and cycle.draft.position != start:
                raise RuntimeError(
                    f"draft cache {cycle.draft.position} diverged from the target cursor {start}"
                )
            return start, prompt[start]
        raise ValueError("MTP prefill needs a held token after at least one token")
    for begin, end in spans:
        _forward(runner, prompt[begin:end], begin, embed_ids, inv_freq)
        cycle.absorb_prefill(prompt[begin:end], runner.hidden)
        if on_span is not None:
            on_span(end, cycle.carry)
    cache_len = spans[-1][1]
    if cycle.draft.position != cache_len:
        raise RuntimeError(
            f"draft cache {cycle.draft.position} diverged from the target cursor {cache_len}"
        )
    return cache_len, prompt[cache_len]


def continue_drafted(
    runner, prompt, max_new, steps=DRAFT_TOKENS, cycle=None, start=0, stop_ids=None,
    choose=None, on_accepted=None, cancel=None, embed_ids=None, inv_freq=None, on_span=None, cuts=None,
):
    """Tokens from MTP6 windows. The runner keeps the last window's hold.

    The prompt prefix uses the same page cuts as a resident turn. Each cut
    exports its post-norm state, and the last row is the hidden paired with
    the first token of the next cut. The final prompt id stays out of the cache.
    A window that would pass max_new, or that contains a stop id, is trimmed
    before it is repaired, so the hold and the cursor describe only the
    returned tokens. The stop id itself stays held.

    ``choose`` replaces the target sample. ``on_accepted`` sees each accepted
    window before the next one is proposed and may inject the reasoning close.
    ``cancel`` aborts before the next window and leaves the caller to restore
    the last commit. Image rows ride ``embed_ids``; a text turn passes neither.
    """
    cycle = cycle or DraftCycle(runner)
    sync = getattr(runner, "synchronize", None)
    timed = callable(sync)
    if timed:
        sync()
        prefill_started = time.perf_counter()
    cache_len, held = prefill_held(
        runner, cycle, prompt, start, embed_ids=embed_ids, inv_freq=inv_freq, on_span=on_span, cuts=cuts,
    )
    if timed:
        sync()
        cycle.host_prefill_s = time.perf_counter() - prefill_started
    produced = []
    cycle.overshot = False
    rewind = getattr(runner, "rewind_recurrent", None)
    single_pass = callable(rewind)
    sampling = getattr(choose, "draft", None)
    verify = getattr(choose, "verify", None)
    if sampling is not None:
        sampling = dict(sampling)
    suffix = SuffixIndex(prompt[:cache_len]) if NGRAM and single_pass and embed_ids is None else None
    phases = cycle.phase_s
    while len(produced) < max_new:
        if _stop_requested(cancel):
            raise CancelledTurn(produced)
        window_started = time.perf_counter()
        copied = suffix.propose(held, steps) if suffix is not None else None
        if copied is not None:
            drafted = copied
            cycle.proposals = [{token: 1.0} for token in drafted] if sampling is not None else None
        else:
            drafted = cycle.propose(held, steps, sampling=sampling) if sampling is not None else cycle.propose(held, steps)
        verify_started = time.perf_counter()
        before = None if single_pass else runner.capture()
        _forward(runner, [held, *drafted], cache_len, embed_ids, inv_freq)
        sample_started = time.perf_counter()
        proposals = getattr(cycle, "proposals", None)
        if verify is not None and proposals is not None:
            samples = _verify_sampled(runner, len(drafted) + 1, verify, drafted, proposals)
        else:
            samples = _samples(runner, len(drafted) + 1, choose)
        verify_hidden = _snapshot(runner.hidden)
        emitted, matched = emitted_tokens(drafted, samples)
        room = max_new - len(produced)
        if len(emitted) > room:
            # The hold is the last returned token. Draft tokens past that
            # budget are not repaired into the cache.
            emitted = emitted[:room]
            matched = len(emitted) - 1
        stopped = False
        if stop_ids:
            for index, token in enumerate(emitted):
                if token in stop_ids:
                    emitted = emitted[: index + 1]
                    matched = len(emitted) - 1
                    stopped = True
                    break
        replay_started = time.perf_counter()
        if not emitted:
            if single_pass:
                rewind(0)
            else:
                runner.restore(before)
            phases[0] += verify_started - window_started
            phases[1] += sample_started - verify_started
            phases[2] += replay_started - sample_started
            phases[3] += time.perf_counter() - replay_started
            cycle.windows += 1
            cycle.window_s.append(time.perf_counter() - window_started)
            break
        body = [held, *emitted[:-1]]
        if single_pass:
            rewind(len(body))
        else:
            runner.restore(before)
            _forward(runner, body, cache_len, embed_ids, inv_freq)
        if copied is not None:
            cycle.repair_copied(verify_hidden, held, drafted, matched)
        else:
            cycle.repair(verify_hidden, drafted, matched)
        if suffix is not None:
            suffix.extend(body)
        cache_len += len(body)
        if cycle.draft.position != cache_len:
            raise RuntimeError(
                f"draft cache {cycle.draft.position} diverged from the target cursor {cache_len}"
            )
        produced.extend(emitted)
        held = emitted[-1]
        directive = on_accepted(produced) if on_accepted is not None else None
        phases[0] += verify_started - window_started
        phases[1] += sample_started - verify_started
        phases[2] += replay_started - sample_started
        phases[3] += time.perf_counter() - replay_started
        cycle.windows += 1
        cycle.window_s.append(time.perf_counter() - window_started)
        if _stop_requested(cancel):
            raise CancelledTurn(produced)
        if directive:
            cache_len, held = _apply_inject(
                runner, cycle, cache_len, held, produced, directive, embed_ids, inv_freq, suffix,
            )
        if len(produced) > max_new:
            cycle.overshot = True
        if directive and directive.get("halt"):
            break
        if stopped and not (directive and directive.get("inject")):
            break
    if _stop_requested(cancel):
        raise CancelledTurn(produced)
    cycle.tokens = list(produced)
    cycle.cache_len = cache_len
    cycle.overshot = len(produced) > max_new
    if cycle.overshot:
        return list(produced), cycle
    return produced[:max_new], cycle
