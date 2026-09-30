"""Target-token choice for one resident window.

Temperature 0 is argmax, so the same logits always repeat. A positive
temperature draws from the nucleus with the request seed. The draw is pure
Python so a fake forward can show two seeds diverge without a GPU.
"""
from __future__ import annotations

import math
import random


def _values(logits):
    if callable(getattr(logits, "detach", None)):
        return [float(item) for item in logits.detach().float().reshape(-1).cpu().tolist()]
    return [float(item) for item in logits]


def _argmax(logits):
    if callable(getattr(logits, "argmax", None)) and callable(getattr(logits, "reshape", None)):
        return int(logits.reshape(-1).argmax())
    values = _values(logits)
    return max(range(len(values)), key=lambda index: values[index])


def _sample_tensor(logits, temperature, top_p, rng):
    """Nucleus sample on the logit tensor. The full vocabulary never becomes a Python list."""
    import torch

    values = logits.detach().float().reshape(-1)
    centered = (values - values.max()) / temperature
    weights = centered.exp()
    total = weights.sum()
    if not bool(total.isfinite()) or float(total) <= 0:
        return int(values.argmax())
    ordered, indices = weights.sort(descending=True, stable=True)
    cumulative = ordered.cumsum(0)
    keep = int(torch.searchsorted(cumulative, total * top_p)) + 1
    keep = min(max(keep, 1), int(ordered.shape[0]))
    kept = ordered[:keep]
    draw = rng.random() * kept.sum()
    position = int(torch.searchsorted(kept.cumsum(0), draw))
    position = min(position, keep - 1)
    return int(indices[position])


def sample_id(logits, temperature, top_p, rng):
    """Pick one token. ``rng`` is a ``random.Random`` advanced only when sampling."""
    if temperature == 0:
        return _argmax(logits)
    if callable(getattr(logits, "detach", None)):
        return _sample_tensor(logits, temperature, top_p, rng)
    values = _values(logits)
    if not values:
        raise RuntimeError("the target returned no logits")
    scaled = [value / temperature for value in values]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return _argmax(logits)
    order = sorted(range(len(weights)), key=lambda index: weights[index], reverse=True)
    kept = []
    mass = 0.0
    for index in order:
        kept.append(index)
        mass += weights[index]
        if mass >= top_p * total:
            break
    if not kept:
        kept = [order[0]]
        mass = weights[kept[0]]
    draw = rng.random() * mass
    cursor = 0.0
    for index in kept:
        cursor += weights[index]
        if draw <= cursor:
            return index
    return kept[-1]


def _draw(weights, keep, rng):
    """One seeded draw over the first ``keep`` weights, as the tensor path does."""
    kept = weights[:keep]
    draw = rng.random() * sum(kept)
    cursor = 0.0
    for position, weight in enumerate(kept):
        cursor += weight
        if draw <= cursor:
            return position
    return keep - 1


def _nucleus(weights, top_p):
    """Kept count per row, as ExLlama's top-p step does after top-k.

    The top-k weights are renormalized and a token is kept while the
    cumulative mass including it is <= top_p; the first token always stays.
    The token that crosses top_p is dropped. This is the sampler every gated
    ExLlama stack served; including the crossing token made the tail heavier.
    """
    cumulative = weights.cumsum(-1) / weights.sum(-1, keepdim=True)
    return (cumulative <= top_p).sum(-1).clamp(min=1)


def sample_window(rows, temperature, top_p, top_k, rng, vocab=None):
    """Sample every row of a verify window with one device-to-host copy.

    Temperature, then the ``top_k`` largest logits, then the nucleus over
    them: the order of the Qwen recommended sampler. The draws stay on the
    host ``rng``, one per row in window order, so a seed repeats.
    """
    import torch

    values = rows.detach().reshape(rows.shape[0], -1)
    if vocab is not None and values.shape[-1] > vocab:
        # Rows past the tokenizer are padding in the output head.
        values = values[:, :vocab]
    values = values.float()
    if temperature == 0:
        return [int(token) for token in values.argmax(dim=-1).tolist()]
    top, indices = values.topk(min(int(top_k), values.shape[-1]), dim=-1)
    weights = ((top - top[:, :1]) / temperature).exp()
    keep = _nucleus(weights, top_p)
    packed = torch.cat((weights, indices.to(torch.float32), keep.unsqueeze(-1).to(torch.float32)), dim=-1)
    host = packed.cpu().tolist()
    width = top.shape[-1]
    chosen = []
    for row in host:
        keep_row = min(max(int(row[-1]), 1), width)
        position = _draw(row[:width], keep_row, rng)
        chosen.append(int(row[width + position]))
    return chosen


def _target_rows(rows, temperature, top_p, top_k, vocab):
    """Per row, the served distribution as (ids, probs): top_k, then the nucleus."""
    import torch

    values = rows.detach().reshape(rows.shape[0], -1)
    if vocab is not None and values.shape[-1] > vocab:
        values = values[:, :vocab]
    top, indices = values.float().topk(min(int(top_k), values.shape[-1]), dim=-1)
    weights = ((top - top[:, :1]) / temperature).exp()
    keep = _nucleus(weights, top_p)
    packed = torch.cat((weights, indices.to(torch.float32), keep.unsqueeze(-1).to(torch.float32)), dim=-1)
    width = top.shape[-1]
    out = []
    for row in packed.cpu().tolist():
        keep_row = min(max(int(row[-1]), 1), width)
        kept = row[:keep_row]
        total = sum(kept)
        out.append({int(row[width + j]): kept[j] / total for j in range(keep_row)})
    return out


def _draw_from(dist, rng):
    draw = rng.random()
    cursor = 0.0
    last = None
    for token, prob in dist.items():
        cursor += prob
        last = token
        if draw <= cursor:
            return token
    return last


def speculative_window(rows, drafted, proposals, temperature, top_p, top_k, rng, vocab=None):
    """Leviathan/Chen acceptance of sampled drafts; the output keeps the target law.

    ``proposals[i]`` is the draft distribution the i-th id was drawn from.
    Draft i is kept with probability min(1, p/q); the first rejection draws
    from the normalized max(0, p - q), which never contains the rejected id,
    so the matched prefix ends there. A fully kept window draws the bonus
    from the last row. Returned ids plug into ``emitted_tokens``.
    """
    targets = _target_rows(rows, temperature, top_p, top_k, vocab)
    samples = []
    for index, token in enumerate(drafted):
        p = targets[index]
        q = proposals[index]
        ratio = p.get(token, 0.0) / max(q.get(token, 0.0), 1e-30)
        if rng.random() < min(1.0, ratio):
            samples.append(token)
            continue
        residual = {t: prob - q.get(t, 0.0) for t, prob in p.items() if prob > q.get(t, 0.0)}
        total = sum(residual.values())
        if total <= 0:
            residual, total = p, 1.0
        samples.append(_draw_from({t: v / total for t, v in residual.items()}, rng))
        samples.extend([-1] * (len(drafted) - index))
        return samples
    samples.append(_draw_from(targets[len(drafted)], rng))
    return samples


def chooser(temperature, top_p, seed, top_k=0, vocab=None, speculative=False):
    """One seeded draw per logit row, in window order.

    With ``top_k`` the whole window is sampled on the device at once
    (``choose.window``); without it each row keeps the full-vocabulary nucleus.
    """
    rng = random.Random(seed)

    def choose(logits, _index):
        return sample_id(logits, temperature, top_p, rng)

    if top_k and top_k > 0:
        def window(rows):
            return sample_window(rows, temperature, top_p, top_k, rng, vocab)

        choose.window = window
        if speculative and temperature > 0:
            def verify(rows, drafted, proposals):
                return speculative_window(rows, drafted, proposals, temperature, top_p, top_k, rng, vocab)

            choose.verify = verify
            choose.draft = {"temperature": float(temperature), "top_p": float(top_p),
                            "top_k": int(top_k), "seed": int(seed)}
    return choose
