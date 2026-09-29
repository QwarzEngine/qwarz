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


def chooser(temperature, top_p, seed):
    """One seeded draw per logit row, in window order."""
    rng = random.Random(seed)

    def choose(logits, _index):
        return sample_id(logits, temperature, top_p, rng)

    return choose
