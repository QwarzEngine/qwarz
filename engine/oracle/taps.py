"""Chooses the ExLlama modules the q38 oracle records.

The capture itself stays behind an explicit run. Tests only check the
selection: one Gated DeltaNet layer, one attention layer, one NVFP4 MLP,
the embedding and the vocabulary head.
"""
from __future__ import annotations


REQUIRED = ("GatedDeltaNet", "Attention", "GatedMLP")
OPTIONAL = {"Embedding": "embedding", "LMHead": "head"}
# The last prompt leaves one 256-token page so the greedy token still fits
# in the native 262,144-token cache.
CONTEXTS = (64, 32768, 261888)


def select_taps(modules):
    """``modules`` is a sequence of ``(key, class_name)`` in model order."""
    first = {}
    for key, kind in modules:
        first.setdefault(kind, key)
    missing = [kind for kind in REQUIRED if kind not in first]
    if missing:
        raise ValueError("oracle taps missing: " + ", ".join(missing))
    chosen = {
        "gdn": first["GatedDeltaNet"],
        "attention": first["Attention"],
        "mlp": first["GatedMLP"],
    }
    for kind, name in OPTIONAL.items():
        if kind in first:
            chosen[name] = first[kind]
    return chosen


def capture_plan(modules):
    taps = select_taps(modules)
    return {
        "taps": taps,
        "contexts": list(CONTEXTS),
        "records": ["first_greedy_token", "layer_input_sha256", "layer_output_sha256"],
        "full_tensors": "short context only",
        "serves_traffic": False,
    }
