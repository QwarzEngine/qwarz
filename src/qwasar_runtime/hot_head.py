"""65536-token MTP proposer head (recalibrated, promoted 2026-09-22).

Restricts the MTP proposer's head to 65536 tokens (512 complete 128-token
Hadamard groups) rebuilt from the pinned full head at load time. Only the
proposer changes: the target verifier keeps the full 248320-row head, so an
id outside the map merely stops being proposable.

Production port of the validated campaign
``results/20260922-hot64k-xqa/hot64k.py`` (itself the compose-aware port of
the 2026-09-11 draft-phase head, ExLlamaV3 PR303 at 5705f07b): only the
sub-head construction and ``draft.sample_from_state`` are patched. The draft
walk and the input layer are untouched — the rendezvous walk already chains
draft ids on GPU and calls ``sample_from_state``, and R1b hosts the shared
embedding table on GPU. Measured on the production xqa stack (same day, fix
on both arms): -2.4..-2.5 ms/verify of draft weight reads at every context,
+5..+17% decode from 32K to 128K, TTFT par, +0.24 GiB, quality gate 4/4
(``results/20260910-quality-gate/gate-decision-hot64k.json``).

The map (``hot_blocks_64k.txt``) is frequency-calibrated over the full
37-cell matrix corpus (4 coding families x 4 contexts x 2 seeds + JSON +
warmup prompts and production-stack completions; 99.6% group coverage) and
pinned by SHA-256 here and in the session identity. Swapping the map
invalidates sessions.

``QWASAR_HOT64K=0`` disables the head at boot (the service then runs the
full proposer head and reports it in ``/health``).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import types

REVISION = "hot64k-20260922-promoted"
MAP_PATH = Path(__file__).with_name("hot_blocks_64k.txt")
MAP_SHA256 = "352e5e2d449bb07edfa15ae119fda80c9231a163500f6d37f8c3c5f76422f003"
SUBSET_VOCAB = 65536


def enabled(env=os.environ) -> bool:
    return env.get("QWASAR_HOT64K", "1") != "0"


def load_blocks() -> list[int]:
    """Parses and hash-verifies the pinned map; returns the 4096 packed blocks."""
    digest = hashlib.sha256(MAP_PATH.read_bytes()).hexdigest()
    if digest != MAP_SHA256:
        raise ValueError(f"hot head map hash mismatch: expected {MAP_SHA256}, got {digest}")
    blocks = [int(line) for line in MAP_PATH.read_text().splitlines()
              if line and not line.startswith("#")]
    validate_blocks(blocks)
    return blocks


def validate_blocks(blocks, vocab=248320) -> None:
    """Structural contract: sorted, unique, complete aligned 128-token groups."""
    if not blocks or blocks != sorted(set(blocks)) or blocks[0] < 0:
        raise ValueError("blocks must be sorted, unique, nonnegative and nonempty")
    if len(blocks) % 8 or (blocks[-1] + 1) * 16 > vocab:
        raise ValueError("require full vocabulary blocks and complete Hadamard groups")
    for offset in range(0, len(blocks), 8):
        first = blocks[offset]
        if first % 8 or blocks[offset:offset + 8] != list(range(first, first + 8)):
            raise ValueError("require complete aligned 128-token Hadamard groups")


def validate_weight_mapping(generator) -> dict:
    """Bit-exact check: every selected group's effective weights are identical
    to the full head's, via independent reconstruction (same method as the
    2026-09-11 port; 512/512 groups exact)."""
    import torch
    from exllamav3.ext import exllamav3_ext as ext

    full = generator.model.modules[generator.model.logit_layer_idx].inner
    draft = generator.draft_model
    subset = draft.mtp_sub_lm_head
    original_starts = draft.mtp_hot_id_map[::128].cpu().tolist()
    a = torch.empty((full.in_features, 128), device=full.trellis.device, dtype=torch.float16)
    b = torch.empty_like(a)
    errors = []
    for index, original in enumerate(original_starts):
        ext.reconstruct_had_slice(a, full.trellis, full.suh, full.svh[original:], full.K, full.mcg, full.mul1, original)
        ext.reconstruct_had_slice(b, subset.trellis, subset.suh, subset.svh[index * 128:], subset.K, subset.mcg, subset.mul1, index * 128)
        errors.append((a - b).abs().max())
    errors = torch.stack(errors).cpu().tolist()
    result = {"groups": len(errors), "groups_exact": sum(e == 0 for e in errors),
              "effective_weight_max_abs": max(errors)}
    if result["groups_exact"] != result["groups"]:
        raise ValueError(f"hot head weight reconstruction not exact: {result}")
    return result


def install(generator) -> dict:
    """Installs the 65536-token proposer head on ``generator``.

    Raises on any precondition or reconstruction mismatch; the caller decides
    whether that is fatal or a graceful fallback to the full head. Requires
    the stock MTP6 draft layout (the rendezvous walk calls the patched
    ``sample_from_state`` either way).
    """
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3

    draft, target = generator.draft_model, generator.model
    if not generator.mtp_draft or generator.num_draft_tokens != 6 or generator.dynamic_draft:
        raise ValueError("hot head requires a fixed-width MTP6 draft")
    if target.loaded_tp:
        raise ValueError("hot head does not support tensor-parallel targets")
    head = target.modules[target.logit_layer_idx]
    full = head.inner
    if not isinstance(full, LinearEXL3) or head is not target.modules[-1] or draft.attached_model() is not target:
        raise ValueError("donor API drift: unexpected target head/draft attachment")
    if full.trellis.device.type != "cuda" or full.trellis.device.index != 0:
        raise ValueError("hot head requires the full head resident on GPU 0")
    vocab = head.out_features_unpadded
    blocks = load_blocks()
    block_idx = torch.tensor(blocks, device=full.trellis.device, dtype=torch.long)
    token_ids = (block_idx[:, None] * 16 + torch.arange(16, device=block_idx.device)[None, :]).flatten()
    hot_vocab = token_ids.numel()
    if hot_vocab != SUBSET_VOCAB:
        raise ValueError(f"hot head map covers {hot_vocab} tokens, expected {SUBSET_VOCAB}")
    draft.mtp_hot_id_map = token_ids
    draft.mtp_sub_lm_head = LinearEXL3(
        config=target.config, in_features=full.in_features, out_features=hot_vocab,
        suh=full.suh, svh=full.svh.index_select(0, token_ids).contiguous(),
        trellis=full.trellis.index_select(1, block_idx).contiguous(),
        mcg=full.mcg_tensor, mul1=full.mul1_tensor,
        bias=full.bias.index_select(0, token_ids).contiguous() if full.bias is not None else None,
        out_dtype=full.out_dtype, key="qwasar.mtp.hot_lm_head_64k")
    inverse = torch.full((vocab,), -1, device=token_ids.device, dtype=torch.long)
    inverse[token_ids] = torch.arange(hot_vocab, device=token_ids.device)
    draft.mtp_hot_inverse = inverse
    draft.mtp_hot_vocab = hot_vocab

    def sample_from_state(self, state, params):
        logits = self.mtp_sub_lm_head.forward(state, params)
        return self.mtp_hot_id_map[torch.argmax(logits, dim=-1)]

    draft.sample_from_state = types.MethodType(sample_from_state, draft)
    mapping = validate_weight_mapping(generator)
    return {"revision": REVISION, "subset_vocab": hot_vocab, "full_vocab": vocab,
            "map_sha256": MAP_SHA256, "weight_mapping": mapping,
            "target_verifier_unchanged": True}
