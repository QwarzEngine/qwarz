"""MTP6 proposer.

One 4-bit block reads the target's post-final-norm state. The first draft
token is paired with the hidden state of the previous token. Each later
token is paired with this block's own output. Sampling is the full target
head, argmax.
"""
from __future__ import annotations

import torch
from exllamav3.ext import exllamav3_ext as ext
from torch.nn.functional import scaled_dot_product_attention

from engine.forward.attention import HEAD_DIM, HEADS, KV_HEADS, KV_OUT, Q_OUT, project, rope
from engine.forward.embed import gather
from engine.forward.mlp import HIDDEN
from engine.forward.mtp import DRAFT_TOKENS
from engine.forward.projections import exl3
from engine.forward.token import rms

INTERMEDIATE = 17408


class MTPDraft:
    def __init__(self, catalog, table):
        self.catalog = catalog
        self.table = table
        self.attn = catalog.group("mtp.layers.0.self_attn")
        self.mlp = catalog.group("mtp.layers.0.mlp")
        self.fc = catalog.group("mtp.fc")
        self.norm_hidden = catalog.tensor("mtp.pre_fc_norm_hidden.weight")
        self.norm_embed = catalog.tensor("mtp.pre_fc_norm_embedding.weight")
        self.norm_in = catalog.tensor("mtp.layers.0.input_layernorm.weight")
        self.norm_post = catalog.tensor("mtp.layers.0.post_attention_layernorm.weight")
        self.norm_out = catalog.tensor("mtp.norm.weight")
        self.past_k = None
        self.past_v = None
        self.position = 0

    def reset(self):
        self.past_k = None
        self.past_v = None
        self.position = 0

    def truncate(self, position):
        if self.past_k is None or position < 0 or position > self.past_k.shape[1]:
            raise RuntimeError(f"draft cache cannot rewind to {position}")
        self.past_k = self.past_k[:, :position]
        self.past_v = self.past_v[:, :position]
        self.position = position

    def _fuse(self, token_ids, target_hidden):
        embedded = gather(self.table, token_ids, torch.float16).cuda().contiguous()
        hidden = target_hidden if target_hidden.dtype == torch.float16 else target_hidden.to(torch.float16)
        joined = torch.cat((rms(self.norm_embed, embedded), rms(self.norm_hidden, hidden)), dim=-1)
        layer = exl3(
            "fc", self.fc["trellis"], self.fc["suh"], self.fc["svh"], self.fc["mul1"],
            HIDDEN, HIDDEN * 2,
        )
        return layer.forward(joined, {})

    def _block(self, hidden):
        normed = rms(self.norm_in, hidden)
        seqlen = normed.shape[1]
        query = torch.empty((1, seqlen, HEADS, HEAD_DIM), dtype=torch.float16, device=normed.device)
        gate = torch.empty((1, seqlen, HEADS * HEAD_DIM), dtype=torch.float16, device=normed.device)
        packed = project(normed, self.attn, "q_proj", Q_OUT, HIDDEN, torch.float16)
        ext.deinterleave_qg(packed, query, gate, HEAD_DIM)
        key = project(normed, self.attn, "k_proj", KV_OUT, HIDDEN, torch.float16).view(1, seqlen, KV_HEADS, HEAD_DIM)
        value = project(normed, self.attn, "v_proj", KV_OUT, HIDDEN, torch.float16).view(1, seqlen, KV_HEADS, HEAD_DIM)
        positions = None
        if self.position:
            positions = torch.tensor([self.position], dtype=torch.int32, device=normed.device)
        query, key = rope().apply(
            query, key, 0, positions, None, True,
            self.attn["q_norm.weight"], self.attn["k_norm.weight"], 1e-6, 1.0, None, False,
        )
        if self.past_k is None:
            seen_k, seen_v = key, value
        else:
            seen_k = torch.cat((self.past_k, key), dim=1)
            seen_v = torch.cat((self.past_v, value), dim=1)
        # Bottom-right causal: the chunk sees the cache and not later tokens in the chunk.
        mixed = scaled_dot_product_attention(
            query.transpose(1, 2), seen_k.transpose(1, 2), seen_v.transpose(1, 2),
            is_causal=True, scale=HEAD_DIM ** -0.5, enable_gqa=True,
        ).transpose(1, 2).contiguous()
        flat = mixed.reshape(1, seqlen, HEADS * HEAD_DIM)
        ext.mul_sigmoid_(flat, gate)
        attended = project(flat, self.attn, "o_proj", HIDDEN, HEADS * HEAD_DIM, torch.float32)
        residual = hidden.float() + attended
        entered = rms(self.norm_post, residual)
        gate_up = project(entered, self.mlp, "gate_proj", INTERMEDIATE, HIDDEN, torch.float16)
        up = project(entered, self.mlp, "up_proj", INTERMEDIATE, HIDDEN, torch.float16)
        activated = torch.empty_like(up)
        ext.silu_mul(gate_up, up, activated, 0.0)
        produced = project(activated, self.mlp, "down_proj", HIDDEN, INTERMEDIATE, torch.float32)
        self.past_k = seen_k
        self.past_v = seen_v
        self.position += seqlen
        return rms(self.norm_out, residual + produced)

    def prefill(self, token_ids, target_hidden):
        ids = list(token_ids)
        offset = 0
        while offset < len(ids):
            end = min(offset + 8192, len(ids))
            piece = torch.tensor(ids[offset:end], dtype=torch.long).view(1, -1)
            self._block(self._fuse(piece, target_hidden[:, offset:end]))
            offset = end

    def step(self, token_id, hidden):
        state = self._block(self._fuse(torch.tensor([[token_id]], dtype=torch.long), hidden))
        head = self.catalog.group("lm_head")
        layer = exl3(
            "lm_head", head["trellis"], head["suh"], head["svh"], head["mul1"],
            248320, HIDDEN, torch.float16,
        )
        nxt = int(layer.forward(state, {}).reshape(-1).argmax())
        return nxt, state

    def propose(self, prefill_ids, target_hidden, next_id, steps=DRAFT_TOKENS):
        self.reset()
        hidden = target_hidden if target_hidden.dtype == torch.float16 else target_hidden.to(torch.float16)
        blank = torch.zeros_like(hidden[:, :1])
        shifted = torch.cat((blank, hidden[:, :-1]), dim=1)
        self.prefill(prefill_ids, shifted)
        token = next_id
        carry = hidden[:, -1:]
        drafted = []
        for _ in range(steps):
            token, carry = self.step(token, carry)
            drafted.append(token)
        return drafted
