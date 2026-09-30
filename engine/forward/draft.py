"""MTP6 proposer.

One 4-bit block reads the target's post-final-norm state. The first draft
token is paired with the hidden state of the previous token. Each later
token is paired with this block's own output. Sampling is argmax on the
65536-row proposer head. The target verifier keeps the full head.
"""
from __future__ import annotations

import os

import torch
from exllamav3.ext import exllamav3_ext as ext
from torch.nn.functional import scaled_dot_product_attention

from engine.forward.attention import HEAD_DIM, HEADS, KV_HEADS, KV_OUT, Q_OUT, project, rope
from engine.forward.embed import gather
from engine.forward.mlp import HIDDEN
from engine.forward.mtp import DRAFT_TOKENS
from engine.forward.projections import cached_exl3
from engine.forward.token import _proposer_head, rms

INTERMEDIATE = 17408
# The SM120 FMHA cubin is the paged one already resident for target prefill.
# Queries below 7680, and caches below 32768, stay on Flash: packing a short
# page costs as much as the attention. The page pool covers the native context.
_FP8_PAGE = 128
_FP8_MIN_QUERY = 7680
_FP8_MIN_KV = 32768
_FP8_MAX_PAGES = 2048


def draft_attend(query, key, value):
    """Bottom-right causal mix of the new rows against the whole cache.

    Torch's is_causal flag is top-left. A single decode row would then see
    only the first cached key, and later draft steps leave the greedy chain.
    Equal lengths are a square causal mask. One new row sees every key. A
    shorter chunk sees the whole past plus a causal prefix of itself.
    """
    q_len = query.shape[1]
    kv_len = key.shape[1]
    scale = HEAD_DIM ** -0.5
    if (
        q_len != kv_len
        and q_len != 1
        and q_len >= _FP8_MIN_QUERY
        and kv_len >= _FP8_MIN_KV
    ):
        mixed = _fp8_bottom_right(query, key, value, scale)
        return mixed.transpose(1, 2).contiguous()
    attended_q = query.transpose(1, 2)
    attended_k = key.transpose(1, 2)
    attended_v = value.transpose(1, 2)
    if q_len == kv_len:
        mixed = scaled_dot_product_attention(
            attended_q, attended_k, attended_v, is_causal=True, scale=scale, enable_gqa=True,
        )
    elif q_len == 1:
        mixed = scaled_dot_product_attention(
            attended_q, attended_k, attended_v, is_causal=False, scale=scale, enable_gqa=True,
        )
    else:
        mixed = _merge_bottom_right(attended_q, attended_k, attended_v, scale)
    return mixed.transpose(1, 2).contiguous()


def _merge_bottom_right(query, key, value, scale):
    """Merge full attention over the past with causal attention over the chunk."""
    past = key.shape[-2] - query.shape[-2]
    op = torch.ops.aten._scaled_dot_product_flash_attention
    out_past, lse_past, *_rest = op(
        query, key[:, :, :past], value[:, :, :past], 0.0, False, False, scale=scale,
    )
    out_chunk, lse_chunk, *_rest = op(
        query, key[:, :, past:], value[:, :, past:], 0.0, True, False, scale=scale,
    )
    left = lse_past.unsqueeze(-1)
    right = lse_chunk.unsqueeze(-1)
    total = torch.logaddexp(left, right)
    return (
        out_past.float() * (left - total).exp() + out_chunk.float() * (right - total).exp()
    ).to(dtype=query.dtype)


class _Fp8Workspace:
    """FP8 pages for one draft cache, allocated once for the native context."""

    def __init__(self):
        self.k = None
        self.v = None
        self.out = None
        self.cu = None
        self.seqlens = None
        self.table = None
        self.kernel = None

    def attend(self, query, key, value, scale):
        # query, key, value: [1, seq, heads, dim] fp16, token-major. The
        # kernel reads packed pages and writes [q, heads, dim].
        device = query.device
        q_len = query.shape[1]
        kv_len = key.shape[1]
        pages = (kv_len + _FP8_PAGE - 1) // _FP8_PAGE
        self._pools(pages, device)
        self._pack(key, self.k, kv_len, pages)
        self._pack(value, self.v, kv_len, pages)
        rows = query[0, :q_len]
        if not rows.is_contiguous():
            rows = rows.contiguous()
        q8 = rows.clamp(-448, 448).to(torch.float8_e4m3fn)
        if self.out is None or self.out.shape[0] < q_len or self.out.device != device:
            self.out = torch.empty(max(q_len, 8192), HEADS, HEAD_DIM, dtype=torch.float16, device=device)
        out = self.out[:q_len]
        self.seqlens.fill_(kv_len)
        self.cu[1:].fill_(q_len)
        # Probability rows are stored as fp8 multiplied by 256.
        self._load()(
            q8,
            self.k[:pages],
            self.v[:pages],
            out,
            self.table[:, :pages],
            self.seqlens,
            self.cu,
            is_causal=True,
            sm_scale=float(scale),
            max_seqlen_q=q_len,
            v_scale=1.0 / 256.0,
        )
        return out.transpose(0, 1).unsqueeze(0)

    def _pools(self, pages, device):
        if pages > _FP8_MAX_PAGES:
            raise RuntimeError(f"draft FP8 cache needs {pages} pages")
        if self.k is not None and self.k.shape[0] >= pages and self.k.device == device:
            return
        self.k = torch.empty(
            _FP8_MAX_PAGES, KV_HEADS, _FP8_PAGE, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device,
        )
        self.v = torch.empty_like(self.k)
        self.cu = torch.zeros(2, dtype=torch.int32, device=device)
        self.seqlens = torch.empty(1, dtype=torch.int32, device=device)
        self.table = torch.arange(_FP8_MAX_PAGES, device=device, dtype=torch.int32).view(1, -1)

    def reserve(self, device):
        self._pools(1, device)

    def _pack(self, src, pool, seq, pages):
        # The draft cache is [1, cap, 4, 256] with a tight token stride.
        # repage clamps into the resident fp8 pages, including the zero tail
        # of a partial page. Compacting that prefix would allocate another
        # full KV buffer beside a cache that already fills the card.
        token_stride = KV_HEADS * HEAD_DIM
        base = src[0] if src.ndim == 4 else src
        tight = (
            base.shape[0] >= seq
            and base.stride(0) == token_stride
            and base.stride(1) == HEAD_DIM
            and base.stride(2) == 1
        )
        view = (
            base.as_strided((seq, KV_HEADS, HEAD_DIM), (token_stride, HEAD_DIM, 1))
            if tight
            else base[:seq].contiguous()
        )
        import triton
        from qwasar_bench.prims_prefill import repage

        n = pages * token_stride * _FP8_PAGE
        repage[(triton.cdiv(n, 1024),)](view, pool[:pages], seq, n, 1024)

    def _load(self):
        if self.kernel is None:
            from qwasar_runtime.hybrid import prepare_environment

            prepare_environment()
            from flashinfer.attention.cute_dsl.sm120_fmha import sm120_fmha_fp8_paged_prefill

            self.kernel = sm120_fmha_fp8_paged_prefill
        return self.kernel


_ACTIVE = None


def _workspace():
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = _Fp8Workspace()
    return _ACTIVE


def _fp8_bottom_right(query, key, value, scale):
    return _workspace().attend(query, key, value, scale)


# The six-step proposal is one CUDA graph when the KV store is preallocated.
DRAFT_GRAPH = os.environ.get("QWARZ_DRAFT_GRAPH", "1") != "0"
# The graphed proposal reads an FP8 shadow of the draft KV (half the bytes).
# Prefill and repair keep reading the FP16 store. Drafts only steer speed:
# every emitted token is still the target's.
DRAFT_KV8 = os.environ.get("QWARZ_DRAFT_KV8", "1") != "0"
_FP8_LIMIT = 448.0
# Widest top_k a graphed sampled roll records; wider falls back to eager.
_MAX_WIDTH = 64


def _fp8(tensor):
    return tensor.clamp(-_FP8_LIMIT, _FP8_LIMIT).to(torch.float8_e4m3fn)


class MTPDraft:
    def __init__(self, catalog, table, capacity=None):
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
        # Pages live on the catalog, which the runner owns. Reserving them
        # before the first decode graph keeps the allocation off a fragmented pool.
        global _ACTIVE
        workspace = getattr(catalog, "_draft_fp8", None)
        if workspace is None:
            workspace = _Fp8Workspace()
            catalog._draft_fp8 = workspace
        self._fp8 = workspace
        workspace.reserve(table.device)
        _ACTIVE = workspace
        self.past_k = None
        self.past_v = None
        self.position = 0
        self._k_store = None
        self._v_store = None
        self._pos = None
        self._token = None
        self._ids = None
        self.capacity = capacity
        self._k8 = None
        self._v8 = None
        self._static = None
        self._roll_graphs = {}
        self._pool = None

    def reset(self):
        self.past_k = None
        self.past_v = None
        self.position = 0

    def truncate(self, position):
        if self.past_k is None or position < 0 or position > self.past_k.shape[1]:
            raise RuntimeError(f"draft cache cannot rewind to {position}")
        self.position = position
        self.past_k = self._k_store[:, :position]
        self.past_v = self._v_store[:, :position]

    def _write_kv(self, key, value):
        seq = key.shape[1]
        need = self.position + seq
        cap = 0 if self._k_store is None else self._k_store.shape[1]
        if cap == 0 and self.capacity and need <= self.capacity:
            # One allocation for the whole context: the graphed roll records
            # these addresses, and doubling would briefly hold 1.5x the store.
            self._k_store = torch.empty(1, self.capacity, key.shape[2], key.shape[3], dtype=key.dtype,
                                        device=key.device)
            self._v_store = torch.empty_like(self._k_store)
            if DRAFT_KV8:
                self._k8 = torch.empty(self._k_store.shape, dtype=torch.float8_e4m3fn, device=key.device)
                self._v8 = torch.empty_like(self._k8)
            cap = self.capacity
        if cap < need:
            grown = 256 if cap == 0 else cap
            while grown < need:
                grown *= 2
            new_k = torch.empty(1, grown, key.shape[2], key.shape[3], dtype=key.dtype, device=key.device)
            new_v = torch.empty_like(new_k)
            if self.position:
                new_k[:, : self.position].copy_(self._k_store[:, : self.position])
                new_v[:, : self.position].copy_(self._v_store[:, : self.position])
            self._k_store = new_k
            self._v_store = new_v
        end = self.position + seq
        self._k_store[:, self.position : end].copy_(key)
        self._v_store[:, self.position : end].copy_(value)
        if self._k8 is not None and cap == self.capacity:
            self._k8[:, self.position : end].copy_(_fp8(key))
            self._v8[:, self.position : end].copy_(_fp8(value))
        self.position = end
        self.past_k = self._k_store[:, :end]
        self.past_v = self._v_store[:, :end]
        return self.past_k, self.past_v

    def _fuse(self, token_ids, target_hidden):
        if token_ids.device != self.table.device:
            token_ids = token_ids.to(self.table.device)
        embedded = gather(self.table, token_ids, torch.float16).contiguous()
        hidden = target_hidden if target_hidden.dtype == torch.float16 else target_hidden.to(torch.float16)
        joined = torch.cat((rms(self.norm_embed, embedded), rms(self.norm_hidden, hidden)), dim=-1)
        layer = cached_exl3(
            self.fc, "fc", self.fc["trellis"], self.fc["suh"], self.fc["svh"], self.fc["mul1"],
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
        start = self.position
        positions = None
        if start:
            # One buffer, filled on this stream. RoPE is queued after the fill,
            # so the next draft step cannot overwrite the offset early.
            if self._pos is None:
                self._pos = torch.empty(1, dtype=torch.int32, device=normed.device)
            self._pos.fill_(start)
            positions = self._pos
        query, key = rope().apply(
            query, key, 0, positions, None, True,
            self.attn["q_norm.weight"], self.attn["k_norm.weight"], 1e-6, 1.0, None, False,
        )
        seen_k, seen_v = self._write_kv(key, value)
        mixed = draft_attend(query, seen_k, seen_v)
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
        return rms(self.norm_out, residual + produced)

    def _block_static(self, hidden, position, length):
        """``_block`` for one row at a device-side position: graph-safe."""
        normed = rms(self.norm_in, hidden)
        query = torch.empty((1, 1, HEADS, HEAD_DIM), dtype=torch.float16, device=normed.device)
        gate = torch.empty((1, 1, HEADS * HEAD_DIM), dtype=torch.float16, device=normed.device)
        packed = project(normed, self.attn, "q_proj", Q_OUT, HIDDEN, torch.float16)
        ext.deinterleave_qg(packed, query, gate, HEAD_DIM)
        key = project(normed, self.attn, "k_proj", KV_OUT, HIDDEN, torch.float16).view(1, 1, KV_HEADS, HEAD_DIM)
        value = project(normed, self.attn, "v_proj", KV_OUT, HIDDEN, torch.float16).view(1, 1, KV_HEADS, HEAD_DIM)
        query, key = rope().apply(
            query, key, 0, position, None, True,
            self.attn["q_norm.weight"], self.attn["k_norm.weight"], 1e-6, 1.0, None, False,
        )
        slot = position.long()
        self._k_store[0].index_copy_(0, slot, key[0])
        self._v_store[0].index_copy_(0, slot, value[0])
        mixed = torch.empty_like(query)
        if self._k8 is not None:
            # index_copy_ has no FP8 kernel; the bytes move through a uint8 view.
            self._k8.view(torch.uint8)[0].index_copy_(0, slot, _fp8(key[0]).view(torch.uint8))
            self._v8.view(torch.uint8)[0].index_copy_(0, slot, _fp8(value[0]).view(torch.uint8))
            self._static["attention"].attend(query, self._k8, self._v8, length, mixed)
        else:
            self._static["attention"].attend(query, self._k_store, self._v_store, length, mixed)
        flat = mixed.reshape(1, 1, HEADS * HEAD_DIM)
        ext.mul_sigmoid_(flat, gate)
        attended = project(flat, self.attn, "o_proj", HIDDEN, HEADS * HEAD_DIM, torch.float32)
        residual = hidden.float() + attended
        entered = rms(self.norm_post, residual)
        gate_up = project(entered, self.mlp, "gate_proj", INTERMEDIATE, HIDDEN, torch.float16)
        up = project(entered, self.mlp, "up_proj", INTERMEDIATE, HIDDEN, torch.float16)
        activated = torch.empty_like(up)
        ext.silu_mul(gate_up, up, activated, 0.0)
        produced = project(activated, self.mlp, "down_proj", HIDDEN, INTERMEDIATE, torch.float32)
        return rms(self.norm_out, residual + produced)

    def _graphable(self, steps, sampling=None):
        return (
            DRAFT_GRAPH
            and (sampling is None or sampling["top_k"] <= _MAX_WIDTH)
            and self.capacity is not None
            and self._k_store is not None
            and self._k_store.shape[1] == self.capacity
            and self.position + steps <= self.capacity
            and self.table.device.type == "cuda"
        )

    def _static_buffers(self, steps):
        """Inputs and outputs every graphed roll records; greedy and sampled share them."""
        from engine.forward.draft_attention import DecodeAttention

        static = self._static
        if static is not None and static["steps"] == steps:
            return static
        device = self.table.device
        self._static = static = {
            "steps": steps,
            "token": torch.zeros(1, 1, dtype=torch.long, device=device),
            "carry": torch.zeros(1, 1, HIDDEN, dtype=torch.float16, device=device),
            "base": torch.zeros(1, dtype=torch.int32, device=device),
            "pos": torch.zeros(steps, dtype=torch.int32, device=device),
            "len": torch.zeros(steps, dtype=torch.int32, device=device),
            "ids": torch.zeros(steps, dtype=torch.long, device=device),
            "rows": torch.zeros(steps, 2 * _MAX_WIDTH, dtype=torch.float32, device=device),
            "noise": torch.full((steps, _MAX_WIDTH), 0.5, dtype=torch.float32, device=device),
            "attention": DecodeAttention(HEADS, KV_HEADS, HEAD_DIM, device),
        }
        self._roll_graphs = {}
        return static

    def _roll_static(self, steps, sampling):
        static = self._static
        layer, id_map = _proposer_head(self.catalog)
        carry = static["carry"]
        for index in range(steps):
            position = static["pos"].narrow(0, index, 1)
            length = static["len"].narrow(0, index, 1)
            torch.add(static["base"], index, out=position)
            torch.add(static["base"], index + 1, out=length)
            carry = self._block_static(self._fuse(static["token"], carry), position, length)
            logits = layer.forward(carry, {}).reshape(-1)
            if sampling is None:
                local = logits.argmax()
            else:
                width = sampling["top_k"]
                local, row = self._pick(logits, sampling, id_map, static["noise"][index, :width])
                static["rows"][index, :2 * width].copy_(row)
            # A 0-d index would read the value back to the host (a sync the
            # capture refuses); index_select stays on the device.
            local = local.reshape(1)
            nxt = local if id_map is None else id_map.index_select(0, local)
            static["ids"].narrow(0, index, 1).copy_(nxt)
            static["token"].view(-1).copy_(nxt)

    def _roll_graphed(self, token_id, hidden, steps, sampling):
        width = 0 if sampling is None else sampling["top_k"]
        static = self._static_buffers(steps)
        static["token"].fill_(int(token_id))
        static["carry"].copy_(hidden.reshape(1, 1, HIDDEN))
        static["base"].fill_(self.position)
        if sampling is not None:
            static["noise"].uniform_(generator=self._generator(sampling, static["noise"].device))
        key = None if sampling is None else (sampling["temperature"], sampling["top_p"], sampling["top_k"])
        slot = self._roll_graphs.setdefault(key, {"warm": 0, "graph": None})
        if slot["graph"] is not None:
            slot["graph"].replay()
        elif slot["warm"] >= 2:
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, pool=self._pool):
                self._roll_static(steps, sampling)
            slot["graph"] = graph
            graph.replay()
        else:
            slot["warm"] += 1
            self._roll_static(steps, sampling)
        self.position += steps
        self.past_k = self._k_store[:, :self.position]
        self.past_v = self._v_store[:, :self.position]
        if sampling is None:
            self.proposals = None
            return static["ids"].tolist(), None
        rows = static["rows"][:, :2 * width].reshape(-1)
        packed = torch.cat((static["ids"].to(torch.float32), rows)).cpu().tolist()
        return self._unpack(packed, steps, width), None

    def _unpack(self, packed, steps, width):
        drafted = [int(value) for value in packed[:steps]]
        self.proposals = []
        for index in range(steps):
            base = steps + index * 2 * width
            probs = packed[base:base + width]
            ids = packed[base + width:base + 2 * width]
            self.proposals.append({int(t): w for t, w in zip(ids, probs) if w > 0})
        return drafted

    def prefill(self, token_ids, target_hidden):
        ids = list(token_ids)
        offset = 0
        while offset < len(ids):
            end = min(offset + 8192, len(ids))
            piece = torch.tensor(ids[offset:end], dtype=torch.long).view(1, -1)
            self._block(self._fuse(piece, target_hidden[:, offset:end]))
            offset = end

    def step(self, token_id, hidden):
        drafted, state = self.roll(token_id, hidden, 1)
        return drafted[0], state

    def roll(self, token_id, hidden, steps, sampling=None):
        """Six draft ids stay on device. One copy back, not one sync per token.

        Without ``sampling`` each step is the argmax. With it, each step draws
        from the served transform (temperature, top_k, top_p) of the proposer
        logits, and ``self.proposals`` keeps each draw's distribution for the
        verifier's ratio test. Ids and distributions come back in one copy.
        """
        if self._graphable(steps, sampling):
            return self._roll_graphed(token_id, hidden, steps, sampling)
        device = self.table.device
        if self._token is None or self._token.device != device:
            self._token = torch.empty(1, 1, dtype=torch.long, device=device)
        if self._ids is None or self._ids.numel() < steps or self._ids.device != device:
            self._ids = torch.empty(steps, dtype=torch.long, device=device)
        token = self._token
        token.fill_(int(token_id))
        carry = hidden
        # Six argmaxes per window. The 65536-row slice is the production
        # proposer; the full 248320-row head stays on the target verifier.
        layer, id_map = _proposer_head(self.catalog)
        rows = []
        for index in range(steps):
            carry = self._block(self._fuse(token, carry))
            logits = layer.forward(carry, {}).reshape(-1)
            if sampling is None:
                local = logits.argmax()
            else:
                local, row = self._draw(logits, sampling, id_map)
                rows.append(row)
            nxt = local if id_map is None else id_map[local]
            self._ids.narrow(0, index, 1).copy_(nxt.reshape(1))
            token.view(-1).copy_(nxt.reshape(1))
        if sampling is None:
            self.proposals = None
            return self._ids[:steps].tolist(), carry
        packed = torch.cat((self._ids[:steps].to(torch.float32), torch.cat(rows))).cpu().tolist()
        return self._unpack(packed, steps, sampling["top_k"]), carry

    @staticmethod
    def _generator(sampling, device):
        generator = sampling.get("generator")
        if generator is None or generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(sampling.get("seed", 0))
            sampling["generator"] = generator
        return generator

    def _draw(self, logits, sampling, id_map):
        """One Gumbel-max draw from the nucleus of the top_k proposer logits."""
        generator = self._generator(sampling, logits.device)
        noise = torch.rand(sampling["top_k"], generator=generator, device=logits.device)
        return self._pick(logits, sampling, id_map, noise)

    @staticmethod
    def _pick(logits, sampling, id_map, noise):
        top, local_ids = logits.float().topk(sampling["top_k"])
        weights = ((top - top[0]) / sampling["temperature"]).exp()
        cumulative = weights.cumsum(0) / weights.sum()
        # The first token always stays; built on the device (a Python scalar
        # store would copy from the host during graph capture).
        kept = torch.cat((torch.ones_like(cumulative[:1], dtype=torch.bool), cumulative[1:] <= sampling["top_p"]))
        weights = torch.where(kept, weights, torch.zeros_like(weights))
        probs = weights / weights.sum()
        gumbel = -(-noise.clamp(1e-20, 1.0).log()).log()
        choice = torch.argmax(probs.log() + gumbel).reshape(1)
        global_ids = local_ids if id_map is None else id_map.index_select(0, local_ids)
        return local_ids.index_select(0, choice), torch.cat((probs, global_ids.to(torch.float32)))

    def propose(self, prefill_ids, target_hidden, next_id, steps=DRAFT_TOKENS):
        self.reset()
        hidden = target_hidden if target_hidden.dtype == torch.float16 else target_hidden.to(torch.float16)
        blank = torch.zeros_like(hidden[:, :1])
        shifted = torch.cat((blank, hidden[:, :-1]), dim=1)
        self.prefill(prefill_ids, shifted)
        drafted, _carry = self.roll(next_id, hidden[:, -1:], steps)
        return drafted
