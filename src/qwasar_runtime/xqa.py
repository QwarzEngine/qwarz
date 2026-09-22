"""Production NVFP4/XQA attention adapter (F4b candidate, promoted 2026-09-21).

Process-local: the installed exllamav3 package remains unchanged. ``install``
prepends this adapter to the live NVFP4/FP8 dispatch candidate list and
restores it when the context exits.

Routes (validated in results/20260918-xqa-fase3 and the 2026-09-21 37-cell
gate, both arms with the rendezvous fix):

- decode (Q <= 8): FlashInfer XQA over zero-copy 64-token page views of the
  donor 256-token NVFP4 pages, with per-(layer, q_len) CUDA graphs owning the
  append; staging degrades to the eager route on any replay error, and stale
  graphs (fresh cache tensors) are dropped and recaptured.
- prefill Q >= 8192: gather NVFP4 -> FP16 -> repage FP8 -> FlashInfer PRIMS
  (the promoted Phase-2 production route).
- prefill 8 < Q < 8192: gather NVFP4 -> FP16 + Torch Flash SDPA.

Non-NVFP4 attention (no block scales, e.g. the vision tower) is declined and
falls through to the stock dispatch unchanged.

Also provides the loader-level pieces of the candidate stack: ``draft_k8v4``
keeps the MTP draft cache at K8/V4 while the target cache is NVFP4, and
``validate_caches`` asserts the resulting cache geometry.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from typing import Any
import weakref

import torch
import triton
import triton.language as tl


DONOR_PAGE_SIZE = 256
XQA_PAGE_SIZE = 64
XQA_MAX_QUERY = 8
XQA_WORKSPACE_BYTES = 128 << 20
PRIMS_MIN_QUERY = 8192


@triton.jit
def _expand_block_table_kernel(block_table, expanded, n, BLOCK: tl.constexpr):
    """expanded[i, o] = block_table.flat[i] * 4 + o in ONE launch per layer.

    Replaces four torch ops per call (mul + 3 add into the 3-D view): ~64
    host API calls per MTP verify that drained the CUDA launch queue and
    surfaced the generator's sync-point stalls (qf3-prof3/qf3-prof4-ctrl).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    base = tl.load(block_table + offs, mask=m, other=0).to(tl.int32) * 4
    o = tl.arange(0, 4)
    vals = base[:, None] + o[None, :]
    tl.store(expanded + offs[:, None] * 4 + o[None, :], vals, mask=m[:, None])


@triton.jit
def _post_lengths_kernel(seqlens, post, append, n, BLOCK: tl.constexpr):
    """post[i] = seqlens[i] + append in ONE launch (was copy_ + add_)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(seqlens + offs, mask=m, other=0) + append
    tl.store(post + offs, v, mask=m)


@triton.jit
def _gather_nvfp4_kernel(
    packed,
    scales,
    block_table,
    seq_lens,
    output,
    max_tokens,
    table_width,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather donor NHD pages in logical order and dequantize one FP4 head."""
    pid_t = tl.program_id(0)
    head = tl.program_id(1)
    batch = pid_t // max_tokens
    token = pid_t - batch * max_tokens
    valid_token = token < tl.load(seq_lens + batch)
    logical_page = token // page_size
    page_offset = token - logical_page * page_size
    physical_page = tl.load(
        block_table + batch * table_width + logical_page,
        mask=valid_token,
        other=0,
    )

    d = tl.arange(0, BLOCK_D)
    valid = valid_token & (d < head_dim)
    row = (physical_page * page_size + page_offset) * n_heads + head
    byte = tl.load(
        packed + row * (head_dim // 2) + d // 2,
        mask=valid,
        other=0,
    ).to(tl.uint8)
    code = tl.where((d & 1) == 0, byte & 0xF, byte >> 4)
    magnitude_index = code & 7
    magnitude = tl.where(
        magnitude_index == 0,
        0.0,
        tl.where(
            magnitude_index == 1,
            0.5,
            tl.where(
                magnitude_index == 2,
                1.0,
                tl.where(
                    magnitude_index == 3,
                    1.5,
                    tl.where(
                        magnitude_index == 4,
                        2.0,
                        tl.where(
                            magnitude_index == 5,
                            3.0,
                            tl.where(magnitude_index == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    value = tl.where((code & 8) != 0, -magnitude, magnitude)
    scale = tl.load(
        scales + row * (head_dim // 16) + d // 16,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    destination = ((batch * max_tokens + token) * n_heads + head) * head_dim + d
    tl.store(output + destination, value * scale, mask=valid)


@dataclass
class _LayerState:
    expanded_table: torch.Tensor
    expanded_table_3d: torch.Tensor
    post_lengths: torch.Tensor
    outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    graphs: dict[tuple[int, float], _XqaGraph] = field(default_factory=dict)
    graph_calls: Counter = field(default_factory=Counter)


@dataclass
class _XqaGraph:
    """Per-(layer, q_len) CUDA graph for the NVFP4 decode route.

    BCAttn declines CacheLayer_nvfp4, so the candidate's attention runs
    eager (~10 host API calls per layer per verify) and drains the launch
    queue (qf3-prof3/4: bimodal verify cycles once the queue empties).
    The graph binds the layer's persistent cache tensors, expanded table,
    post-lengths and output buffer; q/k/v/seqlens stage through statics
    that are copied before each replay. Validated bit-exact in
    qf3-graph1 (grown lengths, remapped tail page, append bytes).

    The block table is NOT bound: the generator rebuilds it every verify
    cycle (qf3-smoke16: a drop per replay), so it stages through
    ``static_table`` exactly like q/k/v/seqlens. Only the per-session cache
    tensors are bound, via WEAK references: a generator cache reset
    (fresh_generator) rebuilds them once per cell, and replaying a graph
    against freed memory aborts with an illegal access (qf3-smoke15).
    Identity is validated on every replay; a mismatch drops the graph and
    recaptures against the new tensors. Weakrefs also keep the stale
    graphs from pinning gigabytes of dead cache across the 37 cells.
    """

    graph: Any
    static_q: torch.Tensor
    static_k: torch.Tensor
    static_v: torch.Tensor
    static_seqlens: torch.Tensor
    static_table: torch.Tensor
    output: torch.Tensor
    bound_caches: tuple | None = None
    failed: bool = False

    def matches(self, args) -> bool:
        if self.failed or self.bound_caches is None:
            return False
        return (
            self.bound_caches[0]() is args.k_cache
            and self.bound_caches[1]() is args.v_cache
            and self.bound_caches[2]() is args.k_scales
            and self.bound_caches[3]() is args.v_scales
        )


@dataclass
class _XqaRawPlan:
    """Pre-resolved launch plan for the module-level xqa closure.

    The high-level flashinfer API costs ~0.4 ms of host time per layer
    (two Python wrappers, asserts, custom-op dispatch, workspace slicing),
    ~6.6 ms per MTP verify across 16 layers, which erased the XQA kernel
    gain end-to-end (smoke qf3-smoke10: -20% decode at 32K, parity at 256K).
    The module closure skips both wrappers and keeps the same JIT artifact;
    numerics are identical because the kernel and its arguments are not
    changed, only how they are passed.
    """

    module: Any
    sm_count: int
    q_scale: float
    semaphores: torch.Tensor
    scratch: torch.Tensor


@dataclass
class _DeviceState:
    workspace: torch.Tensor
    masks: dict[tuple[int, int], torch.Tensor | None] = field(default_factory=dict)
    xqa_plans: dict[tuple[int, float], _XqaRawPlan] = field(default_factory=dict)
    scratch_k: torch.Tensor | None = None
    scratch_v: torch.Tensor | None = None
    scratch_shape: tuple[int, int, int, int] | None = None
    # PRIMS prefill buffers: FP8 pages of 128 tokens plus the FlashInfer wrapper.
    k8: torch.Tensor | None = None
    v8: torch.Tensor | None = None
    prims_workspace: torch.Tensor | None = None
    prims_wrapper: Any = None
    prims_plan_key: tuple[int, int, float] | None = None


class NVFP4AttentionAdapter:
    """One process-local attention candidate with explicit append ownership."""

    def __init__(self, mode: str, flashinfer_module: Any, prefill: str = "sdpa",
                 decode_graphs: bool = True):
        if mode not in ("nvfp4-triton", "nvfp4-xqa"):
            raise ValueError(f"unsupported mode: {mode}")
        if prefill not in ("sdpa", "prims"):
            raise ValueError(f"unsupported prefill route: {prefill}")
        self.mode = mode
        self.prefill = prefill
        self.decode_graphs = decode_graphs
        self.flashinfer = flashinfer_module
        self.prims_audit: dict[int, torch.Tensor] = {}
        self.routes: Counter[str] = Counter()
        self.query_rows: Counter[str] = Counter()
        self.layer_states: dict[tuple[int, tuple[int, ...]], _LayerState] = {}
        self.device_states: dict[torch.device, _DeviceState] = {}
        self.max_observed_query = 0
        self.max_flash_total = 0
        self.append_calls = 0
        self._donor_triton = None

    def _device_state(self, device: torch.device) -> _DeviceState:
        device = torch.device(device)
        state = self.device_states.get(device)
        if state is None:
            state = _DeviceState(
                workspace=torch.zeros(
                    XQA_WORKSPACE_BYTES, dtype=torch.uint8, device=device
                ),
            )
            self.device_states[device] = state
        return state

    def _layer_state(self, args, expand: bool = True) -> tuple[_LayerState, _DeviceState]:
        device_state = self._device_state(args.q.device)
        # Holding the tensor identity in the key prevents accidental state reuse
        # if CUDA later recycles a freed allocation at the same data pointer.
        key = (id(args.k_cache), tuple(args.block_table.shape))
        state = self.layer_states.get(key)
        if state is None:
            # fresh_generator rebuilds the caches per cell; without a cap the
            # dead sessions' states (buffers + graph statics) accumulate
            # across the 37-cell matrix. 16 layers/session -> 64 keeps ~4
            # sessions and evicts the oldest first.
            while len(self.layer_states) >= 64:
                self.layer_states.pop(next(iter(self.layer_states)))
            shape = (
                args.block_table.shape[0],
                args.block_table.shape[1] * 4,
            )
            expanded = torch.empty(shape, dtype=torch.int32, device=args.q.device)
            state = _LayerState(
                expanded_table=expanded,
                expanded_table_3d=expanded.view(
                    args.block_table.shape[0], args.block_table.shape[1], 4
                ),
                post_lengths=torch.empty_like(args.cache_seqlens),
            )
            self.layer_states[key] = state
        if expand:
            rows = args.block_table.shape[0] * args.block_table.shape[1]
            _expand_block_table_kernel[(triton.cdiv(rows, 256),)](
                args.block_table, state.expanded_table, rows, 256, num_warps=1
            )
        return state, device_state

    @staticmethod
    def _validate_nvfp4(args) -> int:
        if args.bsz != 1:
            raise ValueError("Qwasar NVFP4 adapter is deliberately restricted to batch size 1")
        if args.is_varlen() or args.non_causal_spans is not None:
            raise ValueError("Qwasar NVFP4 adapter does not support varlen or non-causal spans")
        if not args.causal:
            raise ValueError("Qwasar NVFP4 adapter requires causal attention")
        if args.window_size not in (None, -1):
            raise ValueError("Qwasar full-attention layers must not use a sliding window")
        if args.softcap not in (None, 0, 0.0) or args.sinks is not None:
            raise ValueError("Qwasar adapter does not support softcap or attention sinks")
        if args.q.dtype != torch.float16 or not args.q.is_contiguous():
            raise ValueError("XQA experiment requires contiguous FP16 queries")
        if args.dim != 256 or args.num_q_heads != 24 or args.num_kv_heads != 4:
            raise ValueError("XQA experiment is pinned to Q24/KV4/D256")
        if args.k_cache.dtype != torch.uint8 or args.v_cache.dtype != torch.uint8:
            raise ValueError("NVFP4 packed caches must be uint8")
        if args.k_scales is None or args.v_scales is None:
            raise ValueError("NVFP4 block scales are required")
        expected_cache = (
            args.k_cache.shape[0], DONOR_PAGE_SIZE, args.num_kv_heads, args.dim // 2
        )
        expected_scales = (
            args.k_cache.shape[0], DONOR_PAGE_SIZE, args.num_kv_heads, args.dim // 16
        )
        if tuple(args.k_cache.shape) != expected_cache or tuple(args.v_cache.shape) != expected_cache:
            raise ValueError(f"unexpected donor NVFP4 cache shape: {tuple(args.k_cache.shape)}")
        if tuple(args.k_scales.shape) != expected_scales or tuple(args.v_scales.shape) != expected_scales:
            raise ValueError(f"unexpected donor NVFP4 scale shape: {tuple(args.k_scales.shape)}")
        if not all(t.is_contiguous() for t in (args.k_cache, args.v_cache, args.k_scales, args.v_scales)):
            raise ValueError("zero-copy 256-to-64 repaging requires contiguous donor caches")
        if args.block_table.dtype != torch.int32 or args.cache_seqlens.dtype != torch.int32:
            raise ValueError("block table and pre-append lengths must be int32")
        if not args.block_table.is_contiguous():
            raise ValueError("flat-addressed table kernels require a contiguous block table")
        if args.k is None or args.v is None:
            raise ValueError("adapter requires current K/V so it can own exactly one append")
        append_len = args.k.shape[1]
        if (
            tuple(args.k.shape) != (args.bsz, append_len, args.num_kv_heads, args.dim)
            or tuple(args.v.shape) != tuple(args.k.shape)
            or append_len != args.q_len
        ):
            raise ValueError("current K/V must match the actual query width")
        if args.block_table.shape[0] != args.bsz:
            raise ValueError("block table batch dimension mismatch")
        return append_len

    @staticmethod
    def _append(args, append_len: int) -> None:
        from exllamav3.modules.attention_fn.triton_paged import (
            _paged_kv_update_nvfp4_kernel,
        )

        grid = (args.bsz * append_len, args.num_kv_heads)
        with torch.cuda.device(args.q.device):
            _paged_kv_update_nvfp4_kernel[grid](
                args.k,
                args.v,
                args.k_cache,
                args.v_cache,
                args.k_scales,
                args.v_scales,
                args.block_table,
                args.cache_seqlens,
                args.block_table.shape[1],
                append_len,
                args.num_kv_heads,
                DONOR_PAGE_SIZE,
                args.dim,
                num_warps=4,
            )

    @staticmethod
    def _post_lengths(state: _LayerState, args, append_len: int) -> torch.Tensor:
        _post_lengths_kernel[(1,)](
            args.cache_seqlens, state.post_lengths, append_len, args.bsz, 16, num_warps=1
        )
        return state.post_lengths

    @staticmethod
    def _mask(
        device_state: _DeviceState, batch: int, width: int, device: torch.device
    ) -> torch.Tensor | None:
        key = (batch, width)
        if key not in device_state.masks:
            if width == 1:
                mask = None
            else:
                row_bits = (
                    (1 << torch.arange(1, width + 1, dtype=torch.int64, device=device))
                    - 1
                ).to(torch.uint32)
                mask = (
                    row_bits.view(torch.uint16)
                    .reshape(1, width, 2)
                    .expand(batch, -1, -1)
                    .contiguous()
                )
            device_state.masks[key] = mask
        return device_state.masks[key]

    def _build_xqa_plan(self, device_state: _DeviceState, args, key) -> _XqaRawPlan:
        # Package-level `flashinfer.xqa` is the re-exported function; import
        # the module symbols directly.
        from flashinfer.utils import get_device_sm_count
        from flashinfer.xqa import get_xqa_module

        module = get_xqa_module(
            args.q.dtype,
            args.k_cache.dtype,
            XQA_PAGE_SIZE,
            args.dim,
            args.num_q_heads // args.num_kv_heads,
            False,
            args.q.dtype,
            args.q_len,
            False,
        )
        workspace_u8 = device_state.workspace.view(torch.uint8)
        plan = _XqaRawPlan(
            module=module,
            sm_count=int(get_device_sm_count(args.q.device)),
            q_scale=float(args.sm_scale) * math.sqrt(args.dim),
            semaphores=workspace_u8[: 8 * 1024 * 1024],
            scratch=workspace_u8[8 * 1024 * 1024 :],
        )
        device_state.xqa_plans[key] = plan
        return plan

    XQA_GRAPH_WARMUPS = 3

    def _capture_xqa_graph(self, state: _LayerState, args) -> _XqaGraph:
        """Side-stream warmup, then capture of append + attention.

        The side-stream warmup and the capture both run through the statics,
        so they append zero rows at the current tail position; the replay in
        the same call overwrites them with the real staged rows (verified in
        qf3-graph1: cache bytes bit-equal after replay).
        """
        static_q = torch.zeros_like(args.q)
        static_k = torch.zeros_like(args.k)
        static_v = torch.zeros_like(args.v)
        static_seqlens = args.cache_seqlens.clone()
        # Clone of the LIVE table, not zeros: the side-stream warmup appends
        # zero rows and the physical tail page must come from a real mapping,
        # otherwise the zeros land in physical page 0 and corrupt live KV.
        static_table = args.block_table.clone()
        staged = args._replace(q=static_q, k=static_k, v=static_v,
                               cache_seqlens=static_seqlens,
                               block_table=static_table)
        try:
            side = torch.cuda.Stream(device=args.q.device)
            side.wait_stream(torch.cuda.current_stream(args.q.device))
            with torch.cuda.stream(side):
                self._append(staged, args.q_len)
                self._xqa(staged, args.q_len)
            torch.cuda.current_stream(args.q.device).wait_stream(side)
            torch.cuda.synchronize(args.q.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._append(staged, args.q_len)
                output = self._xqa(staged, args.q_len)
        except Exception:
            return _XqaGraph(graph=None, static_q=static_q, static_k=static_k,
                             static_v=static_v, static_seqlens=static_seqlens,
                             static_table=static_table, output=None,
                             bound_caches=None, failed=True)
        return _XqaGraph(graph=graph, static_q=static_q, static_k=static_k,
                         static_v=static_v, static_seqlens=static_seqlens,
                         static_table=static_table, output=output,
                         bound_caches=tuple(
                             weakref.ref(t) for t in
                             (args.k_cache, args.v_cache,
                              args.k_scales, args.v_scales)))

    def _xqa(self, args, append_len: int) -> torch.Tensor:
        state, device_state = self._layer_state(args)
        post_lengths = self._post_lengths(state, args, append_len)
        packed_shape = (
            args.k_cache.shape[0] * 4,
            XQA_PAGE_SIZE,
            args.num_kv_heads,
            args.dim // 2,
        )
        scale_shape = (
            args.k_scales.shape[0] * 4,
            XQA_PAGE_SIZE,
            args.num_kv_heads,
            args.dim // 16,
        )
        # Both views are zero-copy: each 256-token physical page becomes four
        # consecutive 64-token XQA pages, addressed by the expanded page table.
        k_cache = args.k_cache.view(packed_shape)
        v_cache = args.v_cache.view(packed_shape)
        k_scales = args.k_scales.view(scale_shape).view(torch.uint8)
        v_scales = args.v_scales.view(scale_shape).view(torch.uint8)
        output = state.outputs.get(args.q_len)
        if output is None:
            output = torch.empty_like(args.q)
            state.outputs[args.q_len] = output
        key = (args.q_len, float(args.sm_scale))
        plan = device_state.xqa_plans.get(key)
        if plan is None:
            plan = self._build_xqa_plan(device_state, args, key)
        # Canonical wrapper transforms: rank-4 Q for single-token decode,
        # rank-5 (beam=1) for speculative widths, matching the validated
        # xqa_batch_decode_with_kv_cache call exactly.
        q_rows = args.q.reshape(-1, args.num_q_heads, args.dim)
        o_rows = output.reshape(-1, args.num_q_heads, args.dim)
        if args.q_len == 1:
            q_new = q_rows.unsqueeze(1)
            o_new = o_rows.unsqueeze(1)
        else:
            q_new = q_rows.view(
                args.bsz, args.q_len, args.num_q_heads, args.dim
            ).unsqueeze(1)
            o_new = o_rows.view(
                args.bsz, args.q_len, args.num_q_heads, args.dim
            ).unsqueeze(1)
        plan.module.xqa(
            False,
            plan.sm_count,
            args.num_kv_heads,
            0,
            plan.q_scale,
            o_new,
            1.0,
            q_new,
            None,
            k_cache,
            v_cache,
            k_scales,
            v_scales,
            state.expanded_table,
            state.expanded_table.shape[1] * XQA_PAGE_SIZE,
            post_lengths.unsqueeze(1),
            args.bsz,
            1.0,
            plan.semaphores,
            plan.scratch,
            False,
            args.q_len,
            None,
            self._mask(device_state, args.bsz, args.q_len, args.q.device),
        )
        return output

    @staticmethod
    def _live_total(lengths: torch.Tensor, append_len: int) -> int:
        # Q>8 prefill is deliberately outside CUDA graphs. Read the authoritative
        # device length on every layer because raw CUDA writes are not reliably
        # visible through host tensor metadata.
        return int((lengths + append_len).max().item())

    @staticmethod
    def _ensure_scratch(device_state: _DeviceState, args) -> tuple[torch.Tensor, torch.Tensor]:
        capacity = args.k_cache.shape[0] * DONOR_PAGE_SIZE
        shape = (args.bsz, capacity, args.num_kv_heads, args.dim)
        if device_state.scratch_shape != shape:
            device_state.scratch_k = torch.empty(
                shape, dtype=torch.float16, device=args.q.device
            )
            device_state.scratch_v = torch.empty_like(device_state.scratch_k)
            device_state.scratch_shape = shape
        assert device_state.scratch_k is not None and device_state.scratch_v is not None
        return device_state.scratch_k, device_state.scratch_v

    def _flash_prefill(self, args, append_len: int) -> torch.Tensor:
        state, device_state = self._layer_state(args)
        self._append(args, append_len)
        self.append_calls += 1
        post_lengths = self._post_lengths(state, args, append_len)
        total = self._live_total(args.cache_seqlens, append_len)
        if total > args.k_cache.shape[0] * DONOR_PAGE_SIZE:
            raise ValueError("logical sequence exceeds the physical NVFP4 cache")
        self.max_flash_total = max(self.max_flash_total, total)
        scratch_k, scratch_v = self._ensure_scratch(device_state, args)
        grid = (args.bsz * total, args.num_kv_heads)
        with torch.cuda.device(args.q.device):
            _gather_nvfp4_kernel[grid](
                args.k_cache,
                args.k_scales,
                args.block_table,
                post_lengths,
                scratch_k,
                total,
                args.block_table.shape[1],
                args.num_kv_heads,
                args.dim,
                DONOR_PAGE_SIZE,
                triton.next_power_of_2(args.dim),
                num_warps=4,
                num_stages=1,
            )
            _gather_nvfp4_kernel[grid](
                args.v_cache,
                args.v_scales,
                args.block_table,
                post_lengths,
                scratch_v,
                total,
                args.block_table.shape[1],
                args.num_kv_heads,
                args.dim,
                DONOR_PAGE_SIZE,
                triton.next_power_of_2(args.dim),
                num_warps=4,
                num_stages=1,
            )

        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.nn.attention.bias import causal_lower_right
        from torch.nn.functional import scaled_dot_product_attention

        query = args.q.transpose(1, 2)
        key = scratch_k[:, :total].transpose(1, 2)
        value = scratch_v[:, :total].transpose(1, 2)
        bias = causal_lower_right(args.q_len, total)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=bias,
                dropout_p=0.0,
                scale=float(args.sm_scale),
                enable_gqa=True,
            )
        return output.transpose(1, 2).contiguous()

    def _prims_prefill(self, args, append_len: int) -> torch.Tensor:
        """Large-query route: gather NVFP4 -> FP16 scratch -> repage FP8 -> PRIMS.

        Mirrors the production K8/V4 PRIMS path (P-scale 256, v_scale 1/256),
        replacing `ext.dequant_cache_paged_window` with the NVFP4 gather and
        reusing `qwasar_bench.prims_prefill.repage` unchanged. Only the FP16
        staging source differs; the FP8 kernel and its numerics are identical
        to the promoted Phase-2 production route.
        """
        from qwasar_bench.prims_prefill import repage

        state, device_state = self._layer_state(args)
        self._append(args, append_len)
        self.append_calls += 1
        post_lengths = self._post_lengths(state, args, append_len)
        total = self._live_total(args.cache_seqlens, append_len)
        if total > args.k_cache.shape[0] * DONOR_PAGE_SIZE:
            raise ValueError("logical sequence exceeds the physical NVFP4 cache")
        self.max_flash_total = max(self.max_flash_total, total)
        scratch_k, scratch_v = self._ensure_scratch(device_state, args)
        grid = (args.bsz * total, args.num_kv_heads)
        with torch.cuda.device(args.q.device):
            _gather_nvfp4_kernel[grid](
                args.k_cache,
                args.k_scales,
                args.block_table,
                post_lengths,
                scratch_k,
                total,
                args.block_table.shape[1],
                args.num_kv_heads,
                args.dim,
                DONOR_PAGE_SIZE,
                triton.next_power_of_2(args.dim),
                num_warps=4,
                num_stages=1,
            )
            _gather_nvfp4_kernel[grid](
                args.v_cache,
                args.v_scales,
                args.block_table,
                post_lengths,
                scratch_v,
                total,
                args.block_table.shape[1],
                args.num_kv_heads,
                args.dim,
                DONOR_PAGE_SIZE,
                triton.next_power_of_2(args.dim),
                num_warps=4,
                num_stages=1,
            )

        if device_state.k8 is None:
            cap = args.k_cache.shape[0] * 2
            device_state.k8 = torch.empty(
                (cap, 4, 128, 256), dtype=torch.float8_e4m3fn, device=args.q.device
            )
            device_state.v8 = torch.empty_like(device_state.k8)
            device_state.prims_workspace = torch.empty(
                16 << 20, dtype=torch.uint8, device=args.q.device
            )
        pages = (total + 127) // 128
        last = (total - 1) % 128 + 1
        n = pages * 4 * 128 * 256
        repage[(triton.cdiv(n, 1024),)](scratch_k, device_state.k8, total, n, 1024)
        repage[(triton.cdiv(n, 1024),)](scratch_v, device_state.v8, total, n, 1024)
        layer_key = int(args.k_cache.data_ptr())
        if layer_key not in self.prims_audit:
            self.prims_audit[layer_key] = torch.stack((
                args.q.abs().amax(),
                scratch_k.view(-1, 4, 256)[:total].abs().amax(),
                scratch_v.view(-1, 4, 256)[:total].abs().amax())).float()
        q8 = args.q[0].clamp(-448, 448).to(torch.float8_e4m3fn)
        key = (args.q_len, total, float(args.sm_scale))
        if device_state.prims_plan_key != key:
            if device_state.prims_wrapper is None:
                device_state.prims_wrapper = self.flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    device_state.prims_workspace, "HND", backend="cute-dsl-prims"
                )
            qo = torch.tensor([0, args.q_len], device=args.q.device, dtype=torch.int32)
            pi = torch.tensor([0, pages], device=args.q.device, dtype=torch.int32)
            indices = torch.arange(pages, device=args.q.device, dtype=torch.int32)
            lasts = torch.tensor([last], device=args.q.device, dtype=torch.int32)
            device_state.prims_wrapper.plan(
                qo, pi, indices, lasts, args.num_q_heads, args.num_kv_heads, args.dim, 128,
                head_dim_vo=args.dim, causal=True, sm_scale=float(args.sm_scale),
                q_data_type=torch.float8_e4m3fn, kv_data_type=torch.float8_e4m3fn,
                o_data_type=torch.float16, block_tables=indices.view(1, -1))
            device_state.prims_plan_key = key
        y = device_state.prims_wrapper.run(
            q8, (device_state.k8, device_state.v8), q_scale=1., k_scale=1.,
            v_scale=1 / 256, enable_pdl=False).unsqueeze(0)
        return y

    def prims_saturation_free(self, limit: float = 448.0) -> bool | None:
        """True when every audited PRIMS operand stayed inside E4M3 range."""
        if not self.prims_audit:
            return None
        return all(float(stack.max()) < limit for stack in self.prims_audit.values())

    def __call__(self, args):
        # Preserve the donor's unrelated raw-FP8 behavior.
        if args.k_scales is None:
            return None
        append_len = self._validate_nvfp4(args)
        self.max_observed_query = max(self.max_observed_query, args.q_len)
        if args.q_len <= XQA_MAX_QUERY:
            if self.mode == "nvfp4-triton":
                if self._donor_triton is None:
                    from exllamav3.modules.attention_fn.triton_paged import (
                        fn_triton_paged_attn,
                    )
                    self._donor_triton = fn_triton_paged_attn
                output = self._donor_triton(args)
                if output is None:
                    raise RuntimeError("donor Triton declined a pinned NVFP4 short query")
                self.routes["triton_short"] += 1
                self.query_rows["triton_short"] += args.q_len
                self.append_calls += 1
                return output
            if self.decode_graphs and not torch.cuda.is_current_stream_capturing():
                # The recorded graph owns the table expansion; skip the eager one.
                state, _device_state = self._layer_state(args, expand=False)
                key = (args.q_len, float(args.sm_scale))
                g = state.graphs.get(key)
                if g is not None and not g.matches(args):
                    # The generator rebuilt the cache tensors (fresh_generator):
                    # the graph's bindings are stale and replaying them is an
                    # illegal access (qf3-smoke15). Drop and recapture.
                    del state.graphs[key]
                    state.graph_calls[key] = 0
                    self.routes["xqa_graph_dropped"] += 1
                    g = None
                if g is None:
                    state.graph_calls[key] += 1
                    if state.graph_calls[key] < self.XQA_GRAPH_WARMUPS:
                        self._append(args, append_len)
                        self.append_calls += 1
                        output = self._xqa(args, append_len)
                        self.routes["xqa_short"] += 1
                        self.query_rows["xqa_short"] += args.q_len
                        return output
                    g = self._capture_xqa_graph(state, args)
                    state.graphs[key] = g
                    self.routes[
                        "xqa_graph_failed" if g.failed else "xqa_graph_captured"
                    ] += 1
                if g.failed or g.graph is None:
                    # The failed capture attempt left zero rows at the tail;
                    # the real append below overwrites them.
                    self._append(args, append_len)
                    self.append_calls += 1
                    output = self._xqa(args, append_len)
                    self.routes["xqa_short"] += 1
                    self.query_rows["xqa_short"] += args.q_len
                    return output
                # Stage the varying inputs, then one graph replay (~4 host API
                # calls per layer per verify instead of ~10 eager ones). The
                # table stages too (the generator rebuilds it each cycle), and
                # the graph owns the append. Any staging/replay error degrades
                # to the eager route rather than failing the whole run.
                try:
                    g.static_q.copy_(args.q)
                    g.static_k.copy_(args.k)
                    g.static_v.copy_(args.v)
                    g.static_seqlens.copy_(args.cache_seqlens)
                    g.static_table.copy_(args.block_table)
                    g.graph.replay()
                except Exception:
                    del state.graphs[key]
                    state.graph_calls[key] = 0
                    self.routes["xqa_graph_error"] += 1
                    self._append(args, append_len)
                    self.append_calls += 1
                    output = self._xqa(args, append_len)
                    self.routes["xqa_short"] += 1
                    self.query_rows["xqa_short"] += args.q_len
                    return output
                self.append_calls += 1
                self.routes["xqa_graph"] += 1
                self.query_rows["xqa_graph"] += args.q_len
                return g.output
            self._append(args, append_len)
            self.append_calls += 1
            output = self._xqa(args, append_len)
            self.routes["xqa_short"] += 1
            self.query_rows["xqa_short"] += args.q_len
            return output
        if self.prefill == "prims" and args.q_len >= PRIMS_MIN_QUERY:
            output = self._prims_prefill(args, append_len)
            self.routes["prims_prefill"] += 1
            self.query_rows["prims_prefill"] += args.q_len
            return output
        output = self._flash_prefill(args, append_len)
        self.routes["flash_prefill"] += 1
        self.query_rows["flash_prefill"] += args.q_len
        return output

    def diagnostics(self) -> dict[str, Any]:
        scratch_bytes = sum(
            0
            if state.scratch_k is None
            else 2 * state.scratch_k.numel() * state.scratch_k.element_size()
            for state in self.device_states.values()
        )
        return {
            "mode": self.mode,
            "prefill": self.prefill,
            "contract": {
                "target_cache": "one-level NVFP4 global=1, donor NHD page256",
                "xqa_cache": "zero-copy NHD page64 views",
                "input_lengths": "pre-append int32",
                "xqa_lengths": "post-append = pre-append + actual K rows",
                "append_owner": "selected adapter route exactly once",
                "short_query": "Q1..8",
                "prefill": ("Q>=8192 gather to FP16 + repage FP8 + FlashInfer PRIMS (P-scale 256); "
                            "8<Q<8192 gather + Torch Flash SDPA lower-right causal"
                            if self.prefill == "prims" else
                            "Q>8 donor-NVFP4 gather to reusable FP16 + Torch Flash SDPA lower-right causal"),
            },
            "routes": dict(self.routes),
            "query_rows": dict(self.query_rows),
            "append_calls": self.append_calls,
            "max_observed_query": self.max_observed_query,
            "max_flash_total": self.max_flash_total,
            "layer_states": len(self.layer_states),
            "device_states": len(self.device_states),
            "xqa_workspace_bytes_per_device": XQA_WORKSPACE_BYTES,
            "flash_scratch_bytes": scratch_bytes,
            "prims_layers_audited": len(self.prims_audit),
            "prims_saturation_free": self.prims_saturation_free(),
        }


@contextmanager
def install(mode: str, flashinfer_module: Any, prefill: str = "sdpa"):
    """Install the adapter into the live donor dispatcher and restore on exit."""
    from exllamav3.modules.attention_fn import dispatch

    runtime = NVFP4AttentionAdapter(mode, flashinfer_module, prefill)
    original = list(dispatch._fns_fp8)
    dispatch._fns_fp8[:] = [runtime, *original]
    try:
        yield runtime
    finally:
        dispatch._fns_fp8[:] = original


REVISION = "xqa-20260921-promoted"


@contextmanager
def draft_k8v4():
    """Force the MTP draft cache to K8/V4 while the target cache is NVFP4.

    With ``--cache_quant nvfp4`` the loader would build BOTH caches as NVFP4;
    the draft must stay K8/V4 (Attention64 + the production draft route).
    The first Cache call is the target (asserted NVFP4), the second is the
    draft (overridden to CacheLayer_quant 8/4); any further Cache calls pass
    through untouched (production tolerance: the smoke-time version asserted
    exactly two calls).
    """
    from exllamav3 import model_init
    from exllamav3.cache import CacheLayer_nvfp4, CacheLayer_quant

    original = model_init.Cache
    calls = []

    def factory(*args, **kwargs):
        index = len(calls)
        if index == 0:
            if kwargs.get("layer_type") is not CacheLayer_nvfp4:
                raise RuntimeError("first Cache call is not NVFP4 target")
            calls.append("target-nvfp4")
            return original(*args, **kwargs)
        if index == 1:
            calls.append("draft-k8v4")
            return original(*args, **kwargs, layer_type=CacheLayer_quant, k_bits=8, v_bits=4)
        return original(*args, **kwargs)

    model_init.Cache = factory
    try:
        yield calls
    finally:
        model_init.Cache = original
    if calls[:2] != ["target-nvfp4", "draft-k8v4"]:
        raise RuntimeError(f"unexpected cache calls: {calls}")


def cache_types(cache):
    """Bounded cache geometry summary for assertions and config reporting."""
    names = [type(layer).__name__ for layer in cache.layers.values()]
    bits = sorted({
        (getattr(layer, "k_bits", None), getattr(layer, "v_bits", None))
        for layer in cache.layers.values()
    })
    return {
        "layer_type": cache.layer_type.__name__,
        "count": len(names),
        "unique": sorted(set(names)),
        "bits": bits,
    }


def validate_caches(generator) -> dict:
    """Assert the candidate cache geometry; returns the summaries."""
    target = cache_types(generator.cache)
    draft = cache_types(generator.draft_cache)
    if set(target["unique"]) != {"CacheLayer_nvfp4"} or target["layer_type"] != "CacheLayer_nvfp4":
        raise RuntimeError(f"target cache {target}")
    if set(draft["unique"]) != {"CacheLayer_quant"} or draft["bits"] != [(8, 4)]:
        raise RuntimeError(f"draft cache {draft}")
    return {"target": target, "draft": draft}
