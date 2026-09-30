"""One-row decode attention for the MTP proposer, graph-safe.

Keys and values may be FP16 or FP8 (e4m3); FP8 halves the cache read that
dominates a draft step at long context.

The draft step attends one query row against every cached key. SDPA needs
the key length as a Python shape, so a CUDA graph cannot replay it at a new
position. This split-K kernel reads the length from a device tensor: each
program covers one KV head (its six query heads padded to 16 rows) over one
slice of the cache, and a second pass merges the slices by log-sum-exp.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

SPLITS = 32
BLOCK_N = 64
GROUP_ROWS = 16


@triton.jit
def _partial_kernel(
    q_ptr, k_ptr, v_ptr, length_ptr, part_ptr, stat_ptr,
    stride_kt, stride_kh, scale,
    GROUP: tl.constexpr, ROWS: tl.constexpr, DIM: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    head = tl.program_id(0)
    split = tl.program_id(1)
    length = tl.load(length_ptr)
    chunk = tl.cdiv(tl.cdiv(length, SPLITS), BLOCK_N) * BLOCK_N
    start = split * chunk
    end = tl.minimum(length, start + chunk)
    rows = tl.arange(0, ROWS)
    dims = tl.arange(0, DIM)
    row_ok = rows < GROUP
    q = tl.load(q_ptr + (head * GROUP + rows)[:, None] * DIM + dims[None, :], mask=row_ok[:, None], other=0.0)
    m = tl.full((ROWS,), float("-inf"), tl.float32)
    l = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, DIM), tl.float32)
    for n in range(start, end, BLOCK_N):
        cols = n + tl.arange(0, BLOCK_N)
        col_ok = cols < end
        k = tl.load(k_ptr + cols[:, None] * stride_kt + head * stride_kh + dims[None, :],
                    mask=col_ok[:, None], other=0.0).to(tl.float16)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(col_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + cols[:, None] * stride_kt + head * stride_kh + dims[None, :],
                    mask=col_ok[:, None], other=0.0).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m = m_new
    base = (head * SPLITS + split) * ROWS
    tl.store(part_ptr + (base + rows)[:, None] * DIM + dims[None, :], acc)
    tl.store(stat_ptr + (base + rows) * 2, m)
    tl.store(stat_ptr + (base + rows) * 2 + 1, l)


@triton.jit
def _merge_kernel(part_ptr, stat_ptr, out_ptr, GROUP: tl.constexpr, ROWS: tl.constexpr,
                  DIM: tl.constexpr, SPLITS: tl.constexpr):
    head = tl.program_id(0)
    row = tl.program_id(1)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, DIM)
    index = (head * SPLITS + splits) * ROWS + row
    m = tl.load(stat_ptr + index * 2)
    l = tl.load(stat_ptr + index * 2 + 1)
    top = tl.max(m, 0)
    weight = tl.where(m == float("-inf"), 0.0, tl.exp(m - top))
    total = tl.sum(weight * l, 0)
    part = tl.load(part_ptr + index[:, None] * DIM + dims[None, :])
    out = tl.sum(part * weight[:, None], 0) / total
    tl.store(out_ptr + (head * GROUP + row) * DIM + dims, out.to(tl.float16))


class DecodeAttention:
    """Reusable workspace; ``attend`` launches two kernels and allocates nothing."""

    def __init__(self, heads, kv_heads, dim, device):
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim = dim
        self.group = heads // kv_heads
        self.part = torch.empty(kv_heads * SPLITS * GROUP_ROWS, dim, dtype=torch.float32, device=device)
        self.stat = torch.empty(kv_heads * SPLITS * GROUP_ROWS, 2, dtype=torch.float32, device=device)

    def attend(self, query, k_store, v_store, length, out):
        """``query`` [1,1,H,D]; stores [1,T,KV,D]; ``length`` int32 [1]; ``out`` [1,1,H,D]."""
        scale = self.dim ** -0.5
        k = k_store[0]
        v = v_store[0]
        _partial_kernel[(self.kv_heads, SPLITS)](
            query, k, v, length, self.part, self.stat,
            k.stride(0), k.stride(1), scale,
            GROUP=self.group, ROWS=GROUP_ROWS, DIM=self.dim, SPLITS=SPLITS, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )
        _merge_kernel[(self.kv_heads, self.group)](
            self.part, self.stat, out, GROUP=self.group, ROWS=GROUP_ROWS, DIM=self.dim, SPLITS=SPLITS,
            num_warps=4,
        )
        return out
