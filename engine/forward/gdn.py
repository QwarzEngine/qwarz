"""Call the ported GDN recurrence and the certified ExLlama extension on the same inputs.

Qwen3.8 uses 16 key heads and 48 value heads of dimension 128, which selects
the specialized kernel and a value split of 4. Both launches must produce
the same bytes, and a second launch from the same state must repeat them.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext

LIBRARY = Path(__file__).resolve().parents[1] / "cuda" / "libq38_gdn.so"
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_DIM = 128


def _load():
    library = ctypes.CDLL(str(LIBRARY))
    library.q38_gdn_recurrent.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ]
    library.q38_gdn_recurrent.restype = None
    library.q38_gdn_fused_op_2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    library.q38_gdn_fused_op_2.restype = None
    library.q38_gdn_conv1d.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    library.q38_gdn_conv1d.restype = None
    return library


def _inputs(seqlen, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qkv_dim = 2 * NUM_K_HEADS * HEAD_DIM + NUM_V_HEADS * HEAD_DIM
    mixed_qkv = torch.randn(1, seqlen, qkv_dim, generator=generator, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(1, seqlen, NUM_V_HEADS, generator=generator, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, seqlen, NUM_V_HEADS, generator=generator, device="cuda", dtype=torch.bfloat16)
    return mixed_qkv, g, beta


def _state():
    return torch.zeros(1, 1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.float32)


def _out(seqlen):
    return torch.empty(1, seqlen, NUM_V_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)


def q38(library, mixed_qkv, g, beta, state, out):
    library.q38_gdn_recurrent(
        mixed_qkv.data_ptr(), g.data_ptr(), beta.data_ptr(), state.data_ptr(), out.data_ptr(),
        mixed_qkv.size(0), mixed_qkv.size(1), NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM,
        None, state.size(1), 0,
    )
    torch.cuda.synchronize()


def certified(mixed_qkv, g, beta, state, out):
    ext.cuda_recurrent_gated_delta_rule(
        mixed_qkv, g, beta, state, out,
        NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, None, False,
    )
    torch.cuda.synchronize()


def compare(seqlen=63, seed=20260928):
    library = _load()
    mixed_qkv, g, beta = _inputs(seqlen, seed)
    ours = _out(seqlen)
    theirs = _out(seqlen)
    q38(library, mixed_qkv, g, beta, _state(), ours)
    certified(mixed_qkv, g, beta, _state(), theirs)
    again = _out(seqlen)
    q38(library, mixed_qkv, g, beta, _state(), again)
    return {
        "seqlen": seqlen,
        "matches_certified": torch.equal(ours, theirs),
        "repeats": torch.equal(ours, again),
    }


def compare_fused(seqlen=63, seed=20260928):
    library = _load()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    b = torch.randn(1, seqlen, NUM_V_HEADS, generator=generator, device="cuda")
    a = torch.randn(1, seqlen, NUM_V_HEADS, generator=generator, device="cuda")
    dt_bias = torch.randn(NUM_V_HEADS, generator=generator, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(NUM_V_HEADS, generator=generator, device="cuda")
    ours_beta = torch.empty_like(b, dtype=torch.bfloat16)
    ours_g = torch.empty_like(b)
    theirs_beta = torch.empty_like(ours_beta)
    theirs_g = torch.empty_like(ours_g)
    library.q38_gdn_fused_op_2(
        b.data_ptr(), a.data_ptr(), dt_bias.data_ptr(), a_log.data_ptr(),
        ours_beta.data_ptr(), ours_g.data_ptr(), 1, seqlen, NUM_V_HEADS, 1.0,
    )
    ext.gated_delta_net_fused_op_2(b, a, dt_bias, a_log, theirs_beta, theirs_g, 1.0)
    torch.cuda.synchronize()
    return {
        "seqlen": seqlen,
        "beta": torch.equal(ours_beta, theirs_beta),
        "g": torch.equal(ours_g, theirs_g),
    }


def compare_conv(seqlen=8, kernel=4, seed=20260928):
    library = _load()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    dim = 2 * NUM_K_HEADS * HEAD_DIM + NUM_V_HEADS * HEAD_DIM
    x = torch.randn(1, dim, seqlen, generator=generator, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, kernel, generator=generator, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, generator=generator, device="cuda", dtype=torch.bfloat16)
    ours_state = torch.zeros(1, dim, kernel, device="cuda", dtype=torch.bfloat16)
    theirs_state = torch.zeros_like(ours_state)
    ours = torch.empty(1, seqlen, dim, device="cuda", dtype=torch.bfloat16)
    theirs = torch.empty_like(ours)
    library.q38_gdn_conv1d(
        x.data_ptr(), ours_state.data_ptr(), weight.data_ptr(), bias.data_ptr(), ours.data_ptr(),
        1, dim, seqlen, kernel, kernel, 1,
    )
    ext.cuda_causal_conv1d_update(x, theirs_state, None, weight, bias, theirs, True, False)
    torch.cuda.synchronize()
    return {
        "seqlen": seqlen,
        "out": torch.equal(ours, theirs),
        "state": torch.equal(ours_state, theirs_state),
    }


if __name__ == "__main__":
    import json
    print(json.dumps({
        "recurrent": {"short": compare(8), "prefill": compare(63)},
        "fused": compare_fused(63),
        "conv": compare_conv(8),
    }))
