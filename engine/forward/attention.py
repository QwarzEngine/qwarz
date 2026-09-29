"""Layer-3 full attention on the 63-token oracle prefill.

Layers 0, 1 and 2 are Gated DeltaNet plus the NVFP4 MLP. Their residual
is the input of layer 3. Query carries an interleaved gate. Q and K take
a per-head RMSNorm with bias +1 inside partial RoPE (64 of 256 dims).
K and V are written into one NVFP4 page and gathered back before the
lower-right causal SDPA. The sigmoid gate and the EXL3 output projection
follow.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn.common import AttnArgs
from exllamav3.util.rope import RoPE, RopeSettings, RopeStyle
from safetensors import safe_open

from engine.forward.embed import ORACLE_GDN_INPUT, RMS_EPS
from engine.forward.mlp import (
    HIDDEN,
    ORACLE_MLP_INPUT,
    ORACLE_MLP_OUTPUT,
    _runtime,
    block_input,
    load_linears,
    mlp_forward,
    mlp_input,
)
from engine.forward.projections import (
    ORACLE_GDN_OUTPUT,
    cached_exl3,
    gdn_forward,
    load_layer,
    sha256,
)

ORACLE_ATTN_INPUT = "9c6c75b63232eedeec6bb8af50a9d6fff28c6987a5a24a3ef83d1dc2bf63cabf"
ORACLE_ATTN_OUTPUT = "3686b7513f0011dc127054a392ffd09dc6a30c83561e8f3baf12772f197e4f7e"
ORACLE_K = "9b6f111139d84348ce8d4b178df1f7b5d2a1d0182682f009fe4470108e9c7556"
ORACLE_V = "d62235c6dff2c122cf5ebb013fd51a5146e3317f41e1a4dc1f9fb53edda4d835"
ORACLE_K_SCALE = "8691d32d5d62558ea677e49afad979756c65912f0897281725627c4df20e2a5e"
ORACLE_V_SCALE = "6ee2defcbdd8d740a86cafa813d6367e62060df98a08badb62b844bcfe3f7555"
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
Q_OUT = HEADS * HEAD_DIM * 2
KV_OUT = KV_HEADS * HEAD_DIM
PAGE = 256
LAYER = 3


def input_norm(model, index, residual):
    with safe_open(model / "model-00001-of-00003.safetensors", framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"model.language_model.layers.{index}.input_layernorm.weight")
    normed = torch.empty(residual.shape, dtype=torch.float16, device=residual.device)
    ext.rms_norm(
        residual.reshape(-1, HIDDEN).contiguous(),
        weight.cuda().contiguous(),
        normed.reshape(-1, HIDDEN),
        RMS_EPS,
        1.0,
        1.0,
        False,
        False,
    )
    return normed


def recurrent_stack(model, donor):
    """Three Gated DeltaNet blocks. The returned residual is the layer-3 input."""
    residual = block_input(model)
    for index in range(LAYER):
        normed = input_norm(model, index, residual)
        if index == 0 and sha256(normed) != ORACLE_GDN_INPUT:
            raise RuntimeError("layer 0 norm no longer matches the oracle")
        attended = gdn_forward(normed, load_layer(model, index))[0]
        if index == 0 and sha256(attended) != ORACLE_GDN_OUTPUT:
            raise RuntimeError("layer 0 GDN no longer matches the oracle")
        entered = mlp_input(model, residual, attended, index)
        if index == 0 and sha256(entered) != ORACLE_MLP_INPUT:
            raise RuntimeError("layer 0 MLP input no longer matches the oracle")
        produced = mlp_forward(entered, load_linears(donor, index))
        if index == 0 and sha256(produced) != ORACLE_MLP_OUTPUT:
            raise RuntimeError("layer 0 MLP no longer matches the oracle")
        residual = residual + produced
        del normed, attended, entered, produced
        torch.cuda.empty_cache()
    return residual


def load_attention(model):
    prefix = f"model.language_model.layers.{LAYER}.self_attn"
    with safe_open(model / "model-00001-of-00003.safetensors", framework="pt", device="cpu") as handle:
        names = [key for key in handle.keys() if key.startswith(prefix)]
        return {name[len(prefix) + 1:]: handle.get_tensor(name).cuda().contiguous() for name in names}


def project(hidden, weights, name, out_features, in_features, dtype):
    layer = cached_exl3(
        weights,
        name,
        weights[f"{name}.trellis"],
        weights[f"{name}.suh"],
        weights[f"{name}.svh"],
        weights[f"{name}.mul1"],
        out_features,
        in_features,
        dtype,
    )
    return layer.forward(hidden, {})


_ROPES = {}


def rope(device="cuda"):
    key = str(torch.device(device))
    cached = _ROPES.get(key)
    if cached is None:
        settings = RopeSettings(
            head_dim=HEAD_DIM,
            rope_theta=10_000_000,
            partial_rotary_factor=0.25,
            rope_scaling={
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
            rope_style=RopeStyle.NEOX,
        )
        cached = RoPE(device, settings)
        _ROPES[key] = cached
    return cached


_ZERO_LENS = {}


def cache_seqlens(cache_len, device):
    """One int32 length. A tensor is already on device and is reused by a CUDA graph."""
    if isinstance(cache_len, int):
        if cache_len == 0:
            # The cold prefill graph cannot allocate this scalar while it is recording.
            key = str(device)
            tensor = _ZERO_LENS.get(key)
            if tensor is None:
                tensor = torch.zeros(1, dtype=torch.int32, device=device)
                _ZERO_LENS[key] = tensor
            return tensor, 0
        return torch.tensor([cache_len], dtype=torch.int32, device=device), cache_len
    return cache_len, None


def empty_cache(pages=1):
    packed = (pages, PAGE, KV_HEADS, HEAD_DIM // 2)
    scales = (pages, PAGE, KV_HEADS, HEAD_DIM // 16)
    return (
        torch.zeros(packed, dtype=torch.uint8, device="cuda"),
        torch.zeros(packed, dtype=torch.uint8, device="cuda"),
        torch.zeros(scales, dtype=torch.float8_e4m3fn, device="cuda"),
        torch.zeros(scales, dtype=torch.float8_e4m3fn, device="cuda"),
    )


_ADAPTER = None
_BLOCK_TABLES = {}
_BOUND_PAGES = None


def bind_table_pages(pages):
    """Publish only the first ``pages`` identity entries on the next attend.

    XQA takes its max sequence from the table width. A 7-token verify on a
    1024-page cache would otherwise walk the whole 262144-token allocation.
    ``None`` restores the full table. The caller clears this before returning.
    """
    global _BOUND_PAGES
    _BOUND_PAGES = pages


def _block_table(pages, device):
    key = (pages, str(device))
    table = _BLOCK_TABLES.get(key)
    if table is None:
        table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
        _BLOCK_TABLES[key] = table
    return table


def release_attention_plans():
    """Drop per-cache attention plans. The KV pages themselves live on the runner."""
    global _BOUND_PAGES
    _BOUND_PAGES = None
    if _ADAPTER is not None:
        _ADAPTER.layer_states.clear()


def attend(q, k, v, cache, cache_len=0):
    global _ADAPTER
    _runtime()
    from qwasar_runtime.hybrid import prepare_environment
    from qwasar_runtime.xqa import NVFP4AttentionAdapter

    prepare_environment()
    import flashinfer
    if _ADAPTER is None:
        # Nested graphs cannot record XQA's own graph launch. Decode stays eager
        # here; an outer graph, if one is captured, replays this eager call.
        _ADAPTER = NVFP4AttentionAdapter("nvfp4-xqa", flashinfer, prefill="prims", decode_graphs=False)
    k_cache, v_cache, k_scales, v_scales = cache
    seqlen = q.shape[1]
    pages = k_cache.shape[0]
    # Eager prefill passes a Python length. The flash path would otherwise
    # copy that same length back off the device on every layer.
    _ADAPTER._host_cache_len = cache_len if isinstance(cache_len, int) else None
    table = _block_table(pages, q.device)
    bound = _BOUND_PAGES
    if bound is not None and bound < table.shape[1]:
        table = table.narrow(1, 0, bound)
    args = AttnArgs(
        bsz=1,
        q_len=seqlen,
        num_q_heads=HEADS,
        dim=HEAD_DIM,
        kv_len=seqlen,
        num_kv_heads=KV_HEADS,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        k_cache=k_cache,
        v_cache=v_cache,
        causal=True,
        sm_scale=HEAD_DIM ** -0.5,
        cu_seqlens=None,
        max_seqlen=None,
        window_size=None,
        softcap=0.0,
        block_table=table,
        cache_seqlens=cache_seqlens(cache_len, q.device)[0],
        k_scales=k_scales,
        v_scales=v_scales,
    )
    # Q>=8192 gathers NVFP4 and attends with PRIMS. Shorter prefills use SDPA.
    # A single token uses XQA over the same pages.
    return _ADAPTER(args)


def attention_forward(hidden, weights, cache=None, cache_len=0, inv_freq=None):
    if cache is None:
        cache = empty_cache()
    q_packed = project(hidden, weights, "q_proj", Q_OUT, HIDDEN, torch.float16)
    seqlen = hidden.shape[1]
    query = torch.empty((1, seqlen, HEADS, HEAD_DIM), dtype=torch.float16, device=hidden.device)
    gate = torch.empty((1, seqlen, HEADS * HEAD_DIM), dtype=torch.float16, device=hidden.device)
    ext.deinterleave_qg(q_packed, query, gate, HEAD_DIM)
    key = project(hidden, weights, "k_proj", KV_OUT, HIDDEN, torch.float16).view(1, seqlen, KV_HEADS, HEAD_DIM)
    value = project(hidden, weights, "v_proj", KV_OUT, HIDDEN, torch.float16).view(1, seqlen, KV_HEADS, HEAD_DIM)
    # A later chunk starts at cache_len. The first chunk keeps the zero origin.
    # A CUDA graph passes the length tensor and always has a past.
    seqlens, host_len = cache_seqlens(cache_len, hidden.device)
    positions = None if host_len == 0 else seqlens
    query, key = rope().apply(
        query, key, 0, positions, None, True,
        weights["q_norm.weight"], weights["k_norm.weight"], RMS_EPS, 1.0, inv_freq, False,
    )
    mixed = attend(query, key, value, cache, cache_len)
    flat = mixed.reshape(1, seqlen, HEADS * HEAD_DIM)
    ext.mul_sigmoid_(flat, gate)
    output = project(flat, weights, "o_proj", HIDDEN, HEADS * HEAD_DIM, torch.float32)
    return output, cache


def main():
    _runtime()
    from qwasar_runtime.hybrid import donor_dir

    model = Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw"
    residual = recurrent_stack(model, donor_dir())
    hidden = input_norm(model, LAYER, residual)
    output, (k_cache, v_cache, k_scales, v_scales) = attention_forward(hidden, load_attention(model))
    hidden_sha = sha256(hidden)
    output_sha = sha256(output)
    k_sha = sha256(k_cache)
    v_sha = sha256(v_cache)
    ks_sha = sha256(k_scales)
    vs_sha = sha256(v_scales)
    print(json.dumps({
        "input_sha256": hidden_sha,
        "output_sha256": output_sha,
        "matches_input": hidden_sha == ORACLE_ATTN_INPUT,
        "matches_oracle": output_sha == ORACLE_ATTN_OUTPUT,
        "kv_page": list(k_cache.shape),
        "matches_kv": k_sha == ORACLE_K and v_sha == ORACLE_V and ks_sha == ORACLE_K_SCALE and vs_sha == ORACLE_V_SCALE,
    }))


if __name__ == "__main__":
    main()
