"""Layer-0 Gated DeltaNet projections on the oracle activation.

q and the packed k/v come from an EXL3 matrix, 5120 by 10240. z is EXL3,
5120 by 6144. a and b are stored fp16 as [48, 5120] and multiplied as
x @ W.T into float32 with the same hgemm ExLlama's LinearFP16 uses.
The activation is the 63-token prefill the embedding and input norm
already match. Projection bytes and the layer output are compared with
the recorded ExLlama tensors.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.config import NullConfig
from exllamav3.modules.quant.exl3 import LinearEXL3
from safetensors import safe_open

from engine.forward.embed import (
    ORACLE_GDN_INPUT,
    RMS_EPS,
    gather,
    load_weight,
    prefill_ids,
    tensor_sha256,
)
from engine.forward.gdn import HEAD_DIM, NUM_K_HEADS, NUM_V_HEADS

ORACLE_GDN_OUTPUT = "b8b0b6b5b029cca6712a63c5562a68c4cf4f66b74773d746cb2cdee6687114fb"
ORACLE_QKV = "d133a96464876c675d909cc81815cb4d4076a483ee552ea4ede5fd5d09309b8e"
ORACLE_Z = "20e85a7103286b8d349498f080ea64df4a8ac672dd1a4856b34ca797bdec14ab"
ORACLE_A = "b259571caeaa22f0372924bff42ad255fc19080ed36400b7b5197bc347a00fb9"
ORACLE_B = "fae13eab25c164858e3bdcb287165fd11a94ab3b654ffd461ee858675c5e569a"
QKV_OUT = 2 * NUM_K_HEADS * HEAD_DIM + NUM_V_HEADS * HEAD_DIM
Z_OUT = NUM_V_HEADS * HEAD_DIM


def sha256(tensor):
    return tensor_sha256(torch, tensor)


def hidden_state(model):
    weight = load_weight(torch, model)
    ids = prefill_ids(torch)
    embedded = gather(weight, ids, torch.float32)
    with safe_open(model / "model-00001-of-00003.safetensors", framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor("model.language_model.layers.0.input_layernorm.weight")
    x = embedded.cuda().contiguous().view(-1, embedded.shape[-1])
    y = torch.empty_like(x, dtype=torch.float16)
    ext.rms_norm(x, norm_weight.cuda(), y, RMS_EPS, 1.0, 1.0, False, False)
    hidden = y.view(1, ids.shape[1], -1)
    if sha256(hidden) != ORACLE_GDN_INPUT:
        raise RuntimeError("GDN input no longer matches the oracle")
    return hidden


def exl3(name, trellis, suh, svh, mul1, out_features, in_features=5120, out_dtype=torch.float):
    return LinearEXL3(
        config=NullConfig(),
        in_features=in_features,
        out_features=out_features,
        suh=suh,
        svh=svh,
        trellis=trellis,
        mul1=mul1,
        out_dtype=out_dtype,
        key=name,
    )


def cached_exl3(weights, name, trellis, suh, svh, mul1, out_features, in_features=5120, out_dtype=torch.float):
    """Reuse one BC_LinearEXL3 for this resident weight group.

    Building the module constructs the Blackwell GEMM every call. The cache
    lives on the weight group, not on a process-wide pointer, so a later
    load of the same artifact does not revive a freed module.
    """
    cache = weights.setdefault("_linears", {})
    key = (name, int(out_features), int(in_features), out_dtype)
    layer = cache.get(key)
    if layer is None:
        layer = exl3(name, trellis, suh, svh, mul1, out_features, in_features, out_dtype)
        cache[key] = layer
    return layer


def load_layer(model, index=0):
    prefix = f"model.language_model.layers.{index}.linear_attn"
    with safe_open(model / "model-00001-of-00003.safetensors", framework="pt", device="cpu") as handle:
        names = [key for key in handle.keys() if key.startswith(prefix)]
        got = {name[len(prefix) + 1:]: handle.get_tensor(name).cuda().contiguous() for name in names}
    return got


def fp16_project(hidden, stored, weights=None, name=None):
    """LinearFP16 keeps the transposed weight and writes a float32 hgemm."""
    rows = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    weight = None
    if weights is not None and name is not None:
        table = weights.setdefault("_fp16", {})
        weight = table.get(name)
        if weight is None:
            weight = stored.transpose(0, 1).contiguous()
            table[name] = weight
    if weight is None:
        weight = stored.transpose(0, 1).contiguous()
    out = torch.empty((rows.shape[0], stored.shape[0]), dtype=torch.float32, device=rows.device)
    ext.hgemm(rows, weight, out)
    return out.view(*hidden.shape[:-1], stored.shape[0])


def project(hidden, weights):
    # 63 rows stay on BC_LinearEXL3. Longer prefills take reconstruct_hgemm.
    qkv = cached_exl3(
        weights, "qkv", weights["in_proj_qkv.trellis"], weights["in_proj_qkv.suh"],
        weights["in_proj_qkv.svh"], weights["in_proj_qkv.mul1"], QKV_OUT,
    )
    z_layer = cached_exl3(
        weights, "z", weights["in_proj_z.trellis"], weights["in_proj_z.suh"],
        weights["in_proj_z.svh"], weights["in_proj_z.mul1"], Z_OUT,
    )
    qkv_out = qkv.forward(hidden, {})
    z = z_layer.forward(hidden, {}).view(1, hidden.shape[1], NUM_V_HEADS, HEAD_DIM)
    b = fp16_project(hidden, weights["in_proj_b.weight"], weights, "in_proj_b.weight")
    a = fp16_project(hidden, weights["in_proj_a.weight"], weights, "in_proj_a.weight")
    return qkv_out, z, b, a


def _capture_chunk(mixed_qkv, beta, g, recurrent):
    """Prefill writes the final chunk state without feeding a zero initial state."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    bsz, seqlen, _ = mixed_qkv.shape
    k_dim = NUM_K_HEADS * HEAD_DIM
    v_dim = NUM_V_HEADS * HEAD_DIM
    query, key, value = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim=-1)
    core, new_state = chunk_gated_delta_rule(
        query.view(bsz, seqlen, NUM_K_HEADS, HEAD_DIM),
        key.view(bsz, seqlen, NUM_K_HEADS, HEAD_DIM),
        value.view(bsz, seqlen, NUM_V_HEADS, HEAD_DIM),
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    recurrent[0, 0].copy_(new_state[0])
    return core


def _remember_window(state, mixed, beta, g, conv, bias):
    """Keep the inputs of a short recurrent chunk so it can be cut later.

    The fused rule updates the state to the end of the chunk. One reused
    buffer holds the pre-chunk state; seven full copies would be about a
    gigabyte. Prefill above eight tokens does not take this path.
    """
    length = mixed.shape[-1]
    remembered = state._rewind_mixed
    if remembered is None or remembered.shape[-1] != length:
        state._rewind_mixed = mixed.clone()
        state._rewind_beta = beta.clone()
        state._rewind_g = g.clone()
    else:
        state._rewind_mixed.copy_(mixed)
        state._rewind_beta.copy_(beta)
        state._rewind_g.copy_(g)
    if state._rewind_conv is None:
        state._rewind_conv = state.conv.clone()
        state._rewind_recurrent = state.recurrent.clone()
    else:
        state._rewind_conv.copy_(state.conv)
        state._rewind_recurrent.copy_(state.recurrent)
    state._rewind_weight = conv
    state._rewind_bias = bias
    state._rewind_len = length


def rewind_gdn_state(state, keep):
    """Move one GDN state from the end of the stashed chunk back to ``keep``."""
    length = state._rewind_len
    if length is None:
        raise RuntimeError("no short GDN window to rewind")
    state._rewind_len = None
    if keep == length:
        return
    state.conv.copy_(state._rewind_conv)
    state.recurrent.copy_(state._rewind_recurrent)
    if keep == 0:
        return
    if not 1 <= keep < length:
        raise RuntimeError(f"cannot rewind a {length}-token GDN window to {keep}")
    from exllamav3.modules.gated_delta_net_fn import causal_conv1d_update, gated_delta_rule_fn

    mixed = state._rewind_mixed[:, :, :keep].contiguous()
    beta = state._rewind_beta[:, :keep].contiguous()
    g = state._rewind_g[:, :keep].contiguous()
    slots = torch.arange(mixed.shape[0], device=mixed.device, dtype=torch.int32)
    conv_out = causal_conv1d_update(
        mixed, state.conv, slots, state._rewind_weight, state._rewind_bias, False, {},
    )
    gated_delta_rule_fn(
        conv_out, beta, g, state.recurrent, slots, False, True,
        NUM_K_HEADS, NUM_V_HEADS, NUM_K_HEADS * HEAD_DIM, NUM_V_HEADS * HEAD_DIM,
        HEAD_DIM, HEAD_DIM, {},
    )


def gdn_forward(hidden, weights, state=None):
    from exllamav3.modules.gated_delta_net_fn import causal_conv1d_update, gated_delta_rule_fn

    qkv, z, b, a = project(hidden, weights)
    seqlen = hidden.shape[1]
    beta = torch.empty(1, seqlen, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)
    g = torch.empty(1, seqlen, NUM_V_HEADS, device="cuda", dtype=torch.float32)
    # The ported sigmoid rounds b=-3.7929 to a different bf16 than ExLlama, and
    # that one value moves the layer-2 state. This is the kernel the oracle runs.
    ext.gated_delta_net_fused_op_2(b, a, weights["dt_bias"], weights["A_log"], beta, g, 1.0)
    mixed = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
    conv = weights.get("_conv1d")
    if conv is None:
        raw = weights["conv1d.weight"]
        conv = raw.squeeze(1).contiguous() if raw.dim() == 3 else raw
        weights["_conv1d"] = conv
    bias = weights.get("conv1d.bias")
    # Prefill longer than 32 tokens uses ExLlama's Triton conv. At 63 tokens
    # the delta rule is the chunk kernel, because the length reaches the 48
    # value heads. The ported recurrent kernel remains the short decode path.
    # A verify is 2..8 tokens. Stash that chunk so the state can be cut
    # without a second 64-layer pass. A one-token step has nothing to cut.
    if state is not None and 1 < seqlen <= 8:
        _remember_window(state, mixed, beta, g, conv, bias)
    slots = None if state is None else torch.arange(hidden.shape[0], device=hidden.device, dtype=torch.int32)
    conv_out = causal_conv1d_update(
        mixed, None if state is None else state.conv, slots, conv, bias, False, {},
    )
    if state is None:
        core = gated_delta_rule_fn(
            conv_out, beta, g, None, None, False, False,
            NUM_K_HEADS, NUM_V_HEADS, NUM_K_HEADS * HEAD_DIM, NUM_V_HEADS * HEAD_DIM,
            HEAD_DIM, HEAD_DIM, {},
        )
    elif seqlen >= NUM_V_HEADS and not state.ready:
        core = _capture_chunk(conv_out, beta, g, state.recurrent)
        state.ready = True
    else:
        core = gated_delta_rule_fn(
            conv_out, beta, g, state.recurrent, slots, False, True,
            NUM_K_HEADS, NUM_V_HEADS, NUM_K_HEADS * HEAD_DIM, NUM_V_HEADS * HEAD_DIM,
            HEAD_DIM, HEAD_DIM, {},
        )
    y = torch.empty_like(core, dtype=torch.float16)
    ext.gated_rms_norm(core, weights["norm.weight"], y, z, 1e-6, 0.0, 1, False)
    flat = y.view(1, seqlen, Z_OUT)
    out_layer = cached_exl3(
        weights, "out", weights["out_proj.trellis"], weights["out_proj.suh"], weights["out_proj.svh"],
        weights["out_proj.mul1"], 5120, Z_OUT,
    )
    return out_layer.forward(flat, {}), qkv, z, a, b


def layer_output(hidden, weights):
    out, qkv, z, a, b = gdn_forward(hidden, weights)
    qkv_sha = sha256(qkv)
    z_sha = sha256(z)
    a_sha = sha256(a)
    b_sha = sha256(b)
    out_sha = sha256(out)
    return {
        "qkv": list(qkv.shape),
        "z": list(z.shape),
        "a": list(a.shape),
        "b": list(b.shape),
        "qkv_sha256": qkv_sha,
        "z_sha256": z_sha,
        "a_sha256": a_sha,
        "b_sha256": b_sha,
        "output_sha256": out_sha,
        "matches_projections": qkv_sha == ORACLE_QKV and z_sha == ORACLE_Z and a_sha == ORACLE_A and b_sha == ORACLE_B,
        "matches_oracle": out_sha == ORACLE_GDN_OUTPUT,
    }


def main():
    model = Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw"
    hidden = hidden_state(model)
    weights = load_layer(model)
    print(json.dumps(layer_output(hidden, weights)))


if __name__ == "__main__":
    main()
