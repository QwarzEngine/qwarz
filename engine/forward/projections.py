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
from engine.forward.gdn import HEAD_DIM, NUM_K_HEADS, NUM_V_HEADS, _load

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


def load_layer(model, index=0):
    prefix = f"model.language_model.layers.{index}.linear_attn"
    with safe_open(model / "model-00001-of-00003.safetensors", framework="pt", device="cpu") as handle:
        names = [key for key in handle.keys() if key.startswith(prefix)]
        got = {name[len(prefix) + 1:]: handle.get_tensor(name).cuda().contiguous() for name in names}
    return got


def fp16_project(hidden, stored):
    """LinearFP16 keeps the transposed weight and writes a float32 hgemm."""
    rows = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    weight = stored.transpose(0, 1).contiguous()
    out = torch.empty((rows.shape[0], stored.shape[0]), dtype=torch.float32, device=rows.device)
    ext.hgemm(rows, weight, out)
    return out.view(*hidden.shape[:-1], stored.shape[0])


def project(hidden, weights):
    # 63 rows stay on BC_LinearEXL3. Longer prefills take reconstruct_hgemm.
    qkv = exl3("qkv", weights["in_proj_qkv.trellis"], weights["in_proj_qkv.suh"], weights["in_proj_qkv.svh"], weights["in_proj_qkv.mul1"], QKV_OUT)
    z_layer = exl3("z", weights["in_proj_z.trellis"], weights["in_proj_z.suh"], weights["in_proj_z.svh"], weights["in_proj_z.mul1"], Z_OUT)
    qkv_out = qkv.forward(hidden, {})
    z = z_layer.forward(hidden, {}).view(1, hidden.shape[1], NUM_V_HEADS, HEAD_DIM)
    b = fp16_project(hidden, weights["in_proj_b.weight"])
    a = fp16_project(hidden, weights["in_proj_a.weight"])
    return qkv_out, z, b, a


def gdn_forward(hidden, weights):
    from exllamav3.modules.gated_delta_net_fn import causal_conv1d_update, gated_delta_rule_fn

    library = _load()
    qkv, z, b, a = project(hidden, weights)
    seqlen = hidden.shape[1]
    beta = torch.empty(1, seqlen, NUM_V_HEADS, device="cuda", dtype=torch.bfloat16)
    g = torch.empty(1, seqlen, NUM_V_HEADS, device="cuda", dtype=torch.float32)
    a_log = weights["A_log"].float().contiguous()
    library.q38_gdn_fused_op_2(
        b.data_ptr(), a.data_ptr(), weights["dt_bias"].data_ptr(), a_log.data_ptr(),
        beta.data_ptr(), g.data_ptr(), 1, seqlen, NUM_V_HEADS, 1.0,
    )
    mixed = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
    conv = weights["conv1d.weight"]
    if conv.dim() == 3:
        conv = conv.squeeze(1).contiguous()
    # Prefill longer than 32 tokens uses ExLlama's Triton conv. At 63 tokens
    # the delta rule is the chunk kernel, because the length reaches the 48
    # value heads. The ported recurrent kernel remains the short decode path.
    conv_out = causal_conv1d_update(mixed, None, None, conv, weights.get("conv1d.bias"), False, {})
    core = gated_delta_rule_fn(
        conv_out, beta, g, None, None, False, False,
        NUM_K_HEADS, NUM_V_HEADS, NUM_K_HEADS * HEAD_DIM, NUM_V_HEADS * HEAD_DIM,
        HEAD_DIM, HEAD_DIM, {},
    )
    y = torch.empty_like(core, dtype=torch.float16)
    ext.gated_rms_norm(core, weights["norm.weight"], y, z, 1e-6, 0.0, 1, False)
    flat = y.view(1, seqlen, Z_OUT)
    out_layer = exl3(
        "out", weights["out_proj.trellis"], weights["out_proj.suh"], weights["out_proj.svh"],
        weights["out_proj.mul1"], 5120,
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
