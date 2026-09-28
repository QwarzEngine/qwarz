"""Layer-0 NVFP4 MLP on the oracle residual.

The block input is the fp32 embedding. The Gated DeltaNet output is added
into that residual by the post-attention RMSNorm. Gate and up are fp16,
silu(gate) * up is fp16, and down writes float32. The three matrices are
the NVIDIA donor, with the same global-scale inversion production uses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from safetensors import safe_open

from engine.forward.embed import RMS_EPS, gather, load_weight, prefill_ids, tensor_sha256
from engine.forward.projections import gdn_forward, hidden_state, load_layer, sha256

ROOT = Path(__file__).resolve().parents[2]

ORACLE_MLP_INPUT = "c25c7d2627a7b901fad3184c49d46d841e29498feeb3b0cf5c0a7a3d0081f15b"
ORACLE_MLP_OUTPUT = "0f514f55991ceead04f42180c7393e0bf1538eab9cb68551eb8e3cba70c8ac8d"
HIDDEN = 5120
FIELDS = ("weight", "weight_scale", "input_scale", "weight_scale_2")


def block_input(model):
    embedded = gather(load_weight(torch, model), prefill_ids(torch), torch.float32)
    if tensor_sha256(torch, embedded) != "21e6aac97dfefa4574c5b65b7105603f305029ea7e8abcaf6f9db60c68f1d85c":
        raise RuntimeError("block input no longer matches the oracle embedding")
    return embedded.cuda().contiguous()


def mlp_input(model, block, attention, index=0):
    with safe_open(model / "model-00001-of-00003.safetensors", framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"model.language_model.layers.{index}.post_attention_layernorm.weight")
    normed = torch.empty(block.shape, dtype=torch.float16, device=block.device)
    ext.rms_norm_res_in(
        attention.reshape(-1, HIDDEN).contiguous(),
        weight.cuda().contiguous(),
        normed.reshape(-1, HIDDEN),
        block.reshape(-1, HIDDEN),
        RMS_EPS,
        1.0,
        1.0,
    )
    return normed


def _runtime():
    source = ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def load_linears(donor, index=0):
    _runtime()
    from qwasar_runtime.hybrid import prepare_environment
    from qwasar_runtime.nvidia_mlp import tensors_for
    from qwasar_runtime.nvfp4_linear import NativeLinear

    prepare_environment()
    prefix = f"model.language_model.layers.{index}.mlp"
    dtypes = {"gate_proj": torch.float16, "up_proj": torch.float16, "down_proj": torch.float32}
    with safe_open(donor / "model-00001-of-00003.safetensors", framework="pt", device="cpu") as handle:
        stored = {
            f"{prefix}.{name}.{field}": handle
            for name in dtypes
            for field in FIELDS
        }
        return {
            name: NativeLinear(f"{prefix}.{name}", tensors_for(stored, f"{prefix}.{name}"), "cuda", dtype, "adaptive")
            for name, dtype in dtypes.items()
        }


def mlp_forward(hidden, linears):
    gate = linears["gate_proj"].forward(hidden, {})
    up = linears["up_proj"].forward(hidden, {})
    activated = torch.empty_like(up, dtype=torch.float16)
    ext.silu_mul(gate, up, activated, 0.0)
    return linears["down_proj"].forward(activated, {})


def main():
    _runtime()
    from qwasar_runtime.hybrid import donor_dir

    model = Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw"
    block = block_input(model)
    hidden = hidden_state(model)
    attention = gdn_forward(hidden, load_layer(model))[0]
    if sha256(attention) != "b8b0b6b5b029cca6712a63c5562a68c4cf4f66b74773d746cb2cdee6687114fb":
        raise RuntimeError("GDN output no longer matches the oracle")
    entered = mlp_input(model, block, attention)
    linears = load_linears(donor_dir())
    output = mlp_forward(entered, linears)
    entered_sha = sha256(entered)
    output_sha = sha256(output)
    print(json.dumps({
        "input_shape": list(entered.shape),
        "input_dtype": str(entered.dtype),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "gate": [linears["gate_proj"].in_features, linears["gate_proj"].out_features],
        "down": [linears["down_proj"].in_features, linears["down_proj"].out_features],
        "input_sha256": entered_sha,
        "output_sha256": output_sha,
        "matches_input": entered_sha == ORACLE_MLP_INPUT,
        "matches_oracle": output_sha == ORACLE_MLP_OUTPUT,
    }))


if __name__ == "__main__":
    main()
