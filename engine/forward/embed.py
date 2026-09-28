"""Embedding lookup for the q38 forward, checked against the ExLlama oracle.

The short prompt is 64 ids from the recorded seed. ExLlama embeds the first
63 during prefill and the last one on the decode step. The weight is the
BF16 table stored in the EXL3 artifact, gathered and cast the way the
oracle recorded it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SEED = 20260928
ID_LOW = 1000
ID_HIGH = 20000
PROMPT_LENGTH = 64
PREFILL_IDS = 63
ORACLE_PREFILL_FP32 = "21e6aac97dfefa4574c5b65b7105603f305029ea7e8abcaf6f9db60c68f1d85c"
ORACLE_PREFILL_FP16 = "4c10a3e3c694793e2cddb25c065c6bc6b830d8b146d5931a49083fbd149ae5dc"
ORACLE_PREFILL_IDS = "4f835e0b3017295e4aebfc78f2d234af28537e4f7afd9916c3dbb9e4dffbe048"
ORACLE_GDN_INPUT = "af57d3ec8ccbb3717aa2cd8d7cd48eb98f5b484d53c9287d7228996bca30fe93"
RMS_EPS = 1e-6


def prompt_ids(torch, length=PROMPT_LENGTH):
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(ID_LOW, ID_HIGH, (1, length), generator=generator)


def tensor_sha256(torch, tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).reshape(-1).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def gather(weight, ids, out_dtype):
    rows = weight.index_select(0, ids.reshape(-1)).reshape(*ids.shape, weight.shape[1])
    return rows.to(out_dtype)


def prefill_ids(torch):
    return prompt_ids(torch)[:, :PREFILL_IDS]


def rms_norm(torch, x, weight, bias=1.0):
    """ExLlama input norm: RMS, then multiply by weight + 1, output fp16."""
    hidden = x.float()
    variance = hidden.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS
    hidden = hidden * torch.rsqrt(variance)
    return (hidden * (weight.float() + bias)).to(torch.float16)


def load_weight(torch, model):
    from safetensors import safe_open
    path = Path(model) / "model-00001-of-00003.safetensors"
    with safe_open(path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("model.language_model.embed_tokens.weight")
    return weight


def match_prefill(torch, weight):
    ids = prefill_ids(torch)
    if tensor_sha256(torch, ids) != ORACLE_PREFILL_IDS:
        raise RuntimeError("prefill ids do not match the oracle")
    fp32 = tensor_sha256(torch, gather(weight, ids, torch.float32))
    fp16 = tensor_sha256(torch, gather(weight, ids, torch.float16))
    return {
        "fp32": fp32 == ORACLE_PREFILL_FP32,
        "fp16": fp16 == ORACLE_PREFILL_FP16,
        "fp32_sha256": fp32,
        "fp16_sha256": fp16,
    }


if __name__ == "__main__":
    import json
    import sys
    import torch
    model = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw")
    from safetensors import safe_open
    weight = load_weight(torch, model)
    ids = prefill_ids(torch)
    embedded = gather(weight, ids, torch.float32)
    with safe_open(Path(model) / "model-00001-of-00003.safetensors", framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor("model.language_model.layers.0.input_layernorm.weight")
    from exllamav3.ext import exllamav3_ext as ext
    report = match_prefill(torch, weight)
    x = embedded.cuda().contiguous().view(-1, embedded.shape[-1])
    y = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    ext.rms_norm(x, norm_weight.cuda(), y, RMS_EPS, 1.0, 1.0, False, False)
    got = tensor_sha256(torch, y.cpu())
    report["gdn_input"] = got == ORACLE_GDN_INPUT
    report["gdn_input_sha256"] = got
    print(json.dumps({"dtype": str(weight.dtype), "shape": list(weight.shape), **report}))
