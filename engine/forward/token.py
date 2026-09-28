"""Greedy token after the production prefill schedule.

The last id of the prompt is one decode step. Everything before it is
prefill, in the same pieces the live service uses: chunks of 8192 tokens,
cut on a page boundary, with the final partial page forwarded alone. Each
piece carries the Gated DeltaNet state and the NVFP4 pages forward.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from safetensors import safe_open

from engine.forward.attention import (
    ORACLE_ATTN_INPUT,
    ORACLE_ATTN_OUTPUT,
    attention_forward,
    empty_cache,
)
from engine.forward.embed import ORACLE_GDN_INPUT, RMS_EPS, gather, load_weight, prompt_ids
from engine.forward.mlp import HIDDEN, ORACLE_MLP_INPUT, ORACLE_MLP_OUTPUT, _runtime, mlp_forward
from engine.forward.projections import ORACLE_GDN_OUTPUT, exl3, gdn_forward, sha256
from engine.forward.schedule import CHUNK, PAGE, suffix_bounds

LAYERS = 64
CONV_DIM = 10240
GREEDY = {64: 198, 32768: 317, 261888: 317}


class Catalog:
    def __init__(self, directory):
        self.handles = []
        self.index = {}
        for path in sorted(Path(directory).glob("model-*.safetensors")):
            handle = safe_open(path, framework="pt", device="cpu")
            self.handles.append(handle)
            for key in handle.keys():
                self.index[key] = handle

    def tensor(self, key):
        return self.index[key].get_tensor(key).cuda().contiguous()

    def group(self, prefix):
        head = prefix + "."
        found = {
            key[len(head):]: handle.get_tensor(key).cuda().contiguous()
            for key, handle in self.index.items()
            if key.startswith(head)
        }
        if not found:
            raise KeyError(prefix)
        return found


class GDNState:
    def __init__(self):
        self.conv = torch.zeros(1, CONV_DIM, 4, dtype=torch.bfloat16, device="cuda")
        self.recurrent = torch.zeros(1, 1, 48, 128, 128, dtype=torch.float32, device="cuda")
        self.ready = False


def load_mlp(donor, index):
    from qwasar_runtime.nvidia_mlp import tensors_for
    from qwasar_runtime.nvfp4_linear import NativeLinear

    prefix = f"model.language_model.layers.{index}.mlp"
    dtypes = {"gate_proj": torch.float16, "up_proj": torch.float16, "down_proj": torch.float32}
    return {
        name: NativeLinear(
            f"{prefix}.{name}", tensors_for(donor.index, f"{prefix}.{name}"), "cuda", dtype, "adaptive",
        )
        for name, dtype in dtypes.items()
    }


def rms(weight, residual):
    normed = torch.empty(residual.shape, dtype=torch.float16, device=residual.device)
    ext.rms_norm(
        residual.reshape(-1, HIDDEN).contiguous(),
        weight,
        normed.reshape(-1, HIDDEN),
        RMS_EPS, 1.0, 1.0, False, False,
    )
    return normed


def fuse(weight, residual, sublayer):
    normed = torch.empty(residual.shape, dtype=torch.float16, device=residual.device)
    ext.rms_norm_res_in(
        sublayer.reshape(-1, HIDDEN).contiguous(),
        weight,
        normed.reshape(-1, HIDDEN),
        residual.reshape(-1, HIDDEN),
        RMS_EPS, 1.0, 1.0,
    )
    return normed


def prefill_bounds(length, chunk=CHUNK, page=PAGE):
    """Same cuts as ExLlama's recurrent prefill with an 8192-token chunk."""
    return suffix_bounds(0, length, chunk, page)


def load_spec(model, donor, index):
    kind = "attn" if index % 4 == 3 else "gdn"
    child = "self_attn" if kind == "attn" else "linear_attn"
    return {
        "kind": kind,
        "weights": model.group(f"model.language_model.layers.{index}.{child}"),
        "in_norm": model.tensor(f"model.language_model.layers.{index}.input_layernorm.weight"),
        "post_norm": model.tensor(f"model.language_model.layers.{index}.post_attention_layernorm.weight"),
        "mlp": load_mlp(donor, index),
    }


def apply_layer(residual, index, spec, states, caches, cache_len, pages):
    normed = rms(spec["in_norm"], residual)
    if spec["kind"] == "attn":
        caches.setdefault(index, empty_cache(pages))
        sublayer, _ = attention_forward(normed, spec["weights"], caches[index], cache_len)
    else:
        states.setdefault(index, GDNState())
        sublayer = gdn_forward(normed, spec["weights"], states[index])[0]
    if cache_len == 0 and residual.shape[1] == 63 and index == 0:
        if sha256(normed) != ORACLE_GDN_INPUT or sha256(sublayer) != ORACLE_GDN_OUTPUT:
            raise RuntimeError("layer 0 GDN drifted")
    if cache_len == 0 and residual.shape[1] == 63 and index == 3:
        if sha256(normed) != ORACLE_ATTN_INPUT or sha256(sublayer) != ORACLE_ATTN_OUTPUT:
            raise RuntimeError("layer 3 attention drifted")
    entered = fuse(spec["post_norm"], residual, sublayer)
    produced = mlp_forward(entered, spec["mlp"])
    if cache_len == 0 and residual.shape[1] == 63 and index == 0:
        if sha256(entered) != ORACLE_MLP_INPUT or sha256(produced) != ORACLE_MLP_OUTPUT:
            raise RuntimeError("layer 0 MLP drifted")
    return residual + produced, normed, sublayer, produced


def logits(model, residual):
    normed = rms(model.tensor("model.language_model.norm.weight"), residual)
    head = model.group("lm_head")
    layer = exl3(
        "lm_head", head["trellis"], head["suh"], head["svh"], head["mul1"],
        248320, HIDDEN, torch.float16,
    )
    return layer.forward(normed, {})


def recorded_outputs(length, kind):
    path = Path(__file__).resolve().parents[2] / "results/q38-oracle/capture.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for ctx in data["contexts"]:
        if ctx["length"] == length:
            return [(row["in"]["sha256"], row["out"]["sha256"]) for row in ctx["taps"][kind]]
    return None


def generate(length):
    _runtime()
    from qwasar_runtime.hybrid import donor_dir, prepare_environment

    prepare_environment()
    bounds = prefill_bounds(length)
    pages = (length - 1) // PAGE + 1
    model_dir = Path.home() / "models/Qwen3.8-27B-EXL3-5.0bpw"
    model = Catalog(model_dir)
    donor = Catalog(donor_dir())
    ids = prompt_ids(torch, length)
    table = load_weight(torch, model_dir)
    pieces = [gather(table, ids[:, start:end], torch.float32).cuda().contiguous() for start, end in bounds]
    states, caches = {}, {}
    gdn_oracle = recorded_outputs(length, "gdn")
    mlp_oracle = recorded_outputs(length, "mlp")
    attn_oracle = recorded_outputs(length, "attention")
    order = [(index, piece, start) for index in range(LAYERS) for piece, (start, _end) in enumerate(bounds)]
    loaded = None
    spec = None
    for index, piece, start in order:
        if loaded != index:
            spec = load_spec(model, donor, index)
            loaded = index
            torch.cuda.empty_cache()
        pieces[piece], normed, sublayer, produced = apply_layer(
            pieces[piece], index, spec, states, caches, start, pages,
        )
        if index == 0 and gdn_oracle is not None and sha256(sublayer) != gdn_oracle[piece][1]:
            raise RuntimeError(f"layer 0 chunk {piece} drifted")
        if index == 0 and mlp_oracle is not None and sha256(produced) != mlp_oracle[piece][1]:
            raise RuntimeError(f"layer 0 mlp chunk {piece} drifted")
        if index == 3 and attn_oracle is not None and (
            sha256(normed) != attn_oracle[piece][0] or sha256(sublayer) != attn_oracle[piece][1]
        ):
            raise RuntimeError(f"layer 3 chunk {piece} drifted")
        del normed, sublayer, produced
    held = gather(table, ids[:, length - 1:length], torch.float32).cuda().contiguous()
    for index in range(LAYERS):
        spec = load_spec(model, donor, index)
        held, _normed, _sublayer, _produced = apply_layer(
            held, index, spec, states, caches, length - 1, pages,
        )
        del spec
        torch.cuda.empty_cache()
    scores = logits(model, held)
    token = int(scores.reshape(-1).argmax())
    expected = GREEDY[length]
    print(json.dumps({
        "length": length,
        "greedy_token": token,
        "expected": expected,
        "matches_oracle": token == expected,
    }), flush=True)
    return token


def main():
    import sys
    length = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    generate(length)


if __name__ == "__main__":
    main()
