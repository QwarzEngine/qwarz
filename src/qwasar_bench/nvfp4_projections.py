"""Experimental projection grafts. Never imported by the production loader."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re

from qwasar_runtime.nvfp4_linear import NativeLinear

CHECKPOINT = Path("/home/rekeyea/models/minima-qwen38-27b-nvfp4/"
                  "16e768e7d0461b0b86e565ecedd08a24eca53e9a/model.safetensors")
SHA256 = "4f046fddf0809838b7d3c7d40805c1f2bbcf2e047aac7cb0a2de2db636a569d9"
FIELDS = ("weight_packed", "weight_scale", "input_global_scale", "weight_global_scale")
COUNTS = {"control": 0, "attention": 64, "gdn_input": 96, "gdn_output": 48,
          "gdn_all": 240, "all": 304}
KEY = re.compile(r"model(?:\.language_model)?\.layers\.(\d+)\."
                 r"(self_attn|linear_attn)\.(\w+)")


def selected(key, profile):
    if profile not in COUNTS:
        raise ValueError(f"unknown projection profile: {profile}")
    match = KEY.fullmatch(key)
    if not match or profile == "control":
        return False
    layer, kind, proj = int(match[1]), match[2], match[3]
    if not 0 <= layer < 64:
        return False
    if kind == "self_attn":
        return (layer % 4 == 3 and profile in ("attention", "all")
                and proj in ("q_proj", "k_proj", "v_proj", "o_proj"))
    if layer % 4 == 3:
        return False
    groups = {
        "gdn_input": ("in_proj_qkv", "in_proj_z"),
        "gdn_output": ("out_proj",),
        "gdn_all": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"),
    }
    return proj in groups.get("gdn_all" if profile == "all" else profile, ())


def checkpoint_key(key):
    return key.replace("model.language_model.", "model.", 1)


def verify_file(path, expected_sha256):
    path = Path(path)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def validate_tensors(tensors):
    import torch

    w, s = tensors["weight_packed"], tensors["weight_scale"]
    if w.ndim != 2 or w.dtype != torch.uint8 or w.shape[1] % 8:
        raise ValueError("invalid packed NVFP4 weights")
    if s.dtype != torch.float8_e4m3fn or tuple(s.shape) != (w.shape[0], w.shape[1] // 8):
        raise ValueError("invalid NVFP4 block scale geometry")
    if not torch.isfinite(s.float()).all() or (s.float() < 0).any():
        raise ValueError("invalid NVFP4 block scales")
    for field in FIELDS[2:]:
        value = tensors[field]
        if value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0:
            raise ValueError(f"invalid {field}")
    return w.shape[1] * 2, w.shape[0]


class PaddedLinear(NativeLinear):
    """Minima uses reciprocal globals already; do NOT invert them like ModelOpt."""

    def __init__(self, key, tensors, device, out_dtype=None, backend="adaptive"):
        import torch

        _, self.logical_out = validate_tensors(tensors)
        padded = (self.logical_out + 127) // 128 * 128
        tensors = dict(tensors)
        if padded != self.logical_out:
            for field in FIELDS[:2]:
                value = tensors[field]
                extra = torch.zeros((padded, value.shape[1]), dtype=value.dtype, device=value.device)
                extra[:self.logical_out] = value
                tensors[field] = extra
        super().__init__(key, tensors, device, out_dtype, backend)

    def forward(self, x, params=None, out_dtype=None):
        output = super().forward(x, params, out_dtype)
        return output[..., :self.logical_out].contiguous()


def read_tensors(handle, key):
    return {field: handle.get_tensor(checkpoint_key(key) + "." + field) for field in FIELDS}


@contextmanager
def replace(profile, checkpoint=CHECKPOINT):
    """Compose with NVIDIA's MLP hook; restore even on partial load failure."""
    if profile not in COUNTS:
        raise ValueError(f"unknown projection profile: {profile}")
    loaded = []
    if profile == "control":
        yield loaded
        return
    from exllamav3.modules.linear import Linear
    from safetensors import safe_open

    original = Linear.load
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        def load(module, device, **kwargs):
            if not selected(module.key, profile):
                return original(module, device, **kwargs)
            if module.is_sliced or module.lora_a_tensors or module.weight_scale != 1:
                raise ValueError(f"unsupported projection load: {module.key}")
            tensors = read_tensors(handle, module.key)
            geometry = validate_tensors(tensors)
            if geometry != (module.in_features, module.out_features):
                raise ValueError(f"projection geometry mismatch: {module.key}: {geometry}")
            inner = PaddedLinear(module.key, tensors, device, module.out_dtype)
            module.device, module.inner, module.quant_type = device, inner, inner.quant_type
            loaded.append({"key": module.key, "in": inner.in_features, "out": inner.logical_out,
                           "padded_out": inner.out_features, "dtype": str(module.out_dtype)})

        Linear.load = load
        try:
            yield loaded
        finally:
            Linear.load = original
