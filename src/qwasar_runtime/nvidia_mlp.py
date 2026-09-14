"""Process-local NVIDIA NVFP4 MLP donor for the 192 gate/up/down matrices.

Inverts both ModelOpt global scales to the reciprocal form NativeLinear expects.
GDN, attention, embeddings, lm_head and MTP stay EXL3.
"""
import contextlib
import re

from .nvfp4_linear import NativeLinear, setup_flashinfer

FIELD_MAP = {
    "weight_packed": "weight",
    "weight_scale": "weight_scale",
    "input_global_scale": "input_scale",
    "weight_global_scale": "weight_scale_2",
}
_MLP_RE = re.compile(
    r"^model(?:\.language_model)?\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$")
SHARDS = (
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
)


def selected(key, profile):
    match = _MLP_RE.fullmatch(key)
    if not match:
        return False
    layer = int(match[1])
    if profile == "nvidia56":
        return layer < 56
    return profile == "nvidia64"


def checkpoint_key(module_key):
    if module_key.startswith("model.language_model."):
        return module_key
    return module_key.replace("model.", "model.language_model.", 1)


def convert_modelopt_tensors(raw):
    out = dict(raw)
    for field in ("weight_global_scale", "input_global_scale"):
        value = raw[field].float().reshape(())
        assert value.item() > 0, (field, value.item())
        out[field] = 1.0 / value
    return out


def tensors_for(index, key):
    raw = {field: index[f"{key}.{suffix}"].get_tensor(f"{key}.{suffix}")
           for field, suffix in FIELD_MAP.items()}
    return convert_modelopt_tensors(raw)


@contextlib.contextmanager
def replace(profile, checkpoint_dir):
    setup_flashinfer()
    from exllamav3.modules.linear import Linear
    from safetensors import safe_open

    original = Linear.load
    loaded = []
    handles, index = [], {}
    for shard in SHARDS:
        handle = safe_open(str(checkpoint_dir / shard), framework="pt", device="cpu")
        handles.append(handle)
        for key in handle.keys():
            index[key] = handle

    def load(module, device, **kwargs):
        if not selected(module.key, profile):
            return original(module, device, **kwargs)
        assert not module.is_sliced and not module.lora_a_tensors and module.weight_scale == 1
        key = checkpoint_key(module.key)
        tensors = tensors_for(index, key)
        inner = NativeLinear(key, tensors, device, module.out_dtype, "adaptive")
        assert inner.input_scale.item() > 1 and inner.weight_global_scale.item() > 1, \
            (key, inner.input_scale.item(), inner.weight_global_scale.item())
        logical_out = tensors["weight_packed"].shape[0]
        assert inner.in_features == module.in_features and logical_out == module.out_features, \
            (key, logical_out, module.out_features)
        module.device = device
        module.inner = inner
        module.quant_type = inner.quant_type
        loaded.append({"key": module.key, "source": key,
                       "in": inner.in_features, "out": logical_out,
                       "padded_out": inner.out_features,
                       "dtype": str(inner.default_out_dtype)})

    Linear.load = load
    try:
        yield loaded
    finally:
        Linear.load = original
