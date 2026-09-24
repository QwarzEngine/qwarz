import hashlib
import sys
import types

import pytest

from qwasar_bench import nvfp4_projections as projections
from qwasar_bench.nvfp4_projections import COUNTS, checkpoint_key, replace, selected, verify_file


def model_keys():
    for layer in range(64):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            yield f"model.layers.{layer}.mlp.{proj}"
        kind = "self_attn" if layer % 4 == 3 else "linear_attn"
        projs = ("q_proj", "k_proj", "v_proj", "o_proj") if layer % 4 == 3 else (
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
        for proj in projs:
            yield f"model.layers.{layer}.{kind}.{proj}"


@pytest.mark.parametrize("profile,count", COUNTS.items())
def test_exact_projection_coverage(profile, count):
    keys = list(model_keys())
    assert sum(selected(key, profile) for key in keys) == count
    assert sum(selected(key.replace("model.", "model.language_model.", 1), profile)
               for key in keys) == count
    assert not any(selected(key, profile) for key in keys if ".mlp." in key)


@pytest.mark.parametrize("key", [
    "model.layers.64.linear_attn.in_proj_qkv", "mtp.layers.0.self_attn.q_proj",
    "model.visual.layers.3.self_attn.q_proj", "lm_head", "model.embed_tokens",
    "model.layers.0.linear_attn.conv1d", "model.layers.0.self_attn.q_proj",
    "model.layers.3.linear_attn.out_proj", "model.layers.3.self_attn.q_norm",
])
def test_protected_modules_are_never_selected(key):
    assert not any(selected(key, profile) for profile in COUNTS)


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match="unknown"):
        selected("anything", "typo")
    with pytest.raises(ValueError, match="unknown"):
        with replace("typo"):
            pass


def test_control_does_not_import_gpu_runtime_or_open_donor():
    with replace("control", "/does/not/exist") as loaded:
        assert loaded == []


def test_file_verification_reads_contents(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"abcd")
    expected = hashlib.sha256(b"abcd").hexdigest()
    assert verify_file(path, expected)["sha256"] == expected
    path.write_bytes(b"abce")
    with pytest.raises(ValueError, match="SHA256"):
        verify_file(path, expected)


def test_checkpoint_key_conversion():
    assert checkpoint_key("model.language_model.layers.3.self_attn.q_proj") == (
        "model.layers.3.self_attn.q_proj")


@pytest.mark.parametrize("fail", [False, True])
def test_load_hook_composes_and_restores(monkeypatch, fail):
    class Linear:
        def load(self, device, **kwargs):
            return "original"

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "exllamav3.modules.linear", types.SimpleNamespace(Linear=Linear))
    monkeypatch.setitem(sys.modules, "safetensors", types.SimpleNamespace(safe_open=lambda *a, **k: Handle()))
    monkeypatch.setattr(projections, "read_tensors", lambda *a: {})
    monkeypatch.setattr(projections, "validate_tensors", lambda *a: (5120, 1024))
    monkeypatch.setattr(projections, "PaddedLinear", lambda *a: types.SimpleNamespace(
        in_features=5120, logical_out=1024, out_features=1024, quant_type="nvfp4_spike"))
    original = Linear.load
    module = Linear()
    module.key = "model.layers.3.self_attn.k_proj"
    module.is_sliced, module.lora_a_tensors, module.weight_scale = False, [], 1
    module.in_features, module.out_features, module.out_dtype = 5120, 1024, "float"
    try:
        with replace("attention") as loaded:
            module.load("cuda")
            assert len(loaded) == 1 and module.quant_type == "nvfp4_spike"
            module.key = "model.layers.3.mlp.up_proj"
            assert module.load("cuda") == "original"
            if fail:
                raise RuntimeError("partial load")
    except RuntimeError:
        assert fail
    assert Linear.load is original


def tensors():
    torch = pytest.importorskip("torch")
    return {
        "weight_packed": torch.zeros((48, 128), dtype=torch.uint8),
        "weight_scale": torch.ones((48, 16), dtype=torch.float32).to(torch.float8_e4m3fn),
        "input_global_scale": torch.tensor(440.0),
        "weight_global_scale": torch.tensor(6400.0),
    }


@pytest.mark.parametrize("field,value", [
    ("input_global_scale", 0), ("input_global_scale", float("nan")),
    ("weight_global_scale", -1), ("weight_global_scale", float("inf")),
])
def test_invalid_global_scale_rejected(field, value):
    values = tensors()
    values[field].fill_(value)
    with pytest.raises(ValueError, match=field):
        projections.validate_tensors(values)


def test_invalid_geometry_rejected():
    values = tensors()
    values["weight_scale"] = values["weight_scale"][:, :-1]
    with pytest.raises(ValueError, match="geometry"):
        projections.validate_tensors(values)


def test_padding_preserves_globals_and_logical_output(monkeypatch):
    values = tensors()
    import torch
    received = {}

    def init(self, key, ts, device, out_dtype, backend):
        received.update(ts)
        self.out_features = ts["weight_packed"].shape[0]

    monkeypatch.setattr(projections.NativeLinear, "__init__", init)
    monkeypatch.setattr(projections.NativeLinear, "forward", lambda self, x, *a:
                        torch.ones((*x.shape[:-1], self.out_features), dtype=torch.float32))
    module = projections.PaddedLinear("aux", values, "cpu", torch.float32)
    assert received["weight_packed"].shape == (128, 128)
    assert received["weight_scale"].shape == (128, 16)
    assert received["input_global_scale"] is values["input_global_scale"]
    assert received["weight_global_scale"] is values["weight_global_scale"]
    assert not received["weight_scale"][48:].float().any()
    output = module.forward(torch.ones((2, 7, 256)))
    assert output.shape == (2, 7, 48) and output.dtype == torch.float32 and output.is_contiguous()
    assert values["weight_packed"].shape == (48, 128)
