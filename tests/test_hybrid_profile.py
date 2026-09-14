import ast
import json
import os
from pathlib import Path
import sys

import pytest

from qwasar_runtime.nvidia_mlp import SHARDS, checkpoint_key, convert_modelopt_tensors, selected


ROOT = Path(__file__).resolve().parents[1]
PROJS = ("gate_proj", "up_proj", "down_proj")


def test_production_pin_matches_download_record():
    pin = json.loads((ROOT / "benchmarks/manifests/nvidia-qwen38-27b-nvfp4.json").read_text())
    recorded = json.loads((ROOT / "results/20260910-quality-gate/nvidia-download.json").read_text())
    assert pin["revision"] == recorded["revision"]
    assert pin["path"] == recorded["path"]
    assert pin["shards"] == recorded["shards"]
    assert tuple(pin["shards"]) == SHARDS


def test_engine_does_not_import_hybrid_at_module_level():
    source = (ROOT / "src/qwasar_runtime/engine.py").read_text()
    tree = ast.parse(source)
    imported = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("hybrid" in name for name in imported)


def test_nvfp4_setup_does_not_mutate_flashinfer_environment():
    text = (ROOT / "src/qwasar_runtime/nvfp4_linear.py").read_text()
    assert "FLASHINFER_WORKSPACE_BASE" not in text
    assert "sys.path" not in text


def test_nvidia64_selects_exactly_192_mlp_matrices():
    keys = [f"model.layers.{layer}.mlp.{proj}" for layer in range(64) for proj in PROJS]
    assert sum(selected(key, "nvidia64") for key in keys) == 192
    assert all(selected(key.replace("model.", "model.language_model.", 1), "nvidia64") for key in keys)
    assert not selected("model.layers.0.self_attn.q_proj", "nvidia64")
    assert selected("model.layers.55.mlp.gate_proj", "nvidia56")
    assert not selected("model.layers.56.mlp.gate_proj", "nvidia56")
    assert checkpoint_key("model.layers.0.mlp.gate_proj") == "model.language_model.layers.0.mlp.gate_proj"
    assert checkpoint_key("model.language_model.layers.0.mlp.gate_proj") == (
        "model.language_model.layers.0.mlp.gate_proj")


def test_modelopt_globals_are_inverted_for_the_kernel():
    class Scale:
        def __init__(self, value):
            self.value = value

        def float(self):
            return Scale(float(self.value))

        def reshape(self, *args):
            return self

        def item(self):
            return self.value

        def __rtruediv__(self, other):
            return Scale(other / self.value)

    converted = convert_modelopt_tensors({
        "weight_packed": "keep",
        "weight_scale": "keep",
        "weight_global_scale": Scale(0.000157),
        "input_global_scale": Scale(0.00126),
    })
    assert converted["weight_packed"] == "keep"
    assert converted["input_global_scale"].item() == pytest.approx(1 / 0.00126)
    assert converted["weight_global_scale"].item() == pytest.approx(1 / 0.000157)
    assert converted["input_global_scale"].item() > 1
    assert converted["weight_global_scale"].item() > 1


def test_verify_donor_rejects_size_mismatch(tmp_path, monkeypatch):
    from qwasar_runtime import hybrid
    for name in hybrid.donor_pin()["shards"]:
        (tmp_path / name).write_bytes(b"short")
    monkeypatch.setenv("QWASAR_NVIDIA_DONOR", str(tmp_path))
    with pytest.raises(ValueError, match="size mismatch"):
        hybrid.verify_donor()


def test_verify_donor_accepts_recorded_sizes():
    from qwasar_runtime import hybrid
    donor = hybrid.verify_donor()
    pin = hybrid.donor_pin()
    assert donor["revision"] == pin["revision"]
    assert donor["path"] == pin["path"]
    assert {name: info["size"] for name, info in donor["shards"].items()} == {
        name: info["size"] for name, info in pin["shards"].items()}


def test_prepare_environment_keeps_pscaled_first_and_preserves_workspace(monkeypatch):
    from qwasar_runtime import hybrid
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.delenv("FLASHINFER_WORKSPACE_BASE", raising=False)
    hybrid.prepare_environment()
    assert sys.path[0] == str(hybrid.FLASHINFER_PATHS[0])
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(hybrid.WORKSPACE)
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", "/tmp/keep-workspace")
    hybrid.prepare_environment()
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == "/tmp/keep-workspace"
    assert hybrid.QUANTIZATION != hybrid.BASELINE_QUANTIZATION
    assert hybrid.QUANTIZATION == "5bpw-K8V4-MTP-NV64-PRIMS-ATT64"
    assert hybrid.DECODE_POLICY == {"default": {"block_n": 64, "num_warps": 4, "num_stages": 1}}
