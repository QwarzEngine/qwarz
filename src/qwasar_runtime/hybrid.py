"""Production flash profile: NVIDIA64 + FP8 PRIMS + Attention64.

Import this module only from ExLlamaBackend when prefill == "flash". Module-level
import would pull FlashInfer/Triton into FakeBackend and CPU tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PIN_PATH = ROOT / "benchmarks/manifests/nvidia-qwen38-27b-nvfp4.json"
WORKSPACE = ROOT / "results/20260910-quality-gate/phase2-workspace"
FLASHINFER_PATHS = (
    ROOT / "results/20260908-upstream-experiments/fp8/pscaled",
    ROOT / "results/20260908-upstream-experiments/fp8/deps",
    ROOT / "results/20260908-upstream-experiments/fp8/deps/nvidia_cutlass_dsl/dsl_packages",
    ROOT / "results/20260908-hybrid-backends/flashinfer-deps",
    ROOT / "results/20260908-hybrid-backends/flashinfer-deps/nvidia_cutlass_dsl/dsl_packages",
)

BASELINE_QUANTIZATION = "5bpw-K8V4-MTP"
QUANTIZATION = "5bpw-K8V4-MTP-NV64-PRIMS-ATT64"
DECODE_POLICY = {"default": {"block_n": 64, "num_warps": 4, "num_stages": 1}}
PRIMS_MIN_QUERY = 8192
P_SCALE = 256
EXPECTED_MLP = 192
EXPECTED_GRAPHS = 64


def donor_pin():
    return json.loads(PIN_PATH.read_text())


def donor_dir():
    override = os.environ.get("QWASAR_NVIDIA_DONOR")
    return Path(override) if override else Path(donor_pin()["path"])


def verify_donor():
    pin = donor_pin()
    path = donor_dir()
    if not path.is_dir():
        raise ValueError(f"NVIDIA donor missing: {path}")
    shards = {}
    for name, expected in pin["shards"].items():
        shard = path / name
        if not shard.is_file():
            raise ValueError(f"NVIDIA shard missing: {shard}")
        size = shard.stat().st_size
        if size != expected["size"]:
            raise ValueError(f"NVIDIA shard size mismatch: {name}: {size} != {expected['size']}")
        shards[name] = {"size": expected["size"], "sha256": expected["sha256"]}
    return {"path": str(path), "revision": pin["revision"], "shards": shards}


def prepare_environment():
    missing = [str(path) for path in FLASHINFER_PATHS if not path.is_dir()]
    if missing:
        raise ValueError("FlashInfer hybrid trees missing: " + ", ".join(missing))
    for path in reversed(FLASHINFER_PATHS):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(WORKSPACE))
    prefixes = []
    current = os.environ.get("PATH", "")
    parts = current.split(":") if current else []
    for prefix in (str(Path.home() / "Documents/llm/qwen38-exl3-mia/.venv/bin"), "/opt/cuda/bin"):
        if prefix not in parts:
            prefixes.append(prefix)
    if prefixes:
        os.environ["PATH"] = ":".join(prefixes + ([current] if current else []))


def install_prims(min_q=PRIMS_MIN_QUERY):
    from qwasar_bench.prims_prefill import install
    return install(min_q)


def replace_mlps(profile="nvidia64"):
    from .nvidia_mlp import replace
    return replace(profile, donor_dir())


def graph_mlps(model):
    from .nvfp4_linear import graph_selected_mlps
    return graph_selected_mlps(model)


def attention_context():
    from qwasar_bench.attention_tuning import graph_attention_context
    return graph_attention_context(DECODE_POLICY)
