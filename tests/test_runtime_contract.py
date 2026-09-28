import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from qwasar_runtime.runtime_contract import (
    GDN_SHA256,
    KERNEL_SHA256,
    LEGACY_FLASHINFER,
    missing_pieces,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_checkout_is_missing_both_runtime_pieces(tmp_path):
    assert missing_pieces(tmp_path, tmp_path / "absent-python") == ["exllamav3", "flashinfer-pscale"]


def test_legacy_trees_stand_in_for_the_flashinfer_package(tmp_path):
    for relative in LEGACY_FLASHINFER:
        (tmp_path / relative).mkdir(parents=True)
    assert missing_pieces(tmp_path, tmp_path / "absent-python") == ["exllamav3"]


def test_promoted_patches_reproduce_the_pinned_hashes(tmp_path):
    originals = {
        "gdn": Path.home() / "Documents/llm/qwen38-exl3-mia/.venv/lib/python3.12/site-packages/exllamav3/exllamav3_ext/gdn.cu.orig-pre-determinism",
        "kernel": ROOT / "results/20260908-upstream-experiments/fp8/pscaled/flashinfer/cute_dsl/attention/fmha/sm120/fmha_prefill_fp8_tma.py",
    }
    if not originals["gdn"].is_file() or not originals["kernel"].is_file() or shutil.which("patch") is None:
        pytest.skip("promoted originals or the patch command are not on this machine")
    gdn_dir = tmp_path / "gdn"
    kernel_dir = tmp_path / "flash"
    (gdn_dir / "exllamav3/exllamav3_ext").mkdir(parents=True)
    kernel_path = kernel_dir / "flashinfer/cute_dsl/attention/fmha/sm120"
    kernel_path.mkdir(parents=True)
    shutil.copy(originals["gdn"], gdn_dir / "exllamav3/exllamav3_ext/gdn.cu")
    # The stored kernel is already patched; reverse it back to the upstream bytes first.
    shutil.copy(originals["kernel"], kernel_path / "fmha_prefill_fp8_tma.py")
    subprocess.run(
        ["patch", "-p1", "--reverse", "--batch", "-d", str(kernel_dir), "-i", str(ROOT / "runtime/patches/flashinfer-pscale.patch")],
        check=True,
    )
    subprocess.run(
        ["patch", "-p1", "--forward", "--batch", "-d", str(gdn_dir), "-i", str(ROOT / "runtime/patches/gdn-determinism.patch")],
        check=True,
    )
    subprocess.run(
        ["patch", "-p1", "--forward", "--batch", "-d", str(kernel_dir), "-i", str(ROOT / "runtime/patches/flashinfer-pscale.patch")],
        check=True,
    )
    assert hashlib.sha256((gdn_dir / "exllamav3/exllamav3_ext/gdn.cu").read_bytes()).hexdigest() == GDN_SHA256
    assert hashlib.sha256((kernel_path / "fmha_prefill_fp8_tma.py").read_bytes()).hexdigest() == KERNEL_SHA256


def test_prepare_environment_uses_the_venv_when_research_trees_are_absent(tmp_path, monkeypatch):
    from qwasar_runtime import hybrid

    monkeypatch.setattr(hybrid, "ROOT", tmp_path)
    monkeypatch.setattr(hybrid, "FLASHINFER_PATHS", (tmp_path / "missing-tree",))
    monkeypatch.setattr(hybrid, "installed_flashinfer_ready", lambda: True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("FLASHINFER_WORKSPACE_BASE", raising=False)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    hybrid.prepare_environment()
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(tmp_path / "state" / "flashinfer-workspace")
    assert str(tmp_path / "missing-tree") not in sys.path


def test_prepare_environment_names_qwarz_start_when_nothing_is_installed(tmp_path, monkeypatch):
    from qwasar_runtime import hybrid

    monkeypatch.setattr(hybrid, "FLASHINFER_PATHS", (tmp_path / "missing-tree",))
    monkeypatch.setattr(hybrid, "installed_flashinfer_ready", lambda: False)
    with pytest.raises(ValueError, match="qwarz start"):
        hybrid.prepare_environment()
