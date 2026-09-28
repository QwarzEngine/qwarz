"""What a production runtime has to contain before Qwarz will serve.

The numbers are the promoted builds, not suggestions. ``gdn.cu`` is the
fixed-order reduction patched onto ExLlamaV3 ``63b32f0``. The FlashInfer
kernel is commit ``af78f8fc`` plus the P×256 probability scale. Either the
venv carries that kernel, or the legacy research trees are still on disk.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


GDN_SHA256 = "969eaec94a66feac8f5bb992666e02615dbd7efd381136bcddee16d3a1848541"
GDN_UPSTREAM_SHA256 = "5e8327c30bdd1bd7d51a51674767294f734cde016586251029414f3dde2ed077"
KERNEL_SHA256 = "bc5a71bd588ac20abe59036d7c1f80d8db91ba7d4c7fe48aa0b5b7e77eadfffb"
KERNEL_UPSTREAM_SHA256 = "93d739b5b5d50c873bb6e403230f5f82c108e974c51b69607d2d8679eeeb84d3"

EXLLAMA_COMMIT = "63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed"
EXLLAMA_REPOSITORY = "https://github.com/MiaAI-Lab/exllamav3.git"
FLASHINFER_COMMIT = "af78f8fc17a9654563a619974de56e79257aaea6"
FLASHINFER_SPEC = (
    "flashinfer-python[cu13] @ "
    f"git+https://github.com/flashinfer-ai/flashinfer.git@{FLASHINFER_COMMIT}"
)
CUTLASS_SPEC = "nvidia-cutlass-dsl[cu13]==4.7.1"
TORCH_SPEC = "torch==2.14.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

KERNEL_RELATIVE = Path("cute_dsl/attention/fmha/sm120/fmha_prefill_fp8_tma.py")
LEGACY_FLASHINFER = (
    "results/20260908-upstream-experiments/fp8/pscaled",
    "results/20260908-upstream-experiments/fp8/deps",
    "results/20260908-upstream-experiments/fp8/deps/nvidia_cutlass_dsl/dsl_packages",
    "results/20260908-hybrid-backends/flashinfer-deps",
    "results/20260908-hybrid-backends/flashinfer-deps/nvidia_cutlass_dsl/dsl_packages",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_trees_ready(repo: Path) -> bool:
    return all((repo / relative).is_dir() for relative in LEGACY_FLASHINFER)


def _package_dir(python: Path | None, name: str) -> Path | None:
    if python is None or not python.is_file():
        return None
    import subprocess
    probe = (
        "import importlib.util, pathlib, sys\n"
        f"spec = importlib.util.find_spec({name!r})\n"
        "if spec is None or not spec.origin:\n"
        "    sys.exit(2)\n"
        "print(pathlib.Path(spec.origin).resolve().parent)\n"
    )
    try:
        output = subprocess.check_output(
            [str(python), "-c", probe],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    path = Path(output.strip())
    return path if path.is_dir() else None


def missing_pieces(repo: Path, python: Path | None) -> list[str]:
    """Names of production pieces that are absent or not the promoted bytes."""
    missing = []
    package = _package_dir(python, "exllamav3")
    gdn = package / "exllamav3_ext" / "gdn.cu" if package else None
    if gdn is None or not gdn.is_file():
        missing.append("exllamav3")
    elif sha256_file(gdn) != GDN_SHA256:
        missing.append("gdn-determinism")
    if legacy_trees_ready(repo):
        return missing
    flash = _package_dir(python, "flashinfer")
    kernel = flash / KERNEL_RELATIVE if flash else None
    if kernel is None or not kernel.is_file() or sha256_file(kernel) != KERNEL_SHA256:
        missing.append("flashinfer-pscale")
    elif _package_dir(python, "nvidia_cutlass_dsl") is None:
        missing.append("cutlass-dsl")
    return missing


def installed_flashinfer_ready() -> bool:
    """The venv itself holds the patched kernel and CUTLASS DSL 4.7."""
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.origin:
        return False
    kernel = Path(spec.origin).resolve().parent / KERNEL_RELATIVE
    if not kernel.is_file() or sha256_file(kernel) != KERNEL_SHA256:
        return False
    return importlib.util.find_spec("nvidia_cutlass_dsl") is not None
