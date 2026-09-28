#!/usr/bin/env python3
"""Create the ExLlamaV3 venv Qwarz serves with, and bring it up to the contract.

``check`` only looks. ``install`` creates the venv when ``--python`` is missing,
then installs the pinned torch, ExLlamaV3 and FlashInfer builds and applies the
two promoted source patches. A venv that already matches is left alone, so a
second ``qwarz start`` does not recompile.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwasar_runtime.runtime_contract import (  # noqa: E402
    CUTLASS_SPEC,
    EXLLAMA_COMMIT,
    EXLLAMA_REPOSITORY,
    FLASHINFER_SPEC,
    GDN_SHA256,
    GDN_UPSTREAM_SHA256,
    KERNEL_RELATIVE,
    KERNEL_SHA256,
    KERNEL_UPSTREAM_SHA256,
    TORCH_INDEX,
    TORCH_SPEC,
    legacy_trees_ready,
    missing_pieces,
    sha256_file,
)


GDN_PATCH = ROOT / "runtime/patches/gdn-determinism.patch"
KERNEL_PATCH = ROOT / "runtime/patches/flashinfer-pscale.patch"


def _run(command, *, env=None, cwd=None):
    print("    $ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=cwd)


def _apply_patch(site_packages: Path, patch: Path):
    if shutil.which("patch") is None:
        raise SystemExit("the `patch` command is required to install the runtime")
    _run(["patch", "-p1", "--forward", "--batch", "-d", str(site_packages), "-i", str(patch)])


def _site_packages(python: Path) -> Path:
    output = subprocess.check_output(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    )
    return Path(output.strip())


def _package_file(python: Path, module: str, relative: Path) -> Path | None:
    probe = (
        "import importlib.util, pathlib, sys\n"
        f"spec = importlib.util.find_spec({module!r})\n"
        "if spec is None or not spec.origin:\n"
        "    sys.exit(2)\n"
        f"print(pathlib.Path(spec.origin).resolve().parent / {str(relative)!r})\n"
    )
    try:
        output = subprocess.check_output(
            [str(python), "-c", probe],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    return Path(output.strip())


def _venv_interpreter() -> Path:
    for name in ("python3.12", "python3"):
        path = shutil.which(name)
        if path is None:
            continue
        output = subprocess.check_output([path, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"], text=True)
        major, minor = (int(part) for part in output.strip().split("."))
        if (major, minor) == (3, 12):
            return Path(path)
    raise SystemExit(
        "Python 3.12 is required to create the runtime venv "
        "(torch and exllamav3 are pinned to it); install python3.12 and re-run"
    )


def _ensure_venv(python: Path) -> Path:
    if python.is_file():
        return python
    if python.parent.name != "bin":
        raise SystemExit(f"{python} is not a venv interpreter path (expected …/bin/python)")
    creator = _venv_interpreter()
    root = python.parent.parent
    print(f"    creating {root} with {creator}", flush=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    _run([str(creator), "-m", "venv", str(root)])
    if not python.is_file():
        raise SystemExit(f"venv created at {root} but {python} is missing")
    return python


def _pip(python: Path, arguments, *, env=None):
    _run([str(python), "-m", "pip", "install", *arguments], env=env)


def _compile_environment(cuda_home: Path) -> dict:
    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(cuda_home)
    environment.setdefault("TORCH_CUDA_ARCH_LIST", "12.0+PTX")
    prefix = str(cuda_home / "bin")
    environment["PATH"] = prefix + os.pathsep + environment.get("PATH", "")
    return environment


def _require_python_312(python: Path) -> None:
    output = subprocess.check_output(
        [str(python), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        text=True,
    ).strip()
    if output != "3.12":
        raise SystemExit(f"{python} is Python {output}; the runtime requires 3.12")


def _install_exllamav3(python: Path, cuda_home: Path):
    gdn = _package_file(python, "exllamav3", Path("exllamav3_ext/gdn.cu"))
    if gdn is not None and gdn.is_file() and sha256_file(gdn) == GDN_SHA256:
        print("    ExLlamaV3 GDN patch already installed", flush=True)
        return
    site = _site_packages(python)
    if gdn is None or not gdn.is_file():
        source = ROOT / "state" / "bootstrap" / "exllamav3"
        if not (source / ".git").is_dir():
            if source.exists():
                shutil.rmtree(source)
            source.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "init", str(source)])
            _run(["git", "-C", str(source), "remote", "add", "origin", EXLLAMA_REPOSITORY])
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPTS"] = "0"
        _run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", EXLLAMA_COMMIT], env=environment)
        _run(["git", "-C", str(source), "checkout", "--force", "FETCH_HEAD"])
        checkout = source / "exllamav3" / "exllamav3_ext" / "gdn.cu"
        if sha256_file(checkout) != GDN_UPSTREAM_SHA256:
            raise SystemExit(
                f"exllamav3 {EXLLAMA_COMMIT} gdn.cu is {sha256_file(checkout)}, "
                f"not the promoted upstream {GDN_UPSTREAM_SHA256}; refusing to patch it"
            )
        _apply_patch(source, GDN_PATCH)
        if sha256_file(checkout) != GDN_SHA256:
            raise SystemExit("GDN patch did not reproduce the promoted kernel hash")
        _pip(python, ["--no-build-isolation", str(source)], env=_compile_environment(cuda_home))
        return
    if sha256_file(gdn) != GDN_UPSTREAM_SHA256:
        raise SystemExit(
            f"installed gdn.cu is {sha256_file(gdn)}, neither the promoted patch "
            f"nor the known upstream; refusing to overwrite it"
        )
    _apply_patch(site, GDN_PATCH)
    if sha256_file(gdn) != GDN_SHA256:
        raise SystemExit("GDN patch did not reproduce the promoted kernel hash")
    for built in site.glob("exllamav3_ext*.so"):
        built.unlink()
    compiled = subprocess.check_output(
        [str(python), "-c", "import exllamav3.ext as extension; print(extension.exllamav3_ext.__file__)"],
        text=True,
        env=_compile_environment(cuda_home),
    ).strip().splitlines()[-1]
    suffix = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"],
        text=True,
    ).strip()
    destination = site / f"exllamav3_ext{suffix}"
    shutil.copy2(compiled, destination)
    print(f"    rebuilt {destination.name}", flush=True)


def _install_flashinfer(python: Path):
    kernel = _package_file(python, "flashinfer", KERNEL_RELATIVE)
    if kernel is not None and kernel.is_file() and sha256_file(kernel) == KERNEL_SHA256:
        if _package_file(python, "nvidia_cutlass_dsl", Path("__init__.py")) is not None:
            print("    FlashInfer P×256 patch already installed", flush=True)
            return
    _pip(python, [CUTLASS_SPEC, FLASHINFER_SPEC])
    kernel = _package_file(python, "flashinfer", KERNEL_RELATIVE)
    if kernel is None or not kernel.is_file():
        raise SystemExit("flashinfer installed but the SM120 FP8 kernel file is absent")
    digest = sha256_file(kernel)
    if digest == KERNEL_SHA256:
        return
    if digest != KERNEL_UPSTREAM_SHA256:
        raise SystemExit(
            f"installed FlashInfer kernel is {digest}, not {FLASHINFER_COMMIT}; refusing to patch it"
        )
    site = _site_packages(python)
    _apply_patch(site, KERNEL_PATCH)
    if sha256_file(kernel) != KERNEL_SHA256:
        raise SystemExit("P×256 patch did not reproduce the promoted kernel hash")


def install(python: Path, cuda_home: Path) -> None:
    python = _ensure_venv(python)
    _require_python_312(python)
    if not cuda_home.joinpath("bin/nvcc").is_file():
        raise SystemExit(f"nvcc is missing under {cuda_home}; the runtime compile needs the CUDA toolkit")
    _pip(python, ["--upgrade", "pip", "setuptools", "wheel", "ninja", "packaging", "huggingface_hub"])
    if subprocess.run([str(python), "-c", "import torch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        _pip(python, [TORCH_SPEC, "--extra-index-url", TORCH_INDEX])
    _install_exllamav3(python, cuda_home)
    if not legacy_trees_ready(ROOT):
        _install_flashinfer(python)
    missing = missing_pieces(ROOT, python)
    if missing:
        raise SystemExit("runtime install finished short of the contract: " + ", ".join(missing))
    print("ready venv", flush=True)


def check(python: Path) -> int:
    missing = missing_pieces(ROOT, python if python.is_file() else None)
    if missing:
        print("missing " + ",".join(missing))
        return 10
    print("ready legacy-trees" if legacy_trees_ready(ROOT) else "ready venv")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "install"))
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    if args.repo is not None and args.repo.resolve() != ROOT:
        raise SystemExit(f"--repo must be {ROOT}")
    if args.command == "check":
        raise SystemExit(check(args.python))
    if args.cuda_home is None:
        raise SystemExit("install requires --cuda-home")
    install(args.python, args.cuda_home)


if __name__ == "__main__":
    main()
