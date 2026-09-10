from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class GpuEnvironment:
    available: bool
    error: str | None
    index: int | None = None
    uuid: str | None = None
    name: str | None = None
    total_memory_mib: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    temperature_c: int | None = None
    power_limit_w: float | None = None
    sm_clock_mhz: int | None = None
    compute_capability: str | None = None


@dataclass(frozen=True, slots=True)
class CpuEnvironment:
    model_name: str | None
    logical_cpus: int | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    captured_at: str
    operating_system: str
    machine: str
    python_version: str
    gpu: GpuEnvironment
    cpu: CpuEnvironment
    engine_revision: str | None
    visible_gpu_count: int = 0
    backend_revision: str | None = None
    manifest_sha256: str | None = None
    artifact_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _successful_output(run_command: CommandRunner, command: list[str]) -> str:
    completed = run_command(command)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return completed.stdout.strip()


def _capture_gpu(
    run_command: CommandRunner, selected_gpu_index: int | None
) -> tuple[GpuEnvironment, int]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version,temperature.gpu,power.limit,clocks.sm,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        rows = [
            [field.strip() for field in line.split(",")]
            for line in _successful_output(run_command, query).splitlines()
            if line.strip()
        ]
        if not rows or any(len(fields) != 9 for fields in rows):
            raise ValueError("expected 9 fields for every nvidia-smi GPU row")
        if selected_gpu_index is None:
            fields = next(
                (row for row in rows if "RTX 5090" in row[2]),
                rows[0],
            )
        else:
            fields = next(
                (row for row in rows if int(row[0]) == selected_gpu_index),
                None,
            )
            if fields is None:
                raise ValueError(f"GPU index {selected_gpu_index} is not visible")
        overview = _successful_output(run_command, ["nvidia-smi"])
        cuda_match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9.]+)", overview)
        return (
            GpuEnvironment(
                available=True,
                error=None,
                index=int(fields[0]),
                uuid=fields[1],
                name=fields[2],
                total_memory_mib=int(fields[3]),
                driver_version=fields[4],
                cuda_version=cuda_match.group(1) if cuda_match else None,
                temperature_c=int(fields[5]),
                power_limit_w=float(fields[6]),
                sm_clock_mhz=int(fields[7]),
                compute_capability=fields[8],
            ),
            len(rows),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        return (
            GpuEnvironment(available=False, error=f"nvidia-smi unavailable: {error}"),
            0,
        )


def _capture_cpu(run_command: CommandRunner) -> CpuEnvironment:
    try:
        payload = json.loads(_successful_output(run_command, ["lscpu", "--json"]))
        rows = payload["lscpu"]
        values = {row["field"].rstrip(":"): row["data"] for row in rows}
        logical_cpus = values.get("CPU(s)")
        return CpuEnvironment(
            model_name=values.get("Model name"),
            logical_cpus=int(logical_cpus) if logical_cpus is not None else None,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return CpuEnvironment(model_name=None, logical_cpus=None, error=f"lscpu unavailable: {error}")
    except (OSError, RuntimeError) as error:
        return CpuEnvironment(model_name=None, logical_cpus=None, error=f"lscpu unavailable: {error}")


def _capture_git_revision(run_command: CommandRunner, repository_path: Path | None) -> str | None:
    command = ["git", "rev-parse", "HEAD"]
    if repository_path is not None and repository_path.resolve() != Path.cwd().resolve():
        command = ["git", "-C", str(repository_path), "rev-parse", "HEAD"]
    try:
        return _successful_output(run_command, command)
    except (FileNotFoundError, OSError, RuntimeError):
        return None


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    if source.is_file():
        paths = [source]
        root = source.parent
    elif source.is_dir():
        root = source
        paths = sorted(
            item
            for item in source.rglob("*")
            if item.is_file()
            and not {".cache", ".git"}.intersection(item.relative_to(root).parts)
            and not item.name.endswith((".part", ".lock", ".incomplete"))
        )
    else:
        raise FileNotFoundError(source)

    for item in paths:
        relative_path = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def capture_environment(
    *,
    run_command: CommandRunner = _run_command,
    repository_path: str | Path | None = None,
    backend_repository_path: str | Path | None = None,
    manifest_sha256: str | None = None,
    artifact_path: str | Path | None = None,
    selected_gpu_index: int | None = None,
) -> EnvironmentSnapshot:
    repository = Path(repository_path) if repository_path is not None else Path.cwd()
    backend_repository = (
        Path(backend_repository_path) if backend_repository_path is not None else None
    )
    gpu, visible_gpu_count = _capture_gpu(run_command, selected_gpu_index)
    return EnvironmentSnapshot(
        captured_at=datetime.now(UTC).isoformat(),
        operating_system=platform.platform(),
        machine=platform.machine(),
        python_version=sys.version.split()[0],
        gpu=gpu,
        cpu=_capture_cpu(run_command),
        engine_revision=_capture_git_revision(run_command, repository),
        visible_gpu_count=visible_gpu_count,
        backend_revision=_capture_git_revision(run_command, backend_repository)
        if backend_repository is not None
        else None,
        manifest_sha256=manifest_sha256,
        artifact_sha256=sha256_path(artifact_path) if artifact_path is not None else None,
    )
