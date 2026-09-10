from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from qwasar_bench.compare import compare_run_directories
from qwasar_bench.environment import EnvironmentSnapshot, capture_environment
from qwasar_bench.fixtures import FixturePromptBuilder, HuggingFaceCodec
from qwasar_bench.openai_client import ChatCompletionsClient, ResponsesClient
from qwasar_bench.runner import run_manifest
from qwasar_bench.schema import BenchmarkSample, RunMetadata
from qwasar_bench.workloads import BenchmarkManifest, load_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "benchmarks/manifests/qwen38-27b-rtx5090-v1.json"
DEFAULT_FIXTURES = REPOSITORY_ROOT / "benchmarks/fixtures/smoke.jsonl"
DEFAULT_DRAFT_MODEL = Path(
    "/home/rekeyea/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw"
)


def _parse_contexts(value: str) -> tuple[int, ...]:
    try:
        contexts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "contexts must be comma-separated integers"
        ) from error
    if not contexts or any(context <= 0 for context in contexts):
        raise argparse.ArgumentTypeError("contexts must be positive")
    if contexts != tuple(sorted(set(contexts))):
        raise argparse.ArgumentTypeError("contexts must be unique and increasing")
    return contexts


def build_streaming_client(
    protocol: str, base_url: str, timeout_seconds: float
) -> ResponsesClient | ChatCompletionsClient:
    if protocol == "responses":
        return ResponsesClient(base_url, timeout_seconds=timeout_seconds)
    if protocol == "chat-completions":
        return ChatCompletionsClient(base_url, timeout_seconds=timeout_seconds)
    raise ValueError(f"unsupported backend protocol: {protocol}")


def _json_text(values: Any) -> str:
    return json.dumps(values, indent=2, sort_keys=True) + "\n"


def _write_json_atomic(path: Path, values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_json_text(values), encoding="utf-8")
    temporary.replace(path)


def _model_path(manifest: BenchmarkManifest) -> Path | None:
    value = os.environ.get(manifest.model.artifact_path_env)
    return Path(value).expanduser() if value else None


def _doctor_report(
    manifest: BenchmarkManifest, *, qualification: str = "acceptance"
) -> tuple[dict[str, Any], EnvironmentSnapshot]:
    if qualification not in ("bringup", "acceptance"):
        raise ValueError(f"unsupported qualification: {qualification}")
    model_path = _model_path(manifest)
    model_path_ready = model_path is not None and model_path.exists()
    should_hash_artifact = model_path_ready and manifest.model.artifact_sha256 is not None
    selected_gpu_value = os.environ.get("QWASAR_CUDA_DEVICE")
    if selected_gpu_value is None:
        selected_gpu_index = None
    else:
        try:
            selected_gpu_index = int(selected_gpu_value)
        except ValueError as error:
            raise ValueError("QWASAR_CUDA_DEVICE must be a non-negative integer") from error
        if selected_gpu_index < 0:
            raise ValueError("QWASAR_CUDA_DEVICE must be a non-negative integer")
    snapshot = capture_environment(
        repository_path=REPOSITORY_ROOT,
        manifest_sha256=manifest.sha256,
        artifact_path=model_path if should_hash_artifact else None,
        selected_gpu_index=selected_gpu_index,
    )
    gpu_ready = (
        snapshot.gpu.available
        and snapshot.gpu.name is not None
        and "RTX 5090" in snapshot.gpu.name
        and snapshot.gpu.total_memory_mib is not None
        and snapshot.gpu.total_memory_mib >= 30_000
        and snapshot.gpu.compute_capability == "12.0"
    )
    artifact_pin_ready = manifest.model.artifact_sha256 is not None
    artifact_hash_ready = (
        artifact_pin_ready
        and snapshot.artifact_sha256 == manifest.model.artifact_sha256
    )
    engine_revision_ready = snapshot.engine_revision is not None
    quantization_ready = 4.5 <= manifest.model.target_bpw <= 5.0
    bringup_ready = all(
        (gpu_ready, model_path_ready, artifact_pin_ready, artifact_hash_ready)
    )
    acceptance_ready = bringup_ready and engine_revision_ready and quantization_ready
    ready = bringup_ready if qualification == "bringup" else acceptance_ready
    report = {
        "ready": ready,
        "qualification": qualification,
        "bringup_ready": bringup_ready,
        "acceptance_ready": acceptance_ready,
        "gpu_ready": gpu_ready,
        "model_path_ready": model_path_ready,
        "artifact_pin_ready": artifact_pin_ready,
        "artifact_hash_ready": artifact_hash_ready,
        "engine_revision_ready": engine_revision_ready,
        "quantization_ready": quantization_ready,
        "manifest_sha256": manifest.sha256,
        "model_path_env": manifest.model.artifact_path_env,
        "model_path": str(model_path) if model_path is not None else None,
        "environment": snapshot.to_dict(),
    }
    return report, snapshot


def _validate_run(run_directory: Path) -> dict[str, Any]:
    required = {
        "run.json",
        "environment.json",
        "manifest.json",
        "samples.jsonl",
        "events",
    }
    missing = sorted(name for name in required if not (run_directory / name).exists())
    if missing:
        raise ValueError(f"run is missing required artifacts: {', '.join(missing)}")
    metadata = RunMetadata.from_dict(
        json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    )
    manifest = load_manifest(run_directory / "manifest.json")
    samples = [
        BenchmarkSample.from_dict(json.loads(line))
        for line in (run_directory / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if metadata.manifest_sha256 != manifest.sha256:
        raise ValueError("run metadata manifest hash does not match manifest.json")
    if metadata.sample_count != len(samples):
        raise ValueError("run metadata sample_count does not match samples.jsonl")
    expected_counts = {case.case_id: case.repetitions for case in manifest.cases}
    actual_counts = Counter(sample.case_id for sample in samples)
    if actual_counts != expected_counts:
        raise ValueError(
            f"sample counts differ: expected {expected_counts}, got {dict(actual_counts)}"
        )
    event_count = len(list((run_directory / "events").glob("*.jsonl")))
    if event_count != len(samples):
        raise ValueError("raw SSE event file count does not match samples")
    return {
        "valid": True,
        "run_id": metadata.run_id,
        "manifest_sha256": manifest.sha256,
        "sample_count": len(samples),
        "event_file_count": event_count,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwasar-bench")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor")
    doctor.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    doctor.add_argument(
        "--qualification", choices=("bringup", "acceptance"), default="acceptance"
    )

    validate_manifest = subcommands.add_parser("validate-manifest")
    validate_manifest.add_argument("manifest", type=Path)

    baseline = subcommands.add_parser("baseline")
    baseline.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    baseline.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    baseline.add_argument("--base-url", required=True)
    baseline.add_argument("--api-model", default="qwen38")
    baseline.add_argument("--backend", default="exllamav3")
    baseline.add_argument(
        "--protocol",
        choices=("responses", "chat-completions"),
        default="responses",
    )
    baseline.add_argument("--tokenizer", type=Path)
    baseline.add_argument("--timeout", type=float, default=3600.0)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument(
        "--qualification", choices=("bringup", "acceptance"), default="acceptance"
    )

    compare = subcommands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, required=True)

    validate_run = subcommands.add_parser("validate-run")
    validate_run.add_argument("run_directory", type=Path)

    resident_probe = subcommands.add_parser("resident-probe")
    resident_probe.add_argument("--model", type=Path, required=True)
    resident_probe.add_argument(
        "--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL
    )
    resident_probe.add_argument("--output", type=Path, required=True)
    resident_probe.add_argument(
        "--contexts",
        type=_parse_contexts,
        default=(1024, 32768, 131072, 262144),
    )
    resident_probe.add_argument("--appended-tokens", type=int, default=128)
    resident_probe.add_argument("--max-new-tokens", type=int, default=1)
    resident_probe.add_argument("--repetitions", type=int, default=3)
    resident_probe.add_argument("--cache-size", type=int, default=262400)
    resident_probe.add_argument("--gpu-split-gb", type=float, default=30.0)
    resident_probe.add_argument(
        "--cache-quant",
        choices=("none", "4", "6", "8", "fp8", "nvfp4"),
        default="nvfp4",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "doctor":
            report, _ = _doctor_report(
                load_manifest(arguments.manifest),
                qualification=arguments.qualification,
            )
            sys.stdout.write(_json_text(report))
            return 0 if report["ready"] else 2
        if arguments.command == "validate-manifest":
            manifest = load_manifest(arguments.manifest)
            sys.stdout.write(
                _json_text(
                    {
                        "valid": True,
                        "version": manifest.version,
                        "sha256": manifest.sha256,
                        "case_count": len(manifest.cases),
                    }
                )
            )
            return 0
        if arguments.command == "baseline":
            manifest = load_manifest(arguments.manifest)
            report, environment = _doctor_report(
                manifest, qualification=arguments.qualification
            )
            if not report["ready"]:
                sys.stdout.write(_json_text(report))
                return 2
            model_path = _model_path(manifest)
            assert model_path is not None
            tokenizer_path = arguments.tokenizer or model_path / "tokenizer.json"
            codec = HuggingFaceCodec.from_file(tokenizer_path)
            prompt_builder = FixturePromptBuilder.from_jsonl(arguments.fixtures, codec)
            result = run_manifest(
                manifest,
                build_streaming_client(
                    arguments.protocol, arguments.base_url, arguments.timeout
                ),
                arguments.output,
                backend=(
                    f"{arguments.backend}:{arguments.protocol}:"
                    f"{arguments.qualification}"
                ),
                model=arguments.api_model,
                environment=environment,
                build_input=prompt_builder.build_input,
            )
            sys.stdout.write(
                _json_text(
                    {
                        "completed": True,
                        "output_directory": str(result.output_directory),
                        "sample_count": len(result.samples),
                    }
                )
            )
            return 0
        if arguments.command == "compare":
            report = compare_run_directories(arguments.baseline, arguments.candidate)
            _write_json_atomic(arguments.output, report.to_dict())
            sys.stdout.write(_json_text(report.to_dict()))
            return 0 if report.accepted else 3
        if arguments.command == "validate-run":
            report = _validate_run(arguments.run_directory)
            sys.stdout.write(_json_text(report))
            return 0
        if arguments.command == "resident-probe":
            from qwasar_bench.exllamav3_probe import run_probe

            report = run_probe(
                model_path=arguments.model,
                draft_model_path=arguments.draft_model,
                output_directory=arguments.output,
                context_tokens=arguments.contexts,
                appended_tokens=arguments.appended_tokens,
                max_new_tokens=arguments.max_new_tokens,
                repetitions=arguments.repetitions,
                cache_size=arguments.cache_size,
                cache_quant=arguments.cache_quant,
                gpu_split_gb=arguments.gpu_split_gb,
            )
            sys.stdout.write(_json_text(report))
            return 0
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        sys.stderr.write(_json_text({"error": str(error), "command": arguments.command}))
        return 2
    raise AssertionError(f"unhandled command: {arguments.command}")
