from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from qwasar_bench.environment import EnvironmentSnapshot
from qwasar_bench.openai_client import StreamResult
from qwasar_bench.schema import BenchmarkSample, RunMetadata
from qwasar_bench.workloads import BenchmarkManifest, WorkloadCase


class StreamingClient(Protocol):
    def create_stream(self, **kwargs: Any) -> StreamResult: ...


InputBuilder = Callable[[WorkloadCase], list[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class RunResult:
    output_directory: Path
    metadata: RunMetadata
    samples: tuple[BenchmarkSample, ...]


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())


def _write_json(path: Path, values: Any) -> None:
    _write_text(path, json.dumps(values, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
    )


def _manifest_dict(manifest: BenchmarkManifest) -> dict[str, Any]:
    values = asdict(manifest)
    values.pop("sha256")
    return values


def _metrics(events: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for event in events:
        if event.get("type") in ("qwasar.metrics", "response.qwasar.metrics"):
            merged.update({key: value for key, value in event.items() if key != "type"})
    return merged


def _optional_number(values: dict[str, Any], key: str) -> float | None:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int(values: dict[str, Any], key: str) -> int | None:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _sample_from_result(
    case: WorkloadCase,
    result: StreamResult,
    *,
    backend: str,
    model: str,
    environment: EnvironmentSnapshot,
) -> BenchmarkSample:
    metrics = _metrics(result.events)
    first_delta_ns = result.first_delta_ns or result.completed_ns
    reused_prefix_tokens = _optional_int(metrics, "reused_prefix_tokens")
    return BenchmarkSample(
        case_id=case.case_id,
        backend=backend,
        model=model,
        context_bucket=case.context_bucket,
        prompt_tokens=result.input_tokens,
        reused_prefix_tokens=reused_prefix_tokens,
        output_tokens=result.output_tokens,
        time_to_first_delta_ms=(first_delta_ns - result.request_started_ns) / 1_000_000,
        elapsed_ms=(result.completed_ns - result.request_started_ns) / 1_000_000,
        status="completed",
        prefill_elapsed_ms=_optional_number(metrics, "prefill_elapsed_ms"),
        decode_elapsed_ms=(result.completed_ns - result.first_delta_ns) / 1_000_000
        if result.first_delta_ns is not None
        else None,
        peak_vram_bytes=_optional_int(metrics, "peak_vram_bytes"),
        draft_proposed_tokens=_optional_int(metrics, "draft_proposed_tokens"),
        draft_accepted_tokens=_optional_int(metrics, "draft_accepted_tokens"),
        artifact_sha256=environment.artifact_sha256,
        engine_revision=environment.engine_revision,
    )


def _invoke(
    case: WorkloadCase,
    client: StreamingClient,
    manifest: BenchmarkManifest,
    build_input: InputBuilder,
    previous_response_id: str | None,
    api_model: str,
) -> StreamResult:
    arguments: dict[str, Any] = {
        "model": api_model,
        "input_items": build_input(case),
        "max_output_tokens": manifest.generation.max_output_tokens,
        "temperature": manifest.generation.temperature,
        "top_p": manifest.generation.top_p,
        "seed": manifest.generation.seed,
    }
    if previous_response_id is not None:
        arguments["previous_response_id"] = previous_response_id
    return client.create_stream(**arguments)


def _run_fresh_cases(
    manifest: BenchmarkManifest,
    client: StreamingClient,
    build_input: InputBuilder,
    environment: EnvironmentSnapshot,
    backend: str,
    model: str,
    event_rows: list[tuple[str, tuple[dict[str, Any], ...]]],
) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    for case in (case for case in manifest.cases if case.mode == "fresh_prefill"):
        for _ in range(case.warmups):
            _invoke(case, client, manifest, build_input, None, model)
        for repetition in range(case.repetitions):
            result = _invoke(case, client, manifest, build_input, None, model)
            samples.append(
                _sample_from_result(
                    case,
                    result,
                    backend=backend,
                    model=model,
                    environment=environment,
                )
            )
            event_rows.append((f"{case.case_id}-{repetition}", result.events))
    return samples


def _run_persistent_sessions(
    manifest: BenchmarkManifest,
    client: StreamingClient,
    build_input: InputBuilder,
    environment: EnvironmentSnapshot,
    backend: str,
    model: str,
    event_rows: list[tuple[str, tuple[dict[str, Any], ...]]],
) -> list[BenchmarkSample]:
    sessions: dict[str, list[WorkloadCase]] = {}
    for case in manifest.cases:
        if case.session_id is not None:
            sessions.setdefault(case.session_id, []).append(case)

    samples: list[BenchmarkSample] = []
    for turns in sessions.values():
        turns.sort(key=lambda case: case.turn_index or 0)
        for _ in range(max(case.warmups for case in turns)):
            previous_response_id: str | None = None
            for case in turns:
                result = _invoke(
                    case, client, manifest, build_input, previous_response_id, model
                )
                previous_response_id = result.response_id
        for repetition in range(max(case.repetitions for case in turns)):
            previous_response_id = None
            for case in turns:
                result = _invoke(
                    case, client, manifest, build_input, previous_response_id, model
                )
                previous_response_id = result.response_id
                if repetition >= case.repetitions:
                    continue
                samples.append(
                    _sample_from_result(
                        case,
                        result,
                        backend=backend,
                        model=model,
                        environment=environment,
                    )
                )
                event_rows.append((f"{case.case_id}-{repetition}", result.events))
    return samples


def run_manifest(
    manifest: BenchmarkManifest,
    client: StreamingClient,
    output_directory: str | Path,
    *,
    backend: str,
    model: str,
    environment: EnvironmentSnapshot,
    build_input: InputBuilder,
) -> RunResult:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    started_at = datetime.now(UTC).isoformat()
    event_rows: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    try:
        samples = _run_fresh_cases(
            manifest,
            client,
            build_input,
            environment,
            backend,
            model,
            event_rows,
        )
        samples.extend(
            _run_persistent_sessions(
                manifest,
                client,
                build_input,
                environment,
                backend,
                model,
                event_rows,
            )
        )
        completed_at = datetime.now(UTC).isoformat()
        metadata = RunMetadata(
            run_id=output.name,
            manifest_sha256=manifest.sha256,
            backend=backend,
            model=model,
            started_at=started_at,
            completed_at=completed_at,
            engine_revision=environment.engine_revision,
            artifact_sha256=environment.artifact_sha256,
            sample_count=len(samples),
        )
        events_directory = temporary / "events"
        events_directory.mkdir()
        _write_json(temporary / "run.json", metadata.to_dict())
        _write_json(temporary / "environment.json", environment.to_dict())
        _write_json(temporary / "manifest.json", _manifest_dict(manifest))
        _write_jsonl(temporary / "samples.jsonl", [sample.to_dict() for sample in samples])
        for event_name, events in event_rows:
            _write_jsonl(events_directory / f"{event_name}.jsonl", events)
        temporary.rename(output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return RunResult(output_directory=output, metadata=metadata, samples=tuple(samples))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
