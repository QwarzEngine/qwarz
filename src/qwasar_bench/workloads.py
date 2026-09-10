from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ContextBucket = Literal["1k", "32k", "128k", "256k"]
WorkloadMode = Literal["fresh_prefill", "persistent_turn"]

_BUCKET_LIMITS: dict[str, tuple[int, int]] = {
    "1k": (1, 1024),
    "32k": (1025, 32768),
    "128k": (32769, 131072),
    "256k": (131073, 262144),
}


def _require_keys(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{section} contains unknown keys: {', '.join(unknown)}")
    missing = sorted(allowed - set(values))
    if missing:
        raise ValueError(f"{section} is missing keys: {', '.join(missing)}")


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_number(name: str, value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ValueError(f"{name} must be a number >= {minimum}")
    return float(value)


def _require_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ModelPin:
    architecture: str
    repository: str
    quantization: str
    target_bpw: float
    native_context_tokens: int
    artifact_path_env: str
    artifact_sha256: str | None


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float
    top_p: float
    max_output_tokens: int
    seed: int


@dataclass(frozen=True, slots=True)
class WorkloadCase:
    case_id: str
    fixture_id: str
    mode: WorkloadMode
    context_bucket: ContextBucket
    target_context_tokens: int
    appended_tokens: int
    warmups: int
    repetitions: int
    session_id: str | None
    turn_index: int | None


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    version: int
    model: ModelPin
    generation: GenerationConfig
    cases: tuple[WorkloadCase, ...]
    sha256: str


def _parse_model(values: Any) -> ModelPin:
    if not isinstance(values, dict):
        raise ValueError("model must be an object")
    allowed = {
        "architecture",
        "repository",
        "quantization",
        "target_bpw",
        "native_context_tokens",
        "artifact_path_env",
        "artifact_sha256",
    }
    _require_keys("model", values, allowed)
    artifact_sha256 = values["artifact_sha256"]
    if artifact_sha256 is not None:
        artifact_sha256 = _require_string("model.artifact_sha256", artifact_sha256)
        if len(artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in artifact_sha256
        ):
            raise ValueError("model.artifact_sha256 must be a lowercase SHA-256 digest")
    return ModelPin(
        architecture=_require_string("model.architecture", values["architecture"]),
        repository=_require_string("model.repository", values["repository"]),
        quantization=_require_string("model.quantization", values["quantization"]),
        target_bpw=_require_number("model.target_bpw", values["target_bpw"], minimum=0.1),
        native_context_tokens=_require_int(
            "model.native_context_tokens", values["native_context_tokens"], minimum=1
        ),
        artifact_path_env=_require_string(
            "model.artifact_path_env", values["artifact_path_env"]
        ),
        artifact_sha256=artifact_sha256,
    )


def _parse_generation(values: Any) -> GenerationConfig:
    if not isinstance(values, dict):
        raise ValueError("generation must be an object")
    allowed = {"temperature", "top_p", "max_output_tokens", "seed"}
    _require_keys("generation", values, allowed)
    top_p = _require_number("generation.top_p", values["top_p"], minimum=0.0)
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("generation.top_p must be in (0, 1]")
    return GenerationConfig(
        temperature=_require_number(
            "generation.temperature", values["temperature"], minimum=0.0
        ),
        top_p=top_p,
        max_output_tokens=_require_int(
            "generation.max_output_tokens", values["max_output_tokens"], minimum=1
        ),
        seed=_require_int("generation.seed", values["seed"], minimum=0),
    )


def _parse_case(index: int, values: Any, native_context_tokens: int) -> WorkloadCase:
    if not isinstance(values, dict):
        raise ValueError(f"cases[{index}] must be an object")
    allowed = {
        "case_id",
        "fixture_id",
        "mode",
        "context_bucket",
        "target_context_tokens",
        "appended_tokens",
        "warmups",
        "repetitions",
        "session_id",
        "turn_index",
    }
    _require_keys(f"cases[{index}]", values, allowed)
    mode = values["mode"]
    if mode not in ("fresh_prefill", "persistent_turn"):
        raise ValueError(f"cases[{index}].mode is unsupported: {mode}")
    bucket = values["context_bucket"]
    if bucket not in _BUCKET_LIMITS:
        raise ValueError(f"cases[{index}].context_bucket is unsupported: {bucket}")
    target_context_tokens = _require_int(
        f"cases[{index}].target_context_tokens",
        values["target_context_tokens"],
        minimum=1,
    )
    lower, upper = _BUCKET_LIMITS[bucket]
    if not lower <= target_context_tokens <= upper:
        raise ValueError(
            f"cases[{index}].target_context_tokens does not fit bucket {bucket}"
        )
    if target_context_tokens > native_context_tokens:
        raise ValueError(f"cases[{index}] exceeds the model native context")
    appended_tokens = _require_int(
        f"cases[{index}].appended_tokens", values["appended_tokens"], minimum=1
    )
    if appended_tokens > target_context_tokens:
        raise ValueError(f"cases[{index}].appended_tokens exceeds target context")
    session_id = values["session_id"]
    turn_index = values["turn_index"]
    if mode == "fresh_prefill":
        if session_id is not None or turn_index is not None:
            raise ValueError(f"cases[{index}] fresh_prefill cannot have session metadata")
    else:
        session_id = _require_string(f"cases[{index}].session_id", session_id)
        turn_index = _require_int(f"cases[{index}].turn_index", turn_index, minimum=0)
    return WorkloadCase(
        case_id=_require_string(f"cases[{index}].case_id", values["case_id"]),
        fixture_id=_require_string(f"cases[{index}].fixture_id", values["fixture_id"]),
        mode=mode,
        context_bucket=bucket,
        target_context_tokens=target_context_tokens,
        appended_tokens=appended_tokens,
        warmups=_require_int(f"cases[{index}].warmups", values["warmups"], minimum=0),
        repetitions=_require_int(
            f"cases[{index}].repetitions", values["repetitions"], minimum=1
        ),
        session_id=session_id,
        turn_index=turn_index,
    )


def _validate_cases(cases: tuple[WorkloadCase, ...]) -> None:
    case_ids = [case.case_id for case in cases]
    duplicate_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate case_id values: {', '.join(duplicate_ids)}")

    sessions: dict[str, list[WorkloadCase]] = {}
    for case in cases:
        if case.session_id is not None:
            sessions.setdefault(case.session_id, []).append(case)
    for session_id, turns in sessions.items():
        ordered = sorted(turns, key=lambda case: case.turn_index or 0)
        indices = [case.turn_index for case in ordered]
        if indices != list(range(len(ordered))):
            raise ValueError(f"persistent session {session_id} must begin at turn_index 0")
        previous_target = 0
        bucket = ordered[0].context_bucket
        for turn in ordered:
            if turn.context_bucket != bucket:
                raise ValueError(f"persistent session {session_id} cannot change context bucket")
            if turn.target_context_tokens <= previous_target:
                raise ValueError(f"persistent session {session_id} target context must grow")
            if turn.appended_tokens != turn.target_context_tokens - previous_target:
                raise ValueError(
                    f"persistent session {session_id} appended_tokens must match context growth"
                )
            previous_target = turn.target_context_tokens


def _load_manifest_values(manifest_path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    resolved_path = manifest_path.resolve()
    if resolved_path in seen:
        raise ValueError(f"manifest inheritance cycle at {resolved_path}")
    raw = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be an object")
    if "extends" not in raw:
        return raw
    _require_keys("manifest profile", raw, {"extends", "model"})
    parent_name = _require_string("manifest profile.extends", raw["extends"])
    parent = _load_manifest_values(
        resolved_path.parent / parent_name,
        seen | {resolved_path},
    )
    merged = dict(parent)
    merged["model"] = raw["model"]
    return merged


def load_manifest(path: str | Path) -> BenchmarkManifest:
    manifest_path = Path(path)
    raw = _load_manifest_values(manifest_path, frozenset())
    _require_keys("manifest", raw, {"version", "model", "generation", "cases"})
    version = _require_int("version", raw["version"], minimum=1)
    if version != 1:
        raise ValueError(f"unsupported manifest version: {version}")
    model = _parse_model(raw["model"])
    generation = _parse_generation(raw["generation"])
    case_values = raw["cases"]
    if not isinstance(case_values, list) or not case_values:
        raise ValueError("cases must be a non-empty array")
    cases = tuple(
        _parse_case(index, values, model.native_context_tokens)
        for index, values in enumerate(case_values)
    )
    _validate_cases(cases)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return BenchmarkManifest(
        version=version,
        model=model,
        generation=generation,
        cases=cases,
        sha256=sha256,
    )
