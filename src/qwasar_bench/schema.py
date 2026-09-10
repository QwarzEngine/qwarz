from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Self

SampleStatus = Literal["completed", "failed", "cancelled"]


def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_non_negative(name: str, value: int | float | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_sha256(name: str, value: str | None) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    case_id: str
    backend: str
    model: str
    context_bucket: str
    prompt_tokens: int
    reused_prefix_tokens: int | None
    output_tokens: int
    time_to_first_delta_ms: float
    elapsed_ms: float
    status: SampleStatus
    prefill_elapsed_ms: float | None = None
    decode_elapsed_ms: float | None = None
    peak_vram_bytes: int | None = None
    error: str | None = None
    draft_proposed_tokens: int | None = None
    draft_accepted_tokens: int | None = None
    artifact_sha256: str | None = None
    engine_revision: str | None = None

    def __post_init__(self) -> None:
        for name in ("case_id", "backend", "model", "context_bucket"):
            _require_non_empty(name, getattr(self, name))

        for name in (
            "prompt_tokens",
            "reused_prefix_tokens",
            "output_tokens",
            "time_to_first_delta_ms",
            "elapsed_ms",
            "prefill_elapsed_ms",
            "decode_elapsed_ms",
            "peak_vram_bytes",
            "draft_proposed_tokens",
            "draft_accepted_tokens",
        ):
            _require_non_negative(name, getattr(self, name))

        if (
            self.reused_prefix_tokens is not None
            and self.reused_prefix_tokens > self.prompt_tokens
        ):
            raise ValueError("reused_prefix_tokens cannot exceed prompt_tokens")
        if self.status not in ("completed", "failed", "cancelled"):
            raise ValueError(f"unsupported status: {self.status}")
        if self.status != "completed" and not self.error:
            raise ValueError("error is required for failed or cancelled samples")
        if (
            self.draft_accepted_tokens is not None
            and self.draft_proposed_tokens is not None
            and self.draft_accepted_tokens > self.draft_proposed_tokens
        ):
            raise ValueError("draft_accepted_tokens cannot exceed draft_proposed_tokens")
        _validate_sha256("artifact_sha256", self.artifact_sha256)

    @property
    def prefill_tokens(self) -> int | None:
        if self.reused_prefix_tokens is None:
            return None
        return self.prompt_tokens - self.reused_prefix_tokens

    @property
    def prefill_tokens_per_second(self) -> float | None:
        if self.prefill_tokens is None or self.prefill_elapsed_ms in (None, 0):
            return None
        return self.prefill_tokens * 1000.0 / self.prefill_elapsed_ms

    @property
    def accepted_decode_tokens_per_second(self) -> float | None:
        if self.decode_elapsed_ms in (None, 0):
            return None
        return self.output_tokens * 1000.0 / self.decode_elapsed_ms

    @property
    def draft_acceptance_ratio(self) -> float | None:
        if self.draft_proposed_tokens in (None, 0) or self.draft_accepted_tokens is None:
            return None
        return self.draft_accepted_tokens / self.draft_proposed_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    manifest_sha256: str
    backend: str
    model: str
    started_at: str
    completed_at: str | None
    engine_revision: str | None
    artifact_sha256: str | None
    sample_count: int

    def __post_init__(self) -> None:
        for name in ("run_id", "backend", "model", "started_at"):
            _require_non_empty(name, getattr(self, name))
        _require_non_negative("sample_count", self.sample_count)
        _validate_sha256("manifest_sha256", self.manifest_sha256)
        _validate_sha256("artifact_sha256", self.artifact_sha256)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        return cls(**values)
