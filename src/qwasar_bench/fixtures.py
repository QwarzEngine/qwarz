from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from qwasar_bench.workloads import WorkloadCase


class TokenCodec(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class Fixture:
    fixture_id: str
    category: str
    input: str
    filler_seed: int


class HuggingFaceCodec:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> HuggingFaceCodec:
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "tokenizers is required for acceptance-grade prompt construction"
            ) from error
        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)


def load_fixtures(path: str | Path) -> dict[str, Fixture]:
    fixtures: dict[str, Fixture] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        values = json.loads(line)
        if not isinstance(values, dict):
            raise ValueError(f"fixture line {line_number} must be an object")
        allowed = {"fixture_id", "category", "input", "filler_seed"}
        if set(values) != allowed:
            raise ValueError(f"fixture line {line_number} must contain exactly {sorted(allowed)}")
        if any(not isinstance(values[key], str) or not values[key].strip() for key in allowed - {"filler_seed"}):
            raise ValueError(f"fixture line {line_number} contains an empty string")
        filler_seed = values["filler_seed"]
        if isinstance(filler_seed, bool) or not isinstance(filler_seed, int) or filler_seed < 0:
            raise ValueError(f"fixture line {line_number} filler_seed must be non-negative")
        fixture = Fixture(
            fixture_id=values["fixture_id"],
            category=values["category"],
            input=values["input"],
            filler_seed=filler_seed,
        )
        if fixture.fixture_id in fixtures:
            raise ValueError(f"duplicate fixture_id: {fixture.fixture_id}")
        fixtures[fixture.fixture_id] = fixture
    if not fixtures:
        raise ValueError("fixture file contains no fixtures")
    return fixtures


def _filler_text(seed: int) -> str:
    generator = random.Random(seed)
    words = [
        f"qwasar_{generator.randrange(1 << 32):08x}_{index:03d}"
        for index in range(256)
    ]
    return " ".join(words)


class FixturePromptBuilder:
    def __init__(self, fixtures: dict[str, Fixture], codec: TokenCodec) -> None:
        self._fixtures = fixtures
        self._codec = codec

    @classmethod
    def from_jsonl(
        cls, path: str | Path, codec: TokenCodec
    ) -> FixturePromptBuilder:
        return cls(load_fixtures(path), codec)

    def build_input(self, case: WorkloadCase) -> list[dict[str, str]]:
        try:
            fixture = self._fixtures[case.fixture_id]
        except KeyError as error:
            raise ValueError(f"unknown fixture_id: {case.fixture_id}") from error
        instruction_tokens = self._codec.encode(f"\n\n{fixture.input}")
        if len(instruction_tokens) > case.appended_tokens:
            raise ValueError(
                f"fixture {fixture.fixture_id} requires {len(instruction_tokens)} tokens, "
                f"but case {case.case_id} allows {case.appended_tokens}"
            )
        filler_tokens = self._codec.encode(_filler_text(fixture.filler_seed))
        if not filler_tokens and len(instruction_tokens) < case.appended_tokens:
            raise ValueError(f"fixture {fixture.fixture_id} produced no filler tokens")
        filler_count = case.appended_tokens - len(instruction_tokens)
        repeated_filler = (
            filler_tokens * ((filler_count + len(filler_tokens) - 1) // len(filler_tokens))
            if filler_count
            else []
        )
        token_ids = repeated_filler[:filler_count] + instruction_tokens
        content = self._codec.decode(token_ids)
        actual_tokens = len(self._codec.encode(content))
        if actual_tokens != case.appended_tokens:
            raise RuntimeError(
                f"tokenizer produced {actual_tokens} tokens for case {case.case_id}; "
                f"expected {case.appended_tokens}"
            )
        return [{"role": "user", "content": content}]
