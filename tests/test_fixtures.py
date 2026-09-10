from __future__ import annotations

from pathlib import Path

from qwasar_bench.fixtures import FixturePromptBuilder
from qwasar_bench.workloads import WorkloadCase


class WhitespaceCodec:
    def __init__(self) -> None:
        self.tokens: dict[str, int] = {}
        self.reverse: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        for token in text.split():
            if token not in self.tokens:
                identifier = len(self.tokens) + 1
                self.tokens[token] = identifier
                self.reverse[identifier] = token
            encoded.append(self.tokens[token])
        return encoded

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.reverse[token_id] for token_id in token_ids)


class EquivalentSegmentationCodec:
    def encode(self, text: str) -> list[int]:
        return [1 if token.startswith("qwasar_") else 2 for token in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join("canonical" for _ in token_ids)


def case(appended_tokens: int) -> WorkloadCase:
    return WorkloadCase(
        case_id="persistent-turn",
        fixture_id="coding",
        mode="persistent_turn",
        context_bucket="1k",
        target_context_tokens=appended_tokens,
        appended_tokens=appended_tokens,
        warmups=0,
        repetitions=1,
        session_id="agent",
        turn_index=0,
    )


def test_prompt_builder_produces_exact_deterministic_token_count(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixtures.jsonl"
    fixture_path.write_text(
        '{"fixture_id":"coding","category":"coding","input":"repair the function","filler_seed":42}\n',
        encoding="utf-8",
    )
    codec = WhitespaceCodec()
    builder = FixturePromptBuilder.from_jsonl(fixture_path, codec)

    first = builder.build_input(case(64))
    second = builder.build_input(case(64))
    content = first[0]["content"]

    assert first == second
    assert isinstance(content, str)
    assert len(codec.encode(content)) == 64
    assert content.endswith("repair the function")


def test_prompt_builder_accepts_equivalent_tokenizer_segmentation(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixtures.jsonl"
    fixture_path.write_text(
        '{"fixture_id":"coding","category":"coding","input":"repair function","filler_seed":42}\n',
        encoding="utf-8",
    )
    codec = EquivalentSegmentationCodec()
    builder = FixturePromptBuilder.from_jsonl(fixture_path, codec)

    content = builder.build_input(case(64))[0]["content"]

    assert len(codec.encode(content)) == 64
