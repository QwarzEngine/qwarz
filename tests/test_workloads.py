from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwasar_bench.workloads import load_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "benchmarks/manifests/qwen38-27b-rtx5090-v1.json"


def minimal_manifest() -> dict[str, object]:
    return {
        "version": 1,
        "model": {
            "architecture": "Qwen3.8ForCausalLM",
            "repository": "Qwen/Qwen3.8-27B",
            "quantization": "EXL3",
            "target_bpw": 5.0,
            "native_context_tokens": 262144,
            "artifact_path_env": "QWASAR_MODEL_PATH",
            "artifact_sha256": None,
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 256,
            "seed": 0,
        },
        "cases": [
            {
                "case_id": "fresh-1k",
                "fixture_id": "coding-smoke",
                "mode": "fresh_prefill",
                "context_bucket": "1k",
                "target_context_tokens": 1024,
                "appended_tokens": 1024,
                "warmups": 1,
                "repetitions": 3,
                "session_id": None,
                "turn_index": None,
            }
        ],
    }


def write_manifest(tmp_path: Path, values: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_manifest_freezes_required_context_buckets_and_modes() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    assert manifest.version == 1
    assert {case.context_bucket for case in manifest.cases} == {
        "1k",
        "32k",
        "128k",
        "256k",
    }
    assert any(case.mode == "persistent_turn" for case in manifest.cases)
    assert any(case.mode == "fresh_prefill" for case in manifest.cases)
    assert len(manifest.sha256) == 64


def test_manifest_rejects_unknown_generation_keys(tmp_path: Path) -> None:
    values = minimal_manifest()
    values["generation"] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 256,
        "seed": 0,
        "mystery": 1,
    }

    with pytest.raises(ValueError, match="generation.*mystery"):
        load_manifest(write_manifest(tmp_path, values))


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    values = minimal_manifest()
    cases = values["cases"]
    assert isinstance(cases, list)
    cases.append(dict(cases[0]))

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_manifest(write_manifest(tmp_path, values))


def test_persistent_turns_must_be_contiguous(tmp_path: Path) -> None:
    values = minimal_manifest()
    cases = values["cases"]
    assert isinstance(cases, list)
    cases[0] = {
        **cases[0],
        "mode": "persistent_turn",
        "session_id": "agent-1k",
        "turn_index": 1,
    }

    with pytest.raises(ValueError, match="turn_index 0"):
        load_manifest(write_manifest(tmp_path, values))


def test_profile_manifest_replaces_only_the_model_pin(tmp_path: Path) -> None:
    base_path = write_manifest(tmp_path, minimal_manifest())
    profile_path = tmp_path / "bringup.json"
    profile_path.write_text(
        json.dumps(
            {
                "extends": base_path.name,
                "model": {
                    "architecture": "Qwen3.8ForCausalLM",
                    "repository": "Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw",
                    "quantization": "EXL3",
                    "target_bpw": 3.5,
                    "native_context_tokens": 262144,
                    "artifact_path_env": "QWASAR_MODEL_PATH",
                    "artifact_sha256": "4" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    base = load_manifest(base_path)
    profile = load_manifest(profile_path)

    assert profile.model.target_bpw == 3.5
    assert profile.model.artifact_sha256 == "4" * 64
    assert profile.cases == base.cases
    assert profile.sha256 != base.sha256
