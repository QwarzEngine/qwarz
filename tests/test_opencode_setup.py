import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def installer():
    source = ROOT / "scripts/configure_opencode.py"
    if not source.exists():
        pytest.fail("OpenCode setup installer is missing")
    spec = importlib.util.spec_from_file_location("configure_opencode", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snippet_advertises_native_image_and_reasoning():
    snippet = json.loads((ROOT / "integrations/opencode/opencode.json").read_text())
    provider = snippet["provider"]["qwasar"]
    model = provider["models"]["qwasar-qwen38-27b"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8800/v1"
    assert model["modalities"]["input"] == ["text", "image"]
    assert model["limit"] == {"context": 262144, "output": 32768}
    assert model["reasoning"] is True
    assert model["interleaved"]["field"] == "reasoning_content"


def test_preserves_other_providers_and_makes_exact_backup(tmp_path):
    target = tmp_path / "opencode.json"
    original = b'{"$schema":"https://opencode.ai/config.json","provider":{"qwen-local":{"name":"Qwen"}},"model":"qwen-local/qwen3.6-27b"}'
    target.write_bytes(original)
    backup = installer().configure(target)
    assert backup.read_bytes() == original
    output = json.loads(target.read_text())
    assert output["model"] == "qwen-local/qwen3.6-27b"
    assert output["provider"]["qwen-local"] == {"name": "Qwen"}
    qwasar = output["provider"]["qwasar"]
    assert qwasar["options"]["baseURL"] == "http://127.0.0.1:8800/v1"
    assert qwasar["models"]["qwasar-qwen38-27b"]["modalities"]["input"] == ["text", "image"]
    assert installer().configure(target) is None


def test_owned_qwasar_provider_is_updated_in_place(tmp_path):
    target = tmp_path / "opencode.json"
    target.write_text(json.dumps({"provider": {"qwasar": {
        "npm": "@ai-sdk/openai-compatible",
        "options": {"baseURL": "http://localhost:8800/v1", "apiKey": "old"},
        "models": {"qwasar-qwen38-27b": {"name": "old", "modalities": {"input": ["text"]}}},
    }}}))
    backup = installer().configure(target)
    assert backup is not None
    provider = json.loads(target.read_text())["provider"]["qwasar"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8800/v1"
    assert provider["models"]["qwasar-qwen38-27b"]["modalities"]["input"] == ["text", "image"]
    assert provider["models"]["qwasar-qwen38-27b"]["limit"]["output"] == 32768


def test_conflicting_or_invalid_config_is_never_overwritten(tmp_path):
    target = tmp_path / "opencode.json"
    for content in ('not-json', '{"provider":[]}', '{"provider":{"qwasar":{"options":{"baseURL":"different"}}}}'):
        target.write_text(content)
        with pytest.raises(ValueError):
            installer().configure(target)
        assert target.read_text() == content


def test_new_config_registers_local_qwasar_without_changing_default_model(tmp_path):
    target = tmp_path / "new" / "opencode.json"
    assert installer().configure(target) is None
    output = json.loads(target.read_text())
    assert "model" not in output
    provider = output["provider"]["qwasar"]
    assert provider["name"] == "Qwarz"
    assert provider["options"]["apiKey"] == "qwasar-local"
    assert "qwasar-qwen38-27b" in provider["models"]
