import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def installer():
    source = ROOT / "scripts/configure_qwen_code.py"
    if not source.exists():
        pytest.fail("Qwen Code setup installer is missing")
    spec = importlib.util.spec_from_file_location("configure_qwen_code", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snippet_advertises_native_image_and_qwen_thinking():
    snippet = json.loads((ROOT / "integrations/qwen-code/settings.json").read_text())
    model = snippet["modelProviders"]["openai"][0]
    assert model["id"] == "qwasar-qwen38-27b"
    assert model["baseUrl"] == "http://127.0.0.1:8800/v1"
    assert model["envKey"] == "QWASAR_API_KEY"
    assert model["generationConfig"]["contextWindowSize"] == 262144
    assert model["generationConfig"]["modalities"]["image"] is True
    assert model["capabilities"]["reasoning"]["profile"] == "qwen-chat-template"


def test_preserves_other_openai_models_and_makes_exact_backup(tmp_path):
    target = tmp_path / "settings.json"
    original = b'{"env":{"VLLM_API_KEY":"API"},"modelProviders":{"openai":[{"id":"spark-x2.5-4b","baseUrl":"http://127.0.0.1:8002/v1"}]},"model":{"name":"spark-x2.5-4b"}}'
    target.write_bytes(original)
    backup = installer().configure(target)
    assert backup.read_bytes() == original
    output = json.loads(target.read_text())
    assert output["model"] == {"name": "spark-x2.5-4b"}
    assert output["env"]["VLLM_API_KEY"] == "API"
    assert output["env"]["QWASAR_API_KEY"] == "qwasar-local"
    assert output["modelProviders"]["openai"][0]["id"] == "spark-x2.5-4b"
    qwasar = output["modelProviders"]["openai"][1]
    assert qwasar["id"] == "qwasar-qwen38-27b"
    assert qwasar["baseUrl"] == "http://127.0.0.1:8800/v1"
    assert qwasar["generationConfig"]["modalities"]["image"] is True
    assert output["fastModel"] == "spark-x2.5-4b"
    assert installer().configure(target) is None


def test_owned_qwasar_entry_is_updated_in_place(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"modelProviders": {"openai": [{
        "id": "qwasar-qwen38-27b",
        "baseUrl": "http://localhost:8800/v1",
        "generationConfig": {"contextWindowSize": 8192},
    }]}}))
    backup = installer().configure(target)
    assert backup is not None
    model = json.loads(target.read_text())["modelProviders"]["openai"][0]
    assert model["baseUrl"] == "http://127.0.0.1:8800/v1"
    assert model["generationConfig"]["contextWindowSize"] == 262144
    assert model["capabilities"]["reasoning"]["profile"] == "qwen-chat-template"


def test_conflicting_or_invalid_config_is_never_overwritten(tmp_path):
    target = tmp_path / "settings.json"
    for content in (
        "not-json",
        '{"modelProviders":[]}',
        '{"modelProviders":{"openai":[{"id":"qwasar-qwen38-27b","baseUrl":"different"}]}}',
    ):
        target.write_text(content)
        with pytest.raises((ValueError, json.JSONDecodeError)):
            installer().configure(target)
        assert target.read_text() == content


def test_new_config_registers_local_qwasar_without_changing_default_model(tmp_path):
    target = tmp_path / "new" / "settings.json"
    assert installer().configure(target) is None
    output = json.loads(target.read_text())
    assert "model" not in output
    assert "fastModel" not in output
    model = output["modelProviders"]["openai"][0]
    assert model["name"] == "Qwarz Qwen3.8 27B"
    assert output["env"]["QWASAR_API_KEY"] == "qwasar-local"


def test_pins_empty_or_qwasar_fast_model_to_local_when_present(tmp_path):
    module = installer()
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({
        "env": {"QWASAR_API_KEY": "qwasar-local"},
        "modelProviders": {"openai": [
            json.loads(json.dumps(module.MODEL)),
            {"id": "qwen3.8-27b", "baseUrl": "http://127.0.0.1:8004/v1"},
            {"id": "spark-x2.5-4b", "baseUrl": "http://127.0.0.1:8002/v1"},
        ]},
        "fastModel": "qwasar-qwen38-27b",
    }))
    assert module.configure(target) is not None
    output = json.loads(target.read_text())
    assert output["fastModel"] == "spark-x2.5-4b"
    assert output["modelProviders"]["openai"][1]["id"] == "qwen3.8-27b"
    assert module.configure(target) is None


def test_preserves_user_fast_model_that_is_not_qwasar(tmp_path):
    module = installer()
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({
        "modelProviders": {"openai": [
            {"id": "qwen3.8-27b", "baseUrl": "http://127.0.0.1:8004/v1"},
        ]},
        "fastModel": "qwen3.8-27b",
    }))
    module.configure(target)
    assert json.loads(target.read_text())["fastModel"] == "qwen3.8-27b"
