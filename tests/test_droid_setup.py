import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def installer():
    source = ROOT / "scripts/configure_droid.py"
    if not source.exists():
        pytest.fail("Droid setup installer is missing")
    spec = importlib.util.spec_from_file_location("configure_droid", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserves_other_custom_models_and_makes_exact_backup(tmp_path):
    target = tmp_path / "settings.json"
    original = b'{"customModels":[{"model":"lfm2.5-8b-a1b","displayName":"LFM"}],"logoAnimation":"off"}'
    target.write_bytes(original)
    backup = installer().configure(target)
    assert backup.read_bytes() == original
    output = json.loads(target.read_text())
    assert output["logoAnimation"] == "off"
    qwasar = output["customModels"][0]
    assert qwasar["model"] == "qwasar-qwen38-27b"
    assert qwasar["baseUrl"] == "http://127.0.0.1:8800/v1"
    assert qwasar["provider"] == "generic-chat-completion-api"
    assert qwasar["maxOutputTokens"] == 32768
    other = output["customModels"][1]
    assert other["model"] == "lfm2.5-8b-a1b"
    assert other["displayName"] == "LFM"
    assert other["index"] == 1
    assert installer().configure(target) is None


def test_owned_qwasar_entry_is_updated_in_place(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"customModels": [{
        "model": "qwasar-qwen38-27b",
        "baseUrl": "http://127.0.0.1:8800/v1",
        "provider": "generic-chat-completion-api",
        "maxOutputTokens": 8192,
    }]}))
    backup = installer().configure(target)
    assert backup is not None
    entry = json.loads(target.read_text())["customModels"][0]
    assert entry["maxOutputTokens"] == 32768
    assert entry["displayName"] == "Qwasar Qwen3.8 27B · 5bpw · MTP"


def test_conflicting_or_invalid_config_is_never_overwritten(tmp_path):
    target = tmp_path / "settings.json"
    for content in ('not-json', '{"customModels":"bad"}', '{"customModels":[{"model":"qwasar-qwen38-27b","baseUrl":"different"}]}'):
        target.write_text(content)
        with pytest.raises(ValueError):
            installer().configure(target)
        assert target.read_text() == content


def test_new_config_registers_local_qwasar_model(tmp_path):
    target = tmp_path / "new" / "settings.json"
    assert installer().configure(target) is None
    entry = json.loads(target.read_text())["customModels"][0]
    assert entry["model"] == "qwasar-qwen38-27b"
    assert entry["apiKey"] == "qwasar-local"
    assert entry["noImageSupport"] is False


def test_legacy_local_qwen_on_8004_is_replaced(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"customModels": [
        {
            "model": "qwen3.8-27b",
            "id": "custom:Qwen3.8-27B-[EXL3-5090]-0",
            "index": 0,
            "baseUrl": "http://127.0.0.1:8004/v1",
            "provider": "generic-chat-completion-api",
        },
        {
            "model": "lfm2.5-8b-a1b",
            "id": "custom:LFM2.5-8B-A1B-[SGLang-3090-Ti]-0",
            "index": 1,
            "baseUrl": "http://127.0.0.1:30001/v1",
            "provider": "generic-chat-completion-api",
        },
    ]}))
    installer().configure(target)
    models = json.loads(target.read_text())["customModels"]
    assert models[0]["model"] == "qwasar-qwen38-27b"
    assert models[1]["model"] == "lfm2.5-8b-a1b"
    assert all(entry.get("model") != "qwen3.8-27b" for entry in models)
