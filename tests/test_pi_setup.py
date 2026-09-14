import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def installer():
    source = ROOT / "scripts/configure_pi.py"
    if not source.exists():
        pytest.fail("Pi setup installer is missing")
    spec = importlib.util.spec_from_file_location("configure_pi", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserves_other_providers_and_makes_exact_backup(tmp_path):
    target = tmp_path / "models.json"
    original = b'{"providers":{"existing":{"apiKey":"untouched","models":[]}},"custom":true}'
    target.write_bytes(original)
    backup = installer().configure(target)
    assert backup.read_bytes() == original
    output = json.loads(target.read_text())
    assert output["providers"]["existing"] == {"apiKey": "untouched", "models": []}
    assert output["custom"] is True
    assert output["providers"]["qwasar"]["api"] == "openai-completions"
    assert output["providers"]["qwasar"]["compat"]["thinkingFormat"] == "qwen-chat-template"
    assert output["providers"]["qwasar"]["compat"]["supportsReasoningEffort"] is True
    assert output["providers"]["qwasar"]["models"][0]["thinkingLevelMap"]["xhigh"] == "xhigh"
    assert installer().configure(target) is None


def test_owned_qwasar_provider_is_updated_in_place(tmp_path):
    target = tmp_path / "models.json"
    target.write_text(json.dumps({"providers": {"qwasar": {
        "baseUrl": "http://127.0.0.1:8800/v1", "api": "openai-completions",
        "compat": {"supportsReasoningEffort": False, "thinkingFormat": "qwen-chat-template"},
        "models": [{"id": "qwasar-qwen38-27b"}]}}}))
    backup = installer().configure(target)
    assert backup is not None
    provider = json.loads(target.read_text())["providers"]["qwasar"]
    assert provider["compat"]["supportsReasoningEffort"] is True
    assert provider["models"][0]["thinkingLevelMap"]["high"] == "xhigh"


def test_conflicting_or_invalid_config_is_never_overwritten(tmp_path):
    target = tmp_path / "models.json"
    for content in ('not-json', '{"providers":[]}', '{"providers":{"qwasar":{"baseUrl":"different"}}}'):
        target.write_text(content)
        with pytest.raises(ValueError):
            installer().configure(target)
        assert target.read_text() == content


def test_new_config_uses_local_provider_and_native_window(tmp_path):
    target = tmp_path / "new" / "models.json"
    assert installer().configure(target) is None
    provider = json.loads(target.read_text())["providers"]["qwasar"]
    assert provider["baseUrl"] == "http://127.0.0.1:8800/v1"
    assert provider["models"][0]["contextWindow"] == 262144
    assert provider["models"][0]["maxTokens"] == 32768
