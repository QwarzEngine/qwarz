import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def installer():
    source = ROOT / "scripts/configure_omp.py"
    if not source.exists():
        pytest.fail("OMP setup installer is missing")
    spec = importlib.util.spec_from_file_location("configure_omp", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snippet_advertises_native_image_and_qwen_thinking():
    snippet = yaml.safe_load((ROOT / "integrations/omp/models.yml").read_text())
    provider = snippet["providers"]["qwasar"]
    model = provider["models"][0]
    assert provider["baseUrl"] == "http://127.0.0.1:8800/v1"
    assert provider["api"] == "openai-completions"
    assert provider["compat"]["thinkingFormat"] == "qwen-chat-template"
    assert provider["compat"]["reasoningEffortMap"]["off"] == "off"
    assert provider["compat"]["reasoningEffortMap"]["high"] == "xhigh"
    assert model["id"] == "qwasar-qwen38-27b"
    assert model["input"] == ["text", "image"]
    assert model["contextWindow"] == 262144
    assert model["maxTokens"] == 32768


def test_preserves_other_providers_and_makes_exact_backup(tmp_path):
    target = tmp_path / "models.yml"
    original = b"providers:\n  existing:\n    apiKey: untouched\nkeep: true\n"
    target.write_bytes(original)
    backup = installer().configure(target)
    assert backup.read_bytes() == original
    output = yaml.safe_load(target.read_text())
    assert output["keep"] is True
    assert output["providers"]["existing"] == {"apiKey": "untouched"}
    qwasar = output["providers"]["qwasar"]
    assert qwasar["baseUrl"] == "http://127.0.0.1:8800/v1"
    assert qwasar["models"][0]["input"] == ["text", "image"]
    assert installer().configure(target) is None


def test_owned_qwasar_provider_is_updated_in_place(tmp_path):
    target = tmp_path / "models.yml"
    target.write_text(
        "providers:\n"
        "  qwasar:\n"
        "    baseUrl: http://localhost:8800/v1\n"
        "    api: openai-completions\n"
        "    models:\n"
        "      - id: qwasar-qwen38-27b\n"
        "        input: [text]\n"
    )
    backup = installer().configure(target)
    assert backup is not None
    provider = yaml.safe_load(target.read_text())["providers"]["qwasar"]
    assert provider["baseUrl"] == "http://127.0.0.1:8800/v1"
    assert provider["models"][0]["input"] == ["text", "image"]
    assert provider["compat"]["supportsReasoningEffort"] is True


def test_conflicting_or_invalid_config_is_never_overwritten(tmp_path):
    target = tmp_path / "models.yml"
    for content in (
        ":\n  - bad",
        "providers: []\n",
        "providers:\n  qwasar:\n    baseUrl: different\n",
    ):
        target.write_text(content)
        with pytest.raises((ValueError, yaml.YAMLError)):
            installer().configure(target)
        assert target.read_text() == content


def test_new_config_registers_local_qwasar_without_default_role(tmp_path):
    target = tmp_path / "new" / "models.yml"
    assert installer().configure(target) is None
    output = yaml.safe_load(target.read_text())
    assert "modelRoles" not in output
    provider = output["providers"]["qwasar"]
    assert provider["apiKey"] == "qwasar-local"
    assert provider["models"][0]["name"] == "Qwarz Qwen3.8 27B"
    text = target.read_text()
    assert '"off": "off"' in text or "'off': 'off'" in text
    assert "false: false" not in text


def test_background_roles_leave_qwasar_and_keep_default(tmp_path):
    target = tmp_path / "config.yml"
    original = b"setupVersion: 2\nmodelRoles:\n  default: qwasar/qwasar-qwen38-27b:high\ncomposer:\n  shape: band\n"
    target.write_bytes(original)
    backup = installer().configure_roles(target)
    assert backup.read_bytes() == original
    output = yaml.safe_load(target.read_text())
    assert output["modelRoles"]["default"] == "qwasar/qwasar-qwen38-27b:high"
    assert output["modelRoles"]["tiny"] == "openai-codex/gpt-5.5:off"
    assert output["modelRoles"]["smol"] == "openai-codex/gpt-5.5:off"
    assert output["composer"] == {"shape": "band"}
    assert installer().configure_roles(target) is None


def test_existing_non_qwasar_background_roles_are_kept(tmp_path):
    target = tmp_path / "config.yml"
    target.write_text("modelRoles:\n  tiny: gemma-local/gemma-4-12b\n  smol: openai-codex/gpt-5.5:off\n")
    assert installer().configure_roles(target) is None
    roles = yaml.safe_load(target.read_text())["modelRoles"]
    assert roles["tiny"] == "gemma-local/gemma-4-12b"
