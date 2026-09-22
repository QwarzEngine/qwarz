import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def installer():
    source = ROOT / "scripts/configure_hermes.py"
    if not source.exists():
        pytest.fail("Hermes setup installer is missing")
    spec = importlib.util.spec_from_file_location("configure_hermes", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snippet_advertises_native_vision_and_reasoning():
    snippet = yaml.safe_load((ROOT / "integrations/hermes/provider.yaml").read_text())
    provider = snippet["qwasar"]
    model = provider["models"]["qwasar-qwen38-27b"]
    assert provider["api"] == "http://127.0.0.1:8800/v1"
    assert provider["transport"] == "chat_completions"
    assert provider["context_length"] == 262144
    assert model["supports_vision"] is True
    assert model["supports_reasoning"] is True


def test_preserves_other_providers_and_default_model(tmp_path):
    target = tmp_path / "config.yaml"
    original = (
        "model:\n"
        "  default: qwen3.8-27b\n"
        "  provider: qwen38-local\n"
        "  base_url: http://127.0.0.1:8004/v1\n"
        "providers:\n"
        "  qwen38-local:\n"
        "    api: http://127.0.0.1:8004/v1\n"
        "  lfm-local:\n"
        "    api: http://127.0.0.1:30001/v1\n"
    ).encode()
    target.write_bytes(original)
    backup = installer().configure(target)
    assert backup.read_bytes() == original
    output = yaml.safe_load(target.read_text())
    assert output["model"] == {
        "default": "qwen3.8-27b",
        "provider": "qwen38-local",
        "base_url": "http://127.0.0.1:8004/v1",
    }
    assert output["providers"]["lfm-local"] == {"api": "http://127.0.0.1:30001/v1"}
    qwasar = output["providers"]["qwasar"]
    assert qwasar["api"] == "http://127.0.0.1:8800/v1"
    assert qwasar["models"]["qwasar-qwen38-27b"]["supports_vision"] is True
    assert output["auxiliary"]["title_generation"] == {
        "provider": "qwen38-local",
        "model": "qwen3.8-27b",
    }
    assert installer().configure(target) is None


def test_pins_auto_or_qwasar_title_generation_to_local_when_present(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text(
        "providers:\n"
        "  qwen38-local:\n"
        "    api: http://127.0.0.1:8004/v1\n"
        "  qwasar:\n"
        "    api: http://127.0.0.1:8800/v1\n"
        "    transport: chat_completions\n"
        "    name: Qwarz\n"
        "    default_model: qwasar-qwen38-27b\n"
        "    discover_models: false\n"
        "    context_length: 262144\n"
        "    models:\n"
        "      qwasar-qwen38-27b:\n"
        "        context_length: 262144\n"
        "        supports_vision: true\n"
        "        supports_reasoning: true\n"
        "auxiliary:\n"
        "  title_generation:\n"
        "    provider: auto\n"
        "    model: ''\n"
    )
    backup = installer().configure(target)
    assert backup is not None
    output = yaml.safe_load(target.read_text())
    assert output["auxiliary"]["title_generation"]["provider"] == "qwen38-local"
    assert output["auxiliary"]["title_generation"]["model"] == "qwen3.8-27b"
    assert installer().configure(target) is None


def test_leaves_explicit_title_provider_alone(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text(
        "providers:\n"
        "  qwen38-local:\n"
        "    api: http://127.0.0.1:8004/v1\n"
        "auxiliary:\n"
        "  title_generation:\n"
        "    provider: openai-codex\n"
        "    model: gpt-5.5\n"
    )
    installer().configure(target)
    title = yaml.safe_load(target.read_text())["auxiliary"]["title_generation"]
    assert title == {"provider": "openai-codex", "model": "gpt-5.5"}


def test_owned_qwasar_provider_is_updated_in_place(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text(
        "providers:\n"
        "  qwasar:\n"
        "    api: http://localhost:8800/v1\n"
        "    transport: chat_completions\n"
        "    models:\n"
        "      qwasar-qwen38-27b:\n"
        "        supports_vision: false\n"
    )
    backup = installer().configure(target)
    assert backup is not None
    provider = yaml.safe_load(target.read_text())["providers"]["qwasar"]
    assert provider["api"] == "http://127.0.0.1:8800/v1"
    assert provider["models"]["qwasar-qwen38-27b"]["supports_vision"] is True
    assert provider["models"]["qwasar-qwen38-27b"]["supports_reasoning"] is True


def test_conflicting_or_invalid_config_is_never_overwritten(tmp_path):
    target = tmp_path / "config.yaml"
    for content in (
        ":\n  - bad",
        "providers: []\n",
        "providers:\n  qwasar:\n    api: different\n",
    ):
        target.write_text(content)
        with pytest.raises((ValueError, yaml.YAMLError)):
            installer().configure(target)
        assert target.read_text() == content


def test_new_config_registers_local_qwasar_without_changing_default(tmp_path):
    target = tmp_path / "new" / "config.yaml"
    assert installer().configure(target) is None
    output = yaml.safe_load(target.read_text())
    assert "model" not in output
    provider = output["providers"]["qwasar"]
    assert provider["name"] == "Qwarz"
    assert provider["default_model"] == "qwasar-qwen38-27b"
