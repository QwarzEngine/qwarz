import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def installer():
    source = ROOT / "scripts/configure_codex.py"
    if not source.exists():
        pytest.fail("Codex setup installer is missing")
    spec = importlib.util.spec_from_file_location("configure_codex", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_advertises_native_image_input():
    catalog = json.loads((ROOT / "integrations/codex/catalog.json").read_text())
    model = catalog["models"][0]
    assert model["slug"] == "qwasar-qwen38-27b"
    assert model["input_modalities"] == ["text", "image"]
    assert model["supports_image_detail_original"] is False
    assert model["context_window"] == 262144


def test_preserves_other_catalog_models_and_makes_exact_backup(tmp_path):
    home = tmp_path / "codex"
    catalog = home / "qwasar-catalog.json"
    original = b'{"models":[{"slug":"other","display_name":"Other"}],"keep":true}'
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(original)
    backups = installer().configure(home)
    assert backups["catalog"].read_bytes() == original
    output = json.loads(catalog.read_text())
    assert output["keep"] is True
    assert output["models"][0]["slug"] == "qwasar-qwen38-27b"
    assert output["models"][0]["input_modalities"] == ["text", "image"]
    assert output["models"][1]["slug"] == "other"
    assert installer().configure(home)["catalog"] is None


def test_owned_catalog_entry_gains_image_modalities(tmp_path):
    home = tmp_path / "codex"
    catalog = home / "qwasar-catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"models": [{
        "slug": "qwasar-qwen38-27b",
        "shell_type": "unified_exec",
        "context_window": 262144,
        "input_modalities": ["text"],
    }]}))
    backups = installer().configure(home)
    assert backups["catalog"] is not None
    entry = json.loads(catalog.read_text())["models"][0]
    assert entry["input_modalities"] == ["text", "image"]
    assert entry["display_name"] == "Qwarz Qwen3.8-27B"


def test_conflicting_or_invalid_catalog_is_never_overwritten(tmp_path):
    home = tmp_path / "codex"
    catalog = home / "qwasar-catalog.json"
    catalog.parent.mkdir(parents=True)
    for content in ('not-json', '{"models":"bad"}', '{"models":[{"slug":"qwasar-qwen38-27b","shell_type":"other","context_window":262144}]}'):
        catalog.write_text(content)
        with pytest.raises(ValueError):
            installer().configure(home)
        assert catalog.read_text() == content


def test_new_home_registers_provider_profile_and_catalog(tmp_path):
    home = tmp_path / "new" / "codex"
    backups = installer().configure(home)
    assert backups["catalog"] is None
    assert backups["profile"] is not None or (home / "qwasar.config.toml").exists()
    catalog = json.loads((home / "qwasar-catalog.json").read_text())
    assert catalog["models"][0]["input_modalities"] == ["text", "image"]
    profile = (home / "qwasar.config.toml").read_text()
    assert 'model_provider = "qwasar"' in profile
    assert f'model_catalog_json = "{home / "qwasar-catalog.json"}"' in profile
    config = (home / "config.toml").read_text()
    assert "[model_providers.qwasar]" in config
    assert "http://127.0.0.1:8800/v1" in config
    assert 'wire_api = "responses"' in config


def test_existing_profile_keeps_projects_and_only_sets_catalog_path(tmp_path):
    home = tmp_path / "codex"
    home.mkdir(parents=True)
    (home / "qwasar.config.toml").write_text(
        'model_provider = "qwasar"\nmodel = "qwasar-qwen38-27b"\n\n[projects."/tmp"]\ntrust_level = "trusted"\n'
    )
    (home / "config.toml").write_text(
        '[model_providers.qwasar]\nname = "Qwarz"\nbase_url = "http://127.0.0.1:8800/v1"\n'
    )
    backups = installer().configure(home)
    profile = (home / "qwasar.config.toml").read_text()
    assert '[projects."/tmp"]' in profile
    assert f'model_catalog_json = "{home / "qwasar-catalog.json"}"' in profile
    assert backups["provider"] is None
