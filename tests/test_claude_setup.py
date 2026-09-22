import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def installer():
    source = ROOT / "scripts/configure_claude.py"
    if not source.exists():
        pytest.fail("Claude Code setup installer is missing")
    spec = importlib.util.spec_from_file_location("configure_claude", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snippet_points_at_local_anthropic_messages():
    env = (ROOT / "integrations/claude/env").read_text()
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8800" in env
    assert "ANTHROPIC_MODEL=qwasar-qwen38-27b" in env
    assert "ANTHROPIC_API_KEY=qwasar-local" in env
    assert "ANTHROPIC_AUTH_TOKEN=qwasar-local" in env


def test_writes_env_without_touching_settings_and_is_idempotent(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text('{"model":"claude-opus"}\n')
    target = tmp_path / "qwasar.env"
    backup = installer().configure(target)
    assert backup is None
    text = target.read_text()
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8800" in text
    assert "ANTHROPIC_MODEL=qwasar-qwen38-27b" in text
    assert settings.read_text() == '{"model":"claude-opus"}\n'
    assert installer().configure(target) is None


def test_owned_env_is_updated_and_foreign_env_is_refused(tmp_path):
    target = tmp_path / "qwasar.env"
    target.write_text("ANTHROPIC_BASE_URL=http://127.0.0.1:8800\nANTHROPIC_MODEL=old\n")
    backup = installer().configure(target)
    assert backup.read_text() == "ANTHROPIC_BASE_URL=http://127.0.0.1:8800\nANTHROPIC_MODEL=old\n"
    assert "ANTHROPIC_MODEL=qwasar-qwen38-27b" in target.read_text()
    foreign = tmp_path / "other.env"
    foreign.write_text("ANTHROPIC_BASE_URL=https://api.anthropic.com\n")
    with pytest.raises(ValueError):
        installer().configure(foreign)
    assert foreign.read_text() == "ANTHROPIC_BASE_URL=https://api.anthropic.com\n"
