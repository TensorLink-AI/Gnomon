"""Tests for the stdlib .env loader."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.envfile import load_env_file, parse_env_text


def test_parse_env_text_variants():
    text = (
        "# comment\n"
        "OPENROUTER_API_KEY=sk-or-abc123\n"
        "export QUOTED='hello world'\n"
        'DOUBLE="a=b#c"\n'
        "TRAILING=value # inline comment\n"
        "not a line\n"
        "=nokey\n"
        "BAD KEY=x\n"
    )
    values = parse_env_text(text)
    assert values["OPENROUTER_API_KEY"] == "sk-or-abc123"
    assert values["QUOTED"] == "hello world"
    assert values["DOUBLE"] == "a=b#c"
    assert values["TRAILING"] == "value"
    assert "BAD KEY" not in values and "" not in values


def test_load_env_file_never_overrides_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "TEST_ENVFILE_A=from_file\nTEST_ENVFILE_B=from_file\n"
    )
    monkeypatch.setenv("TEST_ENVFILE_A", "from_shell")
    monkeypatch.delenv("TEST_ENVFILE_B", raising=False)
    applied = load_env_file(tmp_path)
    assert applied == {"TEST_ENVFILE_B": "from_file"}
    assert os.environ["TEST_ENVFILE_A"] == "from_shell"
    assert os.environ["TEST_ENVFILE_B"] == "from_file"
    monkeypatch.delenv("TEST_ENVFILE_B", raising=False)


def test_load_env_file_first_directory_wins(tmp_path, monkeypatch):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir(); second.mkdir()
    (first / ".env").write_text("TEST_ENVFILE_C=first\n")
    (second / ".env").write_text("TEST_ENVFILE_C=second\n")
    monkeypatch.delenv("TEST_ENVFILE_C", raising=False)
    load_env_file(first, second)
    assert os.environ["TEST_ENVFILE_C"] == "first"
    monkeypatch.delenv("TEST_ENVFILE_C", raising=False)


def test_load_env_file_missing_is_noop(tmp_path):
    assert load_env_file(tmp_path) == {}
