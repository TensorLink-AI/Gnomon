"""The ``aionbench`` command line, end to end, without a provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aionbench.cli import _read_roster, _thresholds, main

ROSTER = Path(__file__).resolve().parents[1] / "benchmarks" / "configs" / "models.yaml"


def test_suites_command_lists_all_three_benchmarks(capsys):
    assert main(["suites"]) == 0
    payload = json.loads(capsys.readouterr().out)
    names = {item["name"] for item in payload["suites"]}
    assert names == {"timeseriesexam", "tsaia", "tsqa"}
    for item in payload["suites"]:
        assert item["citation"]


def test_shipped_roster_parses_and_de_duplicates():
    slugs = _read_roster(ROSTER)
    assert len(slugs) == len(set(slugs))
    assert all("/" in slug for slug in slugs)
    assert "anthropic/claude-opus-4.1" in slugs


def test_plain_text_roster_ignores_comments(tmp_path):
    path = tmp_path / "roster.txt"
    path.write_text("# closed\nopenai/gpt-5\n\ndeepseek/deepseek-r1\n", encoding="utf-8")
    assert _read_roster(path) == ["openai/gpt-5", "deepseek/deepseek-r1"]


def test_threshold_overrides_parse_and_reject_junk():
    assert _thresholds(["mape=0.2", "f1=0.6"]) == {"mape": 0.2, "f1": 0.6}
    from aionbench.contracts import BenchError

    with pytest.raises(BenchError, match="not a number"):
        _thresholds(["mape=soon"])
    with pytest.raises(BenchError, match="METRIC=VALUE"):
        _thresholds(["mape"])


def test_dry_run_produces_a_graded_run_directory(tmp_path, capsys):
    output = tmp_path / "run"
    code = main([
        "run", "--suite", "tsqa", "--models", "echo/echo",
        "--conditions", "direct", "--limit", "4", "--dry-run",
        "--output", str(output), "--quiet",
    ])
    assert code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["rows"] == 4
    assert payload["spend_usd"] == 0.0

    assert (output / "rows.jsonl").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "tasks.json").is_file()

    manifest = json.loads((output / "manifest.json").read_text())
    # The manifest must carry every knob that could move a number.
    assert manifest["config"]["temperature"] == 0.0
    assert manifest["config"]["seed"] == 20260801
    assert manifest["suite"] == "tsqa"
    assert manifest["citation"]


def test_report_and_export_round_trip_a_dry_run(tmp_path, capsys):
    output = tmp_path / "run"
    main([
        "run", "--suite", "tsqa", "--models", "echo/echo",
        "--conditions", "direct,aion", "--limit", "3", "--dry-run",
        "--output", str(output), "--quiet",
    ])
    capsys.readouterr()

    assert main(["report", "--run", str(output), "--format", "json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary["conditions"]) == {"direct", "aion"}
    assert summary["uplift"]

    exported = tmp_path / "compare"
    assert main([
        "export", "--run", str(output), "--model", "echo/echo", "--output", str(exported),
    ]) == 0
    paths = json.loads(capsys.readouterr().out)
    assert Path(paths["baseline"]).is_file() and Path(paths["treatment"]).is_file()
    assert "aion eval compare" in paths["next"]

    from aion.agent_eval import compare_runs

    comparison = compare_runs(paths["baseline"], paths["treatment"])
    assert comparison["status"] == "complete"


def test_run_without_a_key_fails_with_a_typed_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code = main([
        "run", "--suite", "tsqa", "--models", "openai/gpt-5",
        "--limit", "1", "--output", str(tmp_path / "run"), "--quiet",
    ])
    assert code == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "OPENROUTER_KEY_MISSING"
    assert "--dry-run" in error["message"]


def test_unsupported_condition_is_reported_as_a_typed_error(tmp_path, capsys):
    code = main([
        "run", "--suite", "tsqa", "--models", "echo/echo", "--conditions", "code",
        "--limit", "1", "--dry-run", "--output", str(tmp_path / "run"), "--quiet",
    ])
    assert code == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "UNSUPPORTED_CONDITION"
    assert "direct" in error["details"]["supported"]
