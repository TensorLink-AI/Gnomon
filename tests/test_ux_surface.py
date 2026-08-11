"""Regression tests for the UX-review fixes.

Each test pins a behaviour that was observed broken in the 2026-08 UX
review: silent state creation on read paths, guidance pointing at the
wrong surface, undisclosed vocabularies, and JSON-only output for humans.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnomon.cli import main
from gnomon.contracts import (
    COVERAGE_WARNING_THRESHOLD,
    GnomonError,
    MIN_VERIFIABLE_COVERAGE,
    coverage_warning,
)

DATA = Path(__file__).resolve().parent.parent / "examples" / "daily_requests.csv"


def _write_tiny(tmp_path: Path) -> Path:
    tiny = tmp_path / "tiny.csv"
    tiny.write_text(
        "timestamp,value\n2026-01-01,5\n2026-01-02,6\n2026-01-03,7\n2026-01-04,9\n"
    )
    return tiny


# --- stores must not materialise on read paths -----------------------------

def test_tracking_store_read_only_mode_creates_nothing(tmp_path) -> None:
    from gnomon.tracking import TrackingStore

    path = tmp_path / "registry.db"
    store = TrackingStore(path, create=False)
    status = store.status(None)
    assert status["open_forecasts"] == []
    assert not path.exists()


def test_temporal_store_read_creates_nothing(tmp_path) -> None:
    from gnomon.temporal_store import TemporalStore

    path = tmp_path / "temporal.db"
    store = TemporalStore(path)
    assert store.list_datasets() == []
    with pytest.raises(GnomonError) as excinfo:
        store.snapshot("nope")
    assert excinfo.value.code == "DATASET_NOT_FOUND"
    assert not path.exists()


# --- calibration thresholds ------------------------------------------------

def test_coverage_warning_threshold_and_wording() -> None:
    assert coverage_warning(None) is None
    assert coverage_warning(COVERAGE_WARNING_THRESHOLD) is None
    warned = coverage_warning(0.6)
    assert warned is not None and "below 70%" in warned
    # The warning tier sits strictly above the verifier band: a run can be
    # warned yet still carry probability weight.
    assert COVERAGE_WARNING_THRESHOLD > MIN_VERIFIABLE_COVERAGE


# --- capabilities reports the tool surface ---------------------------------

def test_capabilities_reports_tool_surface(capsys) -> None:
    from gnomon.toolspec import visible_tools

    assert main(["capabilities"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tools"]["mcp_count"] == len(visible_tools())
    assert "gnomon_forecast" in result["tools"]["mcp"]


# --- CLI flag semantics ----------------------------------------------------

def test_ensemble_flag_conflict_is_an_error(capsys) -> None:
    result = main([
        "forecast", str(DATA), "--horizon", "3",
        "--ensemble", "--selection-strategy", "best",
    ])
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "INVALID_ARGUMENTS"
    assert "--ensemble" in error["error"]["message"]


def test_store_path_rejected_with_multi_target(capsys, tmp_path) -> None:
    wide = tmp_path / "wide.csv"
    wide.write_text(
        "timestamp,hr,spo2\n2026-01-01,60,98\n2026-01-02,61,97\n2026-01-03,62,98\n"
    )
    result = main([
        "forecast", str(wide), "--target", "hr,spo2", "--horizon", "1",
        "--store-path", str(tmp_path / "store.db"),
        "--output", str(tmp_path / "out"),
    ])
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert "--store-path" in error["error"]["message"]


def test_track_export_refuses_silent_overwrite(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    target = tmp_path / "export.json"
    target.write_text("{}")
    result = main(["track", "export", "--output", str(target)])
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "INVALID_ARGUMENTS"
    assert target.read_text() == "{}"


def test_config_not_found_is_a_config_error(capsys) -> None:
    result = main([
        "forecast", str(DATA), "--horizon", "3", "--config", "/nope/gnomon.yaml",
    ])
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "CONFIG_NOT_FOUND"


# --- human output ----------------------------------------------------------

def test_format_human_renders_abstention_with_recovery(capsys, tmp_path) -> None:
    tiny = _write_tiny(tmp_path)
    result = main([
        "forecast", str(tiny), "--horizon", "14", "--format", "human",
        "--output", str(tmp_path / "out"),
    ])
    assert result == 0
    out = capsys.readouterr().out
    assert "NO FORECAST PUBLISHED" in out
    assert "Next (reduce_horizon)" in out
    # Human mode is a rendering, not a different result: no JSON braces.
    assert not out.lstrip().startswith("{")


def test_format_auto_stays_json_when_piped(capsys, tmp_path) -> None:
    result = main([
        "forecast", str(DATA), "--horizon", "3",
        "--output", str(tmp_path / "out"),
    ])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"


def test_brief_conflicts_with_format_human(capsys, tmp_path) -> None:
    result = main([
        "forecast", str(DATA), "--horizon", "3", "--brief", "--format", "human",
        "--output", str(tmp_path / "out"),
    ])
    assert result == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "INVALID_ARGUMENTS"


def test_forecast_summary_md_carries_numbers_and_recovery(tmp_path, capsys) -> None:
    out_dir = tmp_path / "out"
    assert main([
        "forecast", str(DATA), "--horizon", "3", "--output", str(out_dir),
    ]) == 0
    capsys.readouterr()
    artifact_dir = next(out_dir.glob("forecast_*"))
    summary = (artifact_dir / "summary.md").read_text()
    assert "### Forecast" in summary
    assert "| timestamp | q50 | q10 | q90 |" in summary


def test_abstention_summary_md_carries_recovery(tmp_path, capsys) -> None:
    tiny = _write_tiny(tmp_path)
    out_dir = tmp_path / "out"
    assert main([
        "forecast", str(tiny), "--horizon", "14", "--output", str(out_dir),
    ]) == 0
    capsys.readouterr()
    artifact_dir = next(out_dir.glob("forecast_*"))
    summary = (artifact_dir / "summary.md").read_text()
    assert "Next (reduce_horizon)" in summary


# --- config honesty --------------------------------------------------------

def test_missing_pyyaml_refuses_to_ignore_config(tmp_path, monkeypatch) -> None:
    import builtins

    config = tmp_path / "gnomon.yaml"
    config.write_text("models:\n  statistical: [drift]\n")
    real_import = builtins.__import__

    def no_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    from gnomon.config import load_config

    with pytest.raises(GnomonError) as excinfo:
        load_config(str(config))
    assert excinfo.value.code == "MISSING_OPTIONAL_DEPENDENCY"


def test_decide_and_monitor_accept_config(tmp_path, capsys) -> None:
    config = tmp_path / "gnomon.yaml"
    config.write_text("evaluation:\n  minimum_baseline_improvement: 0.02\n")
    result = main([
        "monitor", str(DATA), "--horizon", "3", "--threshold", "300",
        "--config", str(config), "--output", str(tmp_path / "out"),
    ])
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "0.1"
