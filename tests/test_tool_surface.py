"""Tool-surface gating and token discipline.

The MCP surface competes for model attention and context tokens; these
tests pin the subtraction: deprecated and near-duplicate tools stay off
the default surface (returning only under GNOMON_V02_COMPAT=1), and no
default response may balloon past the size budget.
"""

from __future__ import annotations


def _names(monkeypatch, compat: bool = False) -> list[str]:
    from gnomon.toolspec import visible_tools

    if compat:
        monkeypatch.setenv("GNOMON_V02_COMPAT", "1")
    else:
        monkeypatch.delenv("GNOMON_V02_COMPAT", raising=False)
    return [tool["name"] for tool in visible_tools()]


# --- Fix 1: the deprecated decision pair is opt-in -------------------------

def test_deprecated_decision_pair_hidden_by_default(monkeypatch) -> None:
    from gnomon.toolspec import runner_for

    names = _names(monkeypatch)
    assert "gnomon_record_decision" not in names
    assert "gnomon_resolve_decision" not in names
    assert runner_for("gnomon_record_decision") is None
    assert runner_for("gnomon_resolve_decision") is None


def test_v02_compat_restores_decision_pair(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import runner_for
    from gnomon.tracking import TrackingStore

    names = _names(monkeypatch, compat=True)
    assert "gnomon_record_decision" in names
    assert "gnomon_resolve_decision" in names

    # They do not just appear — they still function end to end.
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    TrackingStore().register(
        forecast_id="fc_compat", project="compat", selected_model="drift",
        support="supported", artifact_path=str(tmp_path),
    )
    record = runner_for("gnomon_record_decision")({
        "decision_id": "d1", "project": "compat", "forecast_id": "fc_compat",
        "action": "scale_up", "expected_outcome": "load absorbed",
    })
    assert record["status"] == "ok"
    resolved = runner_for("gnomon_resolve_decision")({
        "decision_id": "d1", "actual_outcome": "load absorbed",
        "correct": True,
    })
    assert resolved["status"] == "ok"
    assert resolved["decision"]["correct"] is True


# --- Fix 2: the tracking reads consolidate into gnomon_status --------------

def test_tracking_read_tools_hidden_by_default(monkeypatch) -> None:
    names = _names(monkeypatch)
    assert "gnomon_list_open_forecasts" not in names
    assert "gnomon_model_performance" not in names
    assert "gnomon_status" in names
    compat_names = _names(monkeypatch, compat=True)
    assert "gnomon_list_open_forecasts" in compat_names
    assert "gnomon_model_performance" in compat_names


def test_status_sections_match_the_retired_tools(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import (
        _run_model_performance,
        _run_open_forecasts,
        _run_status,
    )
    from gnomon.tracking import TrackingStore

    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    TrackingStore().register(
        forecast_id="fc_status", project="status", selected_model="drift",
        support="supported", artifact_path=str(tmp_path), horizon=3,
    )
    args = {"project": "status"}
    assert _run_status({**args, "section": "open_forecasts"}) \
        == _run_open_forecasts(args)
    assert _run_status({**args, "section": "performance"}) \
        == _run_model_performance(args)

    # The decisions slice is exactly the decision fields of the full view.
    full = _run_status(args)
    decisions = _run_status({**args, "section": "decisions"})
    assert decisions["unresolved_decisions"] == full["unresolved_decisions"]
    assert decisions["decision_summary"] == full["decision_summary"]
    assert "open_forecasts" not in decisions

    # performance still requires a project, as the standalone tool did.
    import pytest

    from gnomon.contracts import GnomonError
    with pytest.raises(GnomonError):
        _run_status({"section": "performance"})


# --- Fix 3: guide/proposer/duplicate tools leave the default surface -------

def test_covariate_guide_and_proposer_tools_gated(monkeypatch) -> None:
    names = _names(monkeypatch)
    for gated in ("gnomon_covariate_guide", "gnomon_propose_covariates",
                  "gnomon_proposer_skill"):
        assert gated not in names, gated
    # The survivors still cover the workflow: validation carries the
    # format contract, and the forecast takes every covariate argument.
    from gnomon.toolspec import TOOLS
    by_name = {tool["name"]: tool for tool in TOOLS}
    validate = by_name["gnomon_validate_covariates"]["description"]
    assert "known_at" in validate
    assert "name:type:future_known" in validate
    forecast_props = by_name["gnomon_forecast"]["inputSchema"]["properties"]
    assert "covariates_file" in forecast_props
    assert "covariate_mapping" in forecast_props

    compat = set(_names(monkeypatch, compat=True))
    for gated in ("gnomon_covariate_guide", "gnomon_propose_covariates",
                  "gnomon_proposer_skill"):
        assert gated in compat, gated


# --- Fix 4: brief by default, hard response budget -------------------------

def _panel_csv(tmp_path, rows_per_series: int = 60):
    # Regular daily grid per series, distinct values.
    lines = ["timestamp,series,value"]
    from datetime import date, timedelta
    start = date(2026, 1, 1)
    for series in ("api", "batch"):
        base = 100 if series == "api" else 500
        for day in range(rows_per_series):
            lines.append(f"{start + timedelta(days=day)},{series},{base + day}")
    path = tmp_path / "panel.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_forecast_defaults_to_brief(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": "examples/daily_requests.csv", "horizon": 3,
        "output_dir": str(tmp_path),
    })
    assert payload["format"] == "brief"
    # Brief still carries the whole epistemic contract.
    result = payload["results"][0]
    assert "support_assessment" in result
    assert "warnings" in result


def test_multi_series_forecast_respects_the_budget(tmp_path) -> None:
    import csv
    import json

    from gnomon.toolspec import RESPONSE_BUDGET_BYTES, runner_for

    data = _panel_csv(tmp_path)
    payload = runner_for("gnomon_forecast")({
        "input": str(data), "series_column": "series", "horizon": 30,
        "output_dir": str(tmp_path / "out"),
    })
    raw = json.dumps(payload)
    assert payload.get("truncated") is True
    assert "artifact" in payload["truncation"]["note"]
    # The trim leaves headroom-sized responses: within budget plus the
    # protected epistemics and the truncation metadata itself.
    assert len(raw) < RESPONSE_BUDGET_BYTES * 2
    for result in payload["results"]:
        assert result["support_assessment"], "epistemics must survive the trim"
        assert result["forecast_rows"] == 30
        assert len(result["forecast"]) < 30

    # The artifact on disk still carries every row.
    with open(payload["artifact_path"] + "/forecast.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60


def test_budget_never_trims_the_contract() -> None:
    from gnomon.toolspec import enforce_response_budget

    payload = {
        "status": "complete",
        "artifact_path": "/tmp/a",
        "warnings": [f"warning {index}" for index in range(400)],
        "support_assessment": {
            "reasons": [{"code": "x", "message": "m" * 40}] * 50,
        },
        "bulk": list(range(5000)),
    }
    trimmed = enforce_response_budget(payload)
    assert trimmed["truncated"] is True
    assert trimmed["warnings"] == payload["warnings"]
    assert trimmed["support_assessment"] == payload["support_assessment"]
    assert len(trimmed["bulk"]) == 5
    numeric = next(item for item in trimmed["truncation"]["trimmed"]
                   if item["path"] == "bulk")
    assert numeric["summary"]["min"] == 0
    assert numeric["summary"]["max"] == 4999

    error = {"status": "error", "error": {"code": "X", "message": "y" * 20000}}
    assert enforce_response_budget(error) is error


def test_capabilities_reports_compat_state(monkeypatch) -> None:
    from gnomon.runtime import capabilities

    monkeypatch.delenv("GNOMON_V02_COMPAT", raising=False)
    assert capabilities()["compat"]["v02_tools"] is False
    monkeypatch.setenv("GNOMON_V02_COMPAT", "1")
    assert capabilities()["compat"]["v02_tools"] is True
