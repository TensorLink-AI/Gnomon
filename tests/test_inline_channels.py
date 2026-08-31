"""Inline channels: no tool parameter may be reachable only via a file.

An MCP client holds no filesystem. context_events gained an inline
channel first; these tests cover the rest of the surface — inline
``observations`` (the data itself), inline ``covariates``, and inline
``actuals`` — and pin the contract that both channels of each pair run
the identical validation, loudly.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from gnomon.contracts import GnomonError
from gnomon.covariates import covariates_from_rows, load_covariates
from gnomon.toolspec import runner_for, visible_tools

EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
DAY = timedelta(days=1)


def _stamp(index: int) -> str:
    return (EPOCH + index * DAY).isoformat()


def _observation_rows(n: int = 60) -> list[dict]:
    return [
        {"timestamp": _stamp(i),
         "value": 50.0 + 5.0 * math.sin(2 * math.pi * i / 7) + 0.1 * i}
        for i in range(n)
    ]


def _covariate_rows(n_history: int = 60, horizon: int = 3) -> list[dict]:
    return [
        {"timestamp": _stamp(i), "known_at": _stamp(0),
         "load": float(i % 7)}
        for i in range(n_history + horizon)
    ]


# -- inline observations ----------------------------------------------------

def test_forecast_runs_from_inline_observations(tmp_path):
    payload = runner_for("gnomon_forecast")({
        "observations": _observation_rows(), "time_column": "timestamp",
        "target_column": "value", "horizon": 3,
        "output_dir": str(tmp_path),
    })
    assert payload["status"] == "complete"
    assert payload["results"][0]["forecast_rows"] == 3


def test_inspect_runs_from_inline_observations():
    payload = runner_for("gnomon_inspect")({
        "observations": _observation_rows(16), "time_column": "timestamp",
        "target_column": "value",
    })
    assert payload.get("status") in ("ok", "complete") or "series" in payload


def test_observations_and_input_are_mutually_exclusive(tmp_path):
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({
            "observations": _observation_rows(), "input": "history.csv",
            "time_column": "timestamp", "target_column": "value",
            "horizon": 3, "output_dir": str(tmp_path),
        })
    assert caught.value.code == "INVALID_ARGUMENTS"


def test_missing_both_channels_teaches_both():
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({
            "time_column": "timestamp", "target_column": "value",
            "horizon": 3,
        })
    assert caught.value.code == "INVALID_ARGUMENTS"
    assert "observations" in caught.value.message
    assert "input" in caught.value.message


def test_malformed_observations_fail_loudly():
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({
            "observations": ["not-a-row"], "time_column": "timestamp",
            "target_column": "value", "horizon": 3,
        })
    assert caught.value.code == "INVALID_ARGUMENTS"


def test_every_data_tool_offers_the_inline_channel():
    expects = {
        "gnomon_inspect", "gnomon_forecast",
        "gnomon_validate_covariates",
        "gnomon_preflight_context", "gnomon_route", "gnomon_ingest",
        "gnomon_investigate_change", "gnomon_detect_anomalies",
        "gnomon_decide", "gnomon_monitor",
    }
    for tool in visible_tools():
        properties = tool["inputSchema"].get("properties") or {}
        if tool["name"] in expects:
            assert "observations" in properties, tool["name"]
            # input must not be schema-required when observations can
            # replace it — clients enforce required lists client-side.
            assert "input" not in tool["inputSchema"].get("required", []), \
                tool["name"]


# -- inline covariates ------------------------------------------------------

def test_inline_rows_equal_the_csv_loader(tmp_path):
    rows = _covariate_rows()
    csv_path = tmp_path / "covariates.csv"
    lines = ["timestamp,known_at,load"] + [
        f"{row['timestamp']},{row['known_at']},{row['load']}" for row in rows
    ]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from_file = load_covariates(str(csv_path), "load:continuous:future_known")
    from_rows = covariates_from_rows(rows, "load:continuous:future_known")
    assert from_rows.path == "inline"
    probe = (EPOCH + 5 * DAY, EPOCH + 60 * DAY)
    for dataset in (from_file, from_rows):
        dataset.bind_as_of(None)
    assert from_file.value_at("load", "__default__", *probe) \
        == from_rows.value_at("load", "__default__", *probe)


def test_inline_covariate_rows_validate_like_the_file():
    with pytest.raises(GnomonError) as caught:
        covariates_from_rows(
            [{"timestamp": _stamp(0), "known_at": _stamp(0)}],  # no 'load'
            "load:continuous:future_known",
        )
    assert caught.value.code == "MISSING_COVARIATE_COLUMNS"
    assert "row 1" in caught.value.message

    with pytest.raises(GnomonError) as caught:
        covariates_from_rows(["nope"], "load:continuous:future_known")
    assert caught.value.code == "INVALID_COVARIATE_ROW"


def test_forecast_considers_inline_covariates(tmp_path):
    # The covariate gate detail lives in the full payload (brief keeps
    # only the epistemics); this test is about the inline channel, so it
    # opts into full deliberately.
    payload = runner_for("gnomon_forecast")({
        "observations": _observation_rows(), "time_column": "timestamp",
        "target_column": "value", "horizon": 3,
        "covariates": _covariate_rows(),
        "covariate_mapping": "load:continuous:future_known",
        "output_dir": str(tmp_path),
        "format": "full",
    })
    covariates = payload["results"][0]["covariates"]
    assert covariates["considered"] is True


def test_multi_target_forecast_considers_inline_covariates(tmp_path):
    observations = []
    covariates = []
    for i in range(63):
        if i < 60:
            observations.append({
                "timestamp": _stamp(i), "a": 10 + math.sin(i),
                "b": 20 + math.cos(i),
            })
        for series in ("a", "b"):
            covariates.append({
                "timestamp": _stamp(i), "known_at": _stamp(0),
                "series": series, "load": float(i % 7),
            })
    payload = runner_for("gnomon_forecast")({
        "observations": observations, "time_column": "timestamp",
        "target_column": "a,b", "horizon": 3,
        "covariates": covariates,
        "covariate_mapping": "load:continuous:future_known",
        "covariate_series_column": "series",
        "output_dir": str(tmp_path), "format": "full",
    })
    assert all(item["covariates"]["considered"]
               for item in payload["results"])


def test_covariate_channel_misuse_is_loud(tmp_path):
    base = {
        "observations": _observation_rows(), "time_column": "timestamp",
        "target_column": "value", "horizon": 3, "output_dir": str(tmp_path),
    }
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({
            **base, "covariates": _covariate_rows(),
            "covariates_file": "x.csv",
            "covariate_mapping": "load:continuous:future_known",
        })
    assert "not both" in caught.value.message
    with pytest.raises(GnomonError) as caught:
        runner_for("gnomon_forecast")({
            **base, "covariate_mapping": "load:continuous:future_known",
        })
    assert "without covariate rows" in caught.value.message


def test_validate_covariates_accepts_inline_rows():
    payload = runner_for("gnomon_validate_covariates")({
        "observations": _observation_rows(), "time_column": "timestamp",
        "target_column": "value", "horizon": 3,
        "covariates": _covariate_rows(),
        "covariate_mapping": "load:continuous:future_known",
    })
    assert payload["covariates_path"] == "inline"
    assert "valid" in payload


# -- inline actuals ---------------------------------------------------------

class _StoreStub:
    def __init__(self):
        self.submitted = None

    def submit_actuals(self, project, actuals):
        self.submitted = (project, actuals)
        return []

    def explain_unscored(self, project, timestamps):
        return {"scored": 0, "explanation": f"{len(timestamps)} probed"}


def test_submit_actuals_accepts_inline_rows(monkeypatch):
    import gnomon.tracking

    stub = _StoreStub()
    monkeypatch.setattr(gnomon.tracking, "TrackingStore", lambda: stub)
    payload = runner_for("gnomon_submit_actuals")({
        "project": "p",
        "actuals": [
            {"timestamp": _stamp(60), "value": 51.0},
            {"timestamp": _stamp(61), "value": 52.5, "series": "s1"},
        ],
    })
    assert payload["status"] == "ok"
    project, tuples = stub.submitted
    assert project == "p"
    assert tuples == [(_stamp(60), 51.0), ("s1", _stamp(61), 52.5)]
    assert payload["explanation"] == "2 probed"


def test_submit_actuals_channel_misuse_is_loud(monkeypatch):
    import gnomon.tracking

    monkeypatch.setattr(gnomon.tracking, "TrackingStore", lambda: _StoreStub())
    runner = runner_for("gnomon_submit_actuals")
    with pytest.raises(GnomonError) as caught:
        runner({"project": "p", "actuals": [], "actuals_file": "a.csv"})
    assert "not both" in caught.value.message
    with pytest.raises(GnomonError) as caught:
        runner({"project": "p"})
    assert "actuals" in caught.value.message
    with pytest.raises(GnomonError) as caught:
        runner({"project": "p", "actuals": [{"timestamp": _stamp(1)}]})
    assert "finite number" in caught.value.message


# -- provenance: the artifact records the channel ---------------------------

def test_inline_observations_stamp_provenance_and_assumption(tmp_path):
    """A forecast built on rows the caller typed says so, twice: a
    `provenance: inline` field on the persisted task, and an assumption in
    the response. A file at least existed outside the conversation; inline
    rows did not, and a reader weighing the numbers is owed the fact."""
    import json
    from pathlib import Path

    payload = runner_for("gnomon_forecast")({
        "observations": _observation_rows(), "time_column": "timestamp",
        "target_column": "value", "horizon": 3,
        "output_dir": str(tmp_path),
    })
    assert payload["status"] == "complete"
    assumptions = payload["results"][0]["support_assessment"]["assumptions"]
    assert any("supplied inline" in item for item in assumptions)
    artifact = json.loads(
        (Path(payload["artifact_path"]) / "artifact.json").read_text()
    )
    assert artifact["task"]["provenance"] == "inline"


def test_inline_monitor_stamps_provenance_on_task_sources(tmp_path):
    import json
    from pathlib import Path

    payload = runner_for("gnomon_monitor")({
        "observations": _observation_rows(), "time_column": "timestamp",
        "target_column": "value", "horizon": 3, "threshold": 100.0,
        "output_dir": str(tmp_path),
    })
    artifact = json.loads(
        (Path(payload["artifact_path"]) / "artifact.json").read_text()
    )
    assert all(source["provenance"] == "inline"
               for source in artifact["task"]["sources"])
    # The MCP runner exposes the same pure event projection as the CLI. Its
    # live rate includes the durable ledger; the immutable artifact records
    # only this evaluation and says so.
    assert payload["events"] == artifact["events"]
    assert payload["firing_rate"]["history_complete"] is None
    assert payload["firing_rate"]["history_scope"] == \
        "content-distinct evaluations in this state file"
    assert artifact["firing_rate"]["history_complete"] is False


def test_file_runs_carry_no_provenance_key(tmp_path):
    """The implied default is not serialised: artifacts from file runs are
    byte-identical to what they were before the field existed."""
    import csv
    import json
    from pathlib import Path

    source = tmp_path / "series.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "value"])
        writer.writeheader()
        writer.writerows(_observation_rows())
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 3,
        "output_dir": str(tmp_path / "out"),
    })
    artifact = json.loads(
        (Path(payload["artifact_path"]) / "artifact.json").read_text()
    )
    assert "provenance" not in artifact["task"]
    assumptions = payload["results"][0]["support_assessment"]["assumptions"]
    assert not any("supplied inline" in item for item in assumptions)


def test_covariate_mapping_accepts_the_list_form():
    """The schema documents a comma-joined string; an agent handing a JSON
    array is guessing the unambiguous thing. Both parse identically —
    crashing on the guess was a dead end (raw AttributeError, no repair)."""
    from gnomon.covariates import parse_mapping

    joined = parse_mapping("load:continuous:future_known,promo:binary:future_known")
    listed = parse_mapping(["load:continuous:future_known",
                            "promo:binary:future_known"])
    assert joined == listed
    with pytest.raises(GnomonError):
        parse_mapping(["load:continuous"])  # still the typed error


def test_covariate_mapping_accepts_declared_cyclic_periods():
    from gnomon.covariates import parse_mapping

    spec = parse_mapping([{
        "name": "minute_of_day", "type": "cyclic_1440",
        "availability": "future_known",
    }])[0]
    assert spec.value_type == "cyclic_1440"
    with pytest.raises(GnomonError, match="unsupported type"):
        parse_mapping("minute_of_day:cyclic_daily:future_known")
