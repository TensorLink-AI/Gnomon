"""The best-effort lane: an abstention with numbers attached.

CiK's Montreal Fire and Unemployment tasks give six monthly observations
and demand a seven-step forecast. Gnomon's abstention there is correct —
the horizon exceeds what the history can support — but callers scored on
producing *something* need rows. ``--best-effort`` publishes a naive
fallback that says what it is three ways: support ``best_effort``, the
NO RELIABLE FORECAST warning verbatim, and a descriptive (never
predictive) lineage claim. Off by default; every flag-off artifact is
byte-identical.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from gnomon.pipeline import NO_RELIABLE_FORECAST
from gnomon.runtime import forecast


def write_monthly(path: Path, count: int) -> None:
    values = [12.0, 15.0, 11.0, 18.0, 14.0, 16.0, 13.0, 17.0]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "value"])
        writer.writeheader()
        year, month = 2025, 9
        for index in range(count):
            writer.writerow({
                "timestamp": datetime(year, month, 1).isoformat(),
                "value": values[index % len(values)],
            })
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def cik_bucket_three(tmp_path: Path, **kwargs):
    """Six monthly observations, horizon seven — the exact abstention shape."""
    source = tmp_path / "monthly.csv"
    write_monthly(source, 6)
    return forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "output"), **kwargs,
    )


def test_the_default_publishes_the_labelled_fallback(tmp_path: Path) -> None:
    # Graduated support: the default floor answers with disclosed
    # best_effort rows; the refusal is one parameter away.
    artifact, _ = cik_bucket_three(tmp_path)
    result = artifact.results[0]
    assert result.support == "best_effort"
    assert result.forecast


def test_minimum_support_supported_restores_the_abstention(tmp_path: Path) -> None:
    artifact, _ = cik_bucket_three(tmp_path, minimum_support="supported")
    result = artifact.results[0]
    assert result.support == "unsupported"
    assert result.forecast == []


def pure_fallback_run(tmp_path: Path, monkeypatch, **kwargs):
    """The whole-object fallback lane: fires when even the supportable
    prefix abstains (with three or more observations a small evaluated
    prefix normally exists, so the run becomes a horizon split instead).
    The prefix abstention is simulated here; the lane itself runs end to
    end, artifact and lineage included."""
    import gnomon.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_split_prefix",
                        lambda *args, **kw: None)
    source = tmp_path / "tiny.csv"
    write_monthly(source, 3)
    return forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "output"), **kwargs,
    )


def test_best_effort_publishes_labelled_rows(tmp_path: Path, monkeypatch) -> None:
    artifact, directory = pure_fallback_run(tmp_path, monkeypatch)
    result = artifact.results[0]
    assert result.support == "best_effort"
    assert len(result.forecast) == 7
    # The point path is the last observed value, flat — the most defensible
    # naive answer, not a covert model.
    last_value = 11.0
    assert all(row["point"] == last_value for row in result.forecast)
    # Intervals widen with lead time and stay ordered.
    first, last = result.forecast[0], result.forecast[-1]
    assert first["q10"] < first["q50"] < first["q90"]
    assert last["q90"] - last["q10"] > first["q90"] - first["q10"]
    # Every row carries the unstrippable tier label.
    assert all(row["tier"] == "best_effort" for row in result.forecast)
    # The disclosure is verbatim in the warnings.
    assert NO_RELIABLE_FORECAST in result.warnings
    # And the rows are written to the artifact directory like any others.
    persisted = json.loads((directory / "artifact.json").read_text())
    assert len(persisted["results"][0]["forecast"]) == 7


def test_best_effort_support_assessment_stays_inconclusive(tmp_path: Path, monkeypatch) -> None:
    """Rows change what was published, not what the evidence supports."""
    artifact, _ = pure_fallback_run(tmp_path, monkeypatch)
    assessment = artifact.results[0].support_assessment
    assert assessment["status"] == "inconclusive"
    codes = [reason["code"] for reason in assessment["reasons"]]
    assert codes[0] == "no_reliable_forecast"
    recovery = [reason["code"] for reason in assessment["recovery_actions"]]
    assert "provide_more_history" in recovery


def test_best_effort_lineage_claim_is_descriptive(tmp_path: Path, monkeypatch) -> None:
    """The verifier's calibration gate must be unreachable from a fallback:
    the claim class is descriptive, with no calibration reference."""
    artifact, directory = pure_fallback_run(tmp_path, monkeypatch)
    lineage = json.loads((directory / "lineage.json").read_text())
    claims = [claim for claim in lineage["claims"]
              if claim["claim_id"].startswith("claim:best_effort:")]
    assert len(claims) == 1
    assert claims[0]["claim_class"] == "descriptive"
    assert claims[0].get("calibration_ref") is None
    assert "no measured accuracy" in claims[0]["statement"].lower()
    assert not any(claim["claim_class"] == "predictive"
                   for claim in lineage["claims"])


def test_split_lineage_claim_is_descriptive_and_labelled(tmp_path: Path) -> None:
    """A horizon split gets its own descriptive claim naming both ranges
    and the tiers present — never a predictive claim for the merged
    object."""
    artifact, directory = cik_bucket_three(tmp_path)
    result = artifact.results[0]
    assert result.support == "best_effort"
    codes = {reason["code"]
             for reason in result.support_assessment["reasons"]}
    assert "horizon_split" in codes
    lineage = json.loads((directory / "lineage.json").read_text())
    claims = [claim for claim in lineage["claims"]
              if claim["claim_id"].startswith("claim:horizon_split:")]
    assert len(claims) == 1
    assert claims[0]["claim_class"] == "descriptive"
    assert claims[0].get("calibration_ref") is None
    assert "best_effort" in claims[0]["statement"]
    assert not any(claim["claim_class"] == "predictive"
                   for claim in lineage["claims"])


def test_best_effort_does_not_touch_supported_forecasts(tmp_path: Path) -> None:
    """On data the evaluation handles, the flag is a no-op except the ID."""
    source = tmp_path / "monthly_long.csv"
    write_monthly(source, 60)
    common = dict(time_column="timestamp", target_column="value", horizon=3)
    plain, _ = forecast(str(source), output=str(tmp_path / "a"), **common)
    flagged, _ = forecast(str(source), output=str(tmp_path / "b"),
                          best_effort=True, **common)
    assert flagged.results[0].support == plain.results[0].support != "best_effort"
    assert flagged.results[0].forecast == plain.results[0].forecast


def test_flag_off_artifact_id_is_unchanged_by_the_feature(tmp_path: Path) -> None:
    """The ID payload carries the flag only when it is on."""
    source = tmp_path / "monthly.csv"
    write_monthly(source, 6)
    common = dict(time_column="timestamp", target_column="value", horizon=7)
    off_a, _ = forecast(str(source), output=str(tmp_path / "a"), **common)
    on, _ = forecast(str(source), output=str(tmp_path / "b"),
                     best_effort=True, **common)
    off_b, _ = forecast(str(source), output=str(tmp_path / "c"), **common)
    assert off_a.forecast_id == off_b.forecast_id
    assert on.forecast_id != off_a.forecast_id


def test_degenerate_history_yields_flat_intervals_with_a_warning(tmp_path: Path) -> None:
    source = tmp_path / "constant.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "value"])
        writer.writeheader()
        for month in (1, 2, 3):
            writer.writerow({"timestamp": datetime(2026, month, 1).isoformat(),
                             "value": 5.0})
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=6, output=str(tmp_path / "output"), best_effort=True,
    )
    result = artifact.results[0]
    assert result.support == "best_effort"
    assert all(row["q10"] == row["q90"] == row["point"] for row in result.forecast)
    assert any("no dispersion" in warning for warning in result.warnings)


def test_the_refusal_teaches_the_best_effort_path(tmp_path: Path) -> None:
    """An agent that has never read the CLI reference must learn the
    fallback lane from the refusal itself: the recovery actions name the
    minimum_support floor and how to lower it. (The refusal now exists
    only above the default floor.)"""
    artifact, _ = cik_bucket_three(tmp_path, minimum_support="supported")
    assessment = artifact.results[0].support_assessment
    recovery = {reason["code"]: reason["message"]
                for reason in assessment["recovery_actions"]}
    assert "retry_best_effort" in recovery
    assert "minimum_support" in recovery["retry_best_effort"]
    assert "--minimum-support" in recovery["retry_best_effort"]


def test_the_best_effort_run_does_not_advertise_itself(tmp_path: Path) -> None:
    """Once the fallback is already published, telling the caller to retry
    with the flag would be noise."""
    artifact, _ = cik_bucket_three(tmp_path, best_effort=True)
    assessment = artifact.results[0].support_assessment
    recovery = [reason["code"] for reason in assessment["recovery_actions"]]
    assert "retry_best_effort" not in recovery


def test_best_effort_is_reachable_from_the_mcp_tool(tmp_path: Path) -> None:
    """The MCP forecast tool is the primary agent front door; the lane
    must be one argument away, not CLI-only."""
    from gnomon.toolspec import runner_for, visible_tools

    source = tmp_path / "monthly.csv"
    write_monthly(source, 6)
    arguments = {
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 7,
        "output_dir": str(tmp_path / "out"),
    }
    runner = runner_for("gnomon_forecast")
    refused = runner({**arguments, "minimum_support": "supported"})
    result = refused["results"][0]
    assert result["support"] == "unsupported"
    assert result["forecast_rows"] == 0
    codes = [reason["code"] for reason in
             result["support_assessment"]["recovery_actions"]]
    assert "retry_best_effort" in codes

    # The default floor publishes the same labelled fallback the flag
    # used to require.
    published = runner({**arguments, "output_dir": str(tmp_path / "out2")})
    result = published["results"][0]
    assert result["support"] == "best_effort"
    assert result["forecast_rows"] == 7
    assert NO_RELIABLE_FORECAST in result["warnings"]

    spec = next(tool for tool in visible_tools()
                if tool["name"] == "gnomon_forecast")
    assert "minimum_support" in spec["inputSchema"]["properties"]
    assert "best_effort" in spec["inputSchema"]["properties"]
