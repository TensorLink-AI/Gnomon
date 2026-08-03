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


def test_without_the_flag_the_abstention_is_unchanged(tmp_path: Path) -> None:
    artifact, _ = cik_bucket_three(tmp_path)
    result = artifact.results[0]
    assert result.support == "unsupported"
    assert result.forecast == []


def test_best_effort_publishes_labelled_rows(tmp_path: Path) -> None:
    artifact, directory = cik_bucket_three(tmp_path, best_effort=True)
    result = artifact.results[0]
    assert result.support == "best_effort"
    assert len(result.forecast) == 7
    # The point path is the last observed value, flat — the most defensible
    # naive answer, not a covert model.
    last_value = 16.0
    assert all(row["point"] == last_value for row in result.forecast)
    # Intervals widen with lead time and stay ordered.
    first, last = result.forecast[0], result.forecast[-1]
    assert first["q10"] < first["q50"] < first["q90"]
    assert last["q90"] - last["q10"] > first["q90"] - first["q10"]
    # The disclosure is verbatim in the warnings, alongside the original
    # abstention message naming the supportable horizon.
    assert NO_RELIABLE_FORECAST in result.warnings
    assert any("horizon" in warning for warning in result.warnings
               if warning != NO_RELIABLE_FORECAST)
    # And the rows are written to the artifact directory like any others.
    persisted = json.loads((directory / "artifact.json").read_text())
    assert len(persisted["results"][0]["forecast"]) == 7


def test_best_effort_support_assessment_stays_inconclusive(tmp_path: Path) -> None:
    """Rows change what was published, not what the evidence supports."""
    artifact, _ = cik_bucket_three(tmp_path, best_effort=True)
    assessment = artifact.results[0].support_assessment
    assert assessment["status"] == "inconclusive"
    codes = [reason["code"] for reason in assessment["reasons"]]
    assert codes[0] == "no_reliable_forecast"
    recovery = [reason["code"] for reason in assessment["recovery_actions"]]
    assert "reduce_horizon" in recovery


def test_best_effort_lineage_claim_is_descriptive(tmp_path: Path) -> None:
    """The verifier's calibration gate must be unreachable from a fallback:
    the claim class is descriptive, with no calibration reference."""
    artifact, directory = cik_bucket_three(tmp_path, best_effort=True)
    lineage = json.loads((directory / "lineage.json").read_text())
    claims = [claim for claim in lineage["claims"]
              if claim["claim_id"].startswith("claim:best_effort:")]
    assert len(claims) == 1
    assert claims[0]["claim_class"] == "descriptive"
    assert claims[0].get("calibration_ref") is None
    assert "no measured accuracy" in claims[0]["statement"].lower()
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
