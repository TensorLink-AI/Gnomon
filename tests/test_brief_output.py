"""Brief output mode: less bulk, identical epistemics.

The hard rule under test: brief mode may drop volume (extra quantile
levels, the raw point path, gate detail) but may never drop a support
state, a warning, an abstention reason, a recovery action, or a
disclosure — those ride verbatim, and an abstention serialises the same
structured support assessment full mode carries. The artifact on disk is
byte-identical with and without --brief.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gnomon.ids import FixedClock
from gnomon.runtime import forecast
from gnomon.toolspec import brief_summary, forecast_summary, runner_for

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
REPO = Path(__file__).resolve().parent.parent
DAILY = str(REPO / "examples" / "daily_requests.csv")
SHORT = str(REPO / "tests" / "data" / "golden_short.csv")


def _wide_file(path: Path, rows: int = 96) -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "hr", "spo2"])
        for k in range(rows):
            writer.writerow([
                (start + timedelta(hours=k)).isoformat(),
                repr(70 + 5 * math.sin(2 * math.pi * k / 24)),
                repr(97 + 0.1 * (k % 5)),
            ])


def test_brief_keeps_one_interval_and_drops_the_bulk(tmp_path):
    artifact, path = forecast(
        DAILY, time_column="timestamp", target_column="requests",
        horizon=7, output=str(tmp_path), clock=CLOCK,
    )
    brief = brief_summary(artifact, path)
    assert brief["format"] == "brief"
    assert brief["forecast_id"] == artifact.forecast_id
    result = brief["results"][0]
    assert len(result["forecast"]) == 7  # the whole horizon, not a preview
    for row in result["forecast"]:
        assert set(row) == {"timestamp", "q10", "q50", "q90", "tier"}
    full_rows = artifact.results[0].forecast
    assert [row["q50"] for row in result["forecast"]] == [
        row["q50"] for row in full_rows
    ]


def test_brief_never_omits_support_warnings_or_disclosures(tmp_path):
    # golden_short.csv produces a degraded run with warnings and
    # disclosures — exactly what brief mode is forbidden to hide.
    artifact, path = forecast(
        SHORT, time_column="timestamp", target_column="value",
        horizon=3, output=str(tmp_path), clock=CLOCK,
    )
    brief = brief_summary(artifact, path)["results"][0]
    full = forecast_summary(artifact, path)["results"][0]
    assert full["warnings"], "fixture no longer produces warnings"
    assert brief["support"] == full["support"]
    assert brief["warnings"] == full["warnings"]  # verbatim, same objects
    assert brief["support_assessment"] == full["support_assessment"]
    assert brief["support_assessment"]["disclosures"] == (
        full["support_assessment"]["disclosures"]
    )
    assert brief["support_assessment"]["recovery_actions"] == (
        full["support_assessment"]["recovery_actions"]
    )


def test_brief_abstention_is_the_same_structured_abstention(tmp_path):
    artifact, path = forecast(
        SHORT, time_column="timestamp", target_column="value",
        horizon=3, strict_abstention=True, output=str(tmp_path), clock=CLOCK,
    )
    assert artifact.results[0].support == "unsupported", (
        "fixture no longer abstains under strict abstention"
    )
    brief = brief_summary(artifact, path)["results"][0]
    full = forecast_summary(artifact, path)["results"][0]
    assert brief["support"] == "unsupported"
    assert brief["forecast"] == []
    assert brief["support_assessment"] == full["support_assessment"]
    assert brief["warnings"] == full["warnings"]


def test_cli_brief_changes_stdout_but_not_the_artifact(tmp_path, capsys):
    from gnomon.cli import main

    wide = tmp_path / "wide.csv"
    _wide_file(wide)
    assert main([
        "forecast", str(wide), "--target", "hr", "--horizon", "6",
        "--output", str(tmp_path / "full-out"),
    ]) == 0
    full_payload = json.loads(capsys.readouterr().out)
    assert main([
        "forecast", str(wide), "--target", "hr", "--horizon", "6",
        "--brief", "--output", str(tmp_path / "brief-out"),
    ]) == 0
    brief_payload = json.loads(capsys.readouterr().out)
    assert "format" not in full_payload  # default stdout is unchanged
    assert brief_payload["format"] == "brief"
    assert brief_payload["forecast_id"] == full_payload["forecast_id"]
    # The artifact written under --brief is byte-identical to the full run's.
    identifier = full_payload["forecast_id"]
    full_artifact = (tmp_path / "full-out" / identifier / "artifact.json").read_bytes()
    brief_artifact = (tmp_path / "brief-out" / identifier / "artifact.json").read_bytes()
    # created_at differs by wall clock; compare everything but that field.
    full_data = json.loads(full_artifact)
    brief_data = json.loads(brief_artifact)
    full_data.pop("created_at"), brief_data.pop("created_at")
    assert full_data == brief_data
    # CLI-level inferences still reach the brief envelope.
    assessment = brief_payload["results"][0]["support_assessment"]
    assert "assumptions" in assessment


def test_cli_brief_works_with_multi_target(tmp_path, capsys):
    from gnomon.cli import main

    wide = tmp_path / "wide.csv"
    _wide_file(wide)
    assert main([
        "forecast", str(wide), "--target", "hr,spo2", "--horizon", "6",
        "--brief", "--output", str(tmp_path / "out"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "brief"
    assert [result["series"] for result in payload["results"]] == ["hr", "spo2"]
    for result in payload["results"]:
        assert result["support_assessment"] is not None
        assert set(result["forecast"][0]) == {"timestamp", "q10", "q50", "q90", "tier"}


def test_mcp_forecast_accepts_brief_format(tmp_path):
    payload = runner_for("gnomon_forecast")({
        "input": DAILY, "time_column": "timestamp",
        "target_column": "requests", "horizon": 7, "format": "brief",
        "output_dir": str(tmp_path),
    })
    assert payload["format"] == "brief"
    result = payload["results"][0]
    assert set(result["forecast"][0]) == {"timestamp", "q10", "q50", "q90", "tier"}
    assert result["support_assessment"] is not None
