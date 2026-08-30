"""The disclosed-repair layer: messy files parse under safe/aggressive
repair, every fix is logged, assumptive fixes downgrade support, and
excessive messiness is an honest refusal rather than an invented series."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.repair import (
    AmbiguousDateOrder,
    parse_number,
    parse_timestamp_lenient,
    scan_day_first,
    scan_numeric_evidence,
)
from gnomon.runtime import forecast, inspect_dataset

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
REPO = Path(__file__).resolve().parent.parent


# --- unit: numeric leniency -------------------------------------------------

@pytest.mark.parametrize("text,expected,tier", [
    ("118", 118.0, "clean"),
    ("-2.5e3", -2500.0, "clean"),
    ("$149", 149.0, "normalised"),
    ("1 234", 1234.0, "normalised"),
    ("1'234.5", 1234.5, "normalised"),
    ("45%", 45.0, "normalised"),
    ("(200)", -200.0, "normalised"),
    ("€1.234,56", 1234.56, "normalised"),
    ("1,234.56", 1234.56, "normalised"),
    ("12,5", 12.5, "normalised"),
    ("1,449", 1449.0, "assumptive"),
])
def test_parse_number(text: str, expected: float, tier: str) -> None:
    value, got_tier = parse_number(text, None)
    assert value == pytest.approx(expected)
    assert got_tier == tier


@pytest.mark.parametrize("text", ["", "N/A", "na", "null", "-", "#N/A", "?"])
def test_parse_number_sentinels(text: str) -> None:
    assert parse_number(text, None) == (None, "missing")


def test_parse_number_unparseable() -> None:
    with pytest.raises(ValueError):
        parse_number("about twelve", None)


def test_numeric_evidence_decides_comma_role() -> None:
    assert scan_numeric_evidence(["1,234.5", "900"]) == "thousands"
    assert scan_numeric_evidence(["12,5", "900"]) == "decimal"
    assert scan_numeric_evidence(["100", "200"]) is None
    # With column evidence, the grouped case stops being assumptive.
    assert parse_number("1,449", "thousands") == (1449.0, "normalised")
    assert parse_number("1,449", "decimal") == (1.449, "normalised")


# --- unit: timestamp leniency -----------------------------------------------

def test_parse_timestamp_formats() -> None:
    assert parse_timestamp_lenient("2026-05-01", None) == (datetime(2026, 5, 1), "clean")
    assert parse_timestamp_lenient("2026-05-01T12:00:00Z", None)[0].tzinfo is not None
    assert parse_timestamp_lenient("2026/05/18", None) == (datetime(2026, 5, 18), "normalised")
    assert parse_timestamp_lenient("05 Mar 2026", None) == (datetime(2026, 3, 5), "normalised")
    assert parse_timestamp_lenient("20260518", None) == (datetime(2026, 5, 18), "clean")
    epoch, tier = parse_timestamp_lenient("1767225600", None)
    assert (epoch, tier) == (datetime(2026, 1, 1, tzinfo=timezone.utc), "normalised")
    millis, _ = parse_timestamp_lenient("1767225600000", None)
    assert millis == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_ambiguous_dates_need_evidence() -> None:
    # 18/05 forces day-first; 05/18 forces month-first.
    assert parse_timestamp_lenient("18/05/2026", None) == (datetime(2026, 5, 18), "normalised")
    assert parse_timestamp_lenient("05/18/2026", None) == (datetime(2026, 5, 18), "normalised")
    with pytest.raises(AmbiguousDateOrder):
        parse_timestamp_lenient("03/04/2026", None)
    assert parse_timestamp_lenient("03/04/2026", True)[0] == datetime(2026, 4, 3)
    assert parse_timestamp_lenient("03/04/2026", False)[0] == datetime(2026, 3, 4)
    assert scan_day_first(["18/05/2026", "03/04/2026"]) is True
    assert scan_day_first(["2026-05-01"]) is None
    with pytest.raises(GnomonError) as caught:
        scan_day_first(["18/05/2026", "05/18/2026"])
    assert caught.value.code == "AMBIGUOUS_DATE_ORDER"


# --- integration helpers ----------------------------------------------------

def write_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        writer.writerows(rows)


def daily_rows(count: int, start: datetime = datetime(2026, 1, 1)) -> list[tuple[str, str]]:
    return [
        ((start + timedelta(days=index)).date().isoformat(), str(100 + index))
        for index in range(count)
    ]


def run(path: Path, tmp_path: Path, **kwargs):
    return forecast(
        str(path), time_column="timestamp", target_column="value", horizon=3,
        output=str(tmp_path / "out"), clock=CLOCK, **kwargs,
    )


def repair_evidence(artifact) -> dict | None:
    for item in artifact.evidence:
        if item.kind == "data_repair":
            return item.payload
    return None


# --- integration: safe (the default) ----------------------------------------

def test_safe_normalises_text_and_discloses(tmp_path: Path) -> None:
    rows = daily_rows(30)
    rows[5] = (rows[5][0], "$105")
    rows.append(("", ""))                # trailing blank line
    rows.append(rows[10])                # byte-identical duplicate
    source = tmp_path / "messy.csv"
    write_rows(source, rows)
    artifact, _ = run(source, tmp_path)
    payload = repair_evidence(artifact)
    assert payload is not None and payload["level"] == "safe"
    codes = {action["code"] for action in payload["actions"]}
    assert {"numeric_format_normalised", "blank_row_skipped", "duplicate_row_collapsed"} <= codes
    # Text normalisation is not an assumption: support is untouched.
    assert artifact.results[0].support == "supported"
    assert not any("repaired_data" in warning for warning in artifact.results[0].warnings)


def test_clean_file_has_no_repair_trace(tmp_path: Path) -> None:
    source = tmp_path / "clean.csv"
    write_rows(source, daily_rows(30))
    artifact, _ = run(source, tmp_path)
    assert repair_evidence(artifact) is None


def test_safe_never_fills_gaps(tmp_path: Path) -> None:
    rows = daily_rows(30)
    rows[12] = (rows[12][0], "N/A")      # interior sentinel leaves a hole
    source = tmp_path / "gap.csv"
    write_rows(source, rows)
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path)
    assert caught.value.code == "IRREGULAR_TIME_GRID"


def test_off_preserves_strict_errors(tmp_path: Path) -> None:
    rows = daily_rows(30)
    rows[5] = (rows[5][0], "$105")
    source = tmp_path / "messy.csv"
    write_rows(source, rows)
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path, repair="off")
    assert caught.value.code == "INVALID_TARGET"


def test_safe_ambiguous_date_order_is_an_error(tmp_path: Path) -> None:
    rows = [(f"0{1 + index}/03/2026", str(100 + index)) for index in range(9)]
    source = tmp_path / "ambiguous.csv"
    write_rows(source, rows)
    with pytest.raises(GnomonError) as caught:
        inspect_dataset(str(source), time_column="timestamp", target_column="value")
    assert caught.value.code == "AMBIGUOUS_DATE_ORDER"


# --- integration: aggressive -------------------------------------------------

def test_aggressive_fills_gaps_and_downgrades_support(tmp_path: Path) -> None:
    rows = daily_rows(30)
    del rows[12]                          # a real missing day
    source = tmp_path / "gap.csv"
    write_rows(source, rows)
    artifact, _ = run(source, tmp_path, repair="aggressive")
    payload = repair_evidence(artifact)
    filled = [action for action in payload["actions"] if action["code"] == "gap_filled"]
    assert filled and filled[0]["count"] == 1 and filled[0]["assumptive"] is True
    result = artifact.results[0]
    assert any("gap_filled" in warning for warning in result.warnings)
    assert result.support == "weakly_supported"
    assert result.support_assessment["status"] == "conditionally_supported"
    # The interpolated value is the midpoint of its neighbours (111, 113).
    values = [item.value for item in _series_values(source, tmp_path)]
    assert values[12] == pytest.approx(112.0)


def _series_values(source: Path, tmp_path: Path):
    from gnomon.pipeline import load_stage
    loaded = load_stage(
        str(source), time_column="timestamp", target_column="value",
        series_column=None, frequency=None, repair="aggressive",
    )
    return loaded.groups["__default__"]


def test_aggressive_resolves_conflicts_last_wins(tmp_path: Path) -> None:
    rows = daily_rows(30)
    rows.insert(6, (rows[5][0], "999"))  # earlier conflicting value; later row wins
    rows[5], rows[6] = rows[6], rows[5]
    source = tmp_path / "conflict.csv"
    write_rows(source, rows)
    with pytest.raises(GnomonError):
        run(source, tmp_path)             # safe refuses to choose
    artifact, _ = run(source, tmp_path, repair="aggressive")
    values = [item.value for item in _series_values(source, tmp_path)]
    assert values[5] == 105.0             # the last row in file order
    assert any(a["code"] == "conflicting_duplicate_resolved"
               for a in repair_evidence(artifact)["actions"])


def test_safe_aligns_bounded_jitter_without_charging_invention_ceiling(
        tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, 0, 7)
    rows = []
    for index in range(36):
        stamp = start + timedelta(minutes=20 * index)
        stamp += timedelta(seconds=(-1, 1, 0)[index % 3])
        rows.append((stamp.isoformat(), str(100 + index)))
    source = tmp_path / "jitter.csv"
    write_rows(source, rows)
    with pytest.raises(GnomonError) as strict:
        run(source, tmp_path, repair="off")
    assert strict.value.code == "AMBIGUOUS_FREQUENCY"

    artifact, _ = run(source, tmp_path)
    actions = repair_evidence(artifact)["actions"]
    aligned = [a for a in actions if a["code"] == "timestamp_jitter_aligned"]
    assert aligned and aligned[0]["count"] == 24  # > the old 30% ceiling
    assert aligned[0]["metrics"] == {
        "cadence": "20min",
        "grid_phase": "2026-01-01T00:07:00",
        "maximum_displacement_seconds": 1.0,
        "tolerance_seconds": 12.0,
    }
    loaded = _series_values(source, tmp_path)
    assert [item.value for item in loaded] == [float(100 + index)
                                               for index in range(36)]
    assert [item.timestamp for item in loaded] == [
        start + timedelta(minutes=20 * index) for index in range(36)]
    assert any("timestamp_jitter_aligned" in warning
               for warning in artifact.results[0].warnings)


def test_bounded_alignment_refuses_collision_in_safe_and_aggressive(
        tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, 0, 7)
    rows = [
        ((start + timedelta(minutes=20 * index)).isoformat(), str(index))
        for index in range(30)
    ]
    rows.insert(11, ((start + timedelta(minutes=200, seconds=5)).isoformat(), "999"))
    source = tmp_path / "collision.csv"
    write_rows(source, rows)
    for level in ("safe", "aggressive"):
        with pytest.raises(GnomonError) as caught:
            run(source, tmp_path, repair=level, frequency="20min")
        assert caught.value.code == "TIMESTAMP_ALIGNMENT_CONFLICT"
        assert caught.value.to_dict()["error"]["repair_options"]


def test_jitter_outside_cadence_bound_remains_typed_refusal(
        tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, 0, 7)
    rows = []
    for index in range(30):
        stamp = start + timedelta(minutes=20 * index)
        if index == 12:
            stamp += timedelta(seconds=12, microseconds=1000)
        rows.append((stamp.isoformat(), str(index)))
    source = tmp_path / "outside.csv"
    write_rows(source, rows)
    for level in ("safe", "aggressive"):
        with pytest.raises(GnomonError) as caught:
            run(source, tmp_path, repair=level)
        assert caught.value.code == "AMBIGUOUS_FREQUENCY"


def test_reordered_jitter_is_disclosed_separately(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, 0, 7)
    rows = [
        ((start + timedelta(minutes=20 * index,
                            seconds=(-1, 1, 0)[index % 3])).isoformat(),
         str(index))
        for index in range(30)
    ]
    rows[8], rows[9] = rows[9], rows[8]
    source = tmp_path / "reordered-jitter.csv"
    write_rows(source, rows)
    artifact, _ = run(source, tmp_path)
    actions = repair_evidence(artifact)["actions"]
    assert {action["code"] for action in actions} >= {
        "timestamp_jitter_aligned", "timestamps_reordered"}


def test_aggressive_coerces_mixed_timezones(tmp_path: Path) -> None:
    rows = daily_rows(30)
    rows[3] = (rows[3][0] + "T00:00:00+00:00", rows[3][1])
    source = tmp_path / "mixed_tz.csv"
    write_rows(source, rows)
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path)
    assert caught.value.code == "MIXED_TIMEZONES"
    artifact, _ = run(source, tmp_path, repair="aggressive")
    assert any(a["code"] == "timezone_coerced" for a in repair_evidence(artifact)["actions"])


def test_excessive_repair_is_refused(tmp_path: Path) -> None:
    rows = daily_rows(40)
    # Remove every fourth row: a third of the series would be invented.
    rows = [row for index, row in enumerate(rows) if index % 4 != 1]
    source = tmp_path / "swiss_cheese.csv"
    write_rows(source, rows)
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path, repair="aggressive")
    assert caught.value.code == "EXCESSIVE_REPAIR"
    assert caught.value.to_dict()["error"]["repair_options"]


def test_invalid_repair_level_is_typed(tmp_path: Path) -> None:
    source = tmp_path / "clean.csv"
    write_rows(source, daily_rows(10))
    with pytest.raises(GnomonError) as caught:
        run(source, tmp_path, repair="yolo")
    assert caught.value.code == "INVALID_REPAIR_LEVEL"


def test_bounded_jitter_is_visible_through_inspect_mcp_forecast_and_cli(
        tmp_path: Path, capsys) -> None:
    from gnomon.cli import main
    from gnomon.toolspec import runner_for

    start = datetime(2026, 1, 1, 0, 7)
    rows = [
        ((start + timedelta(minutes=20 * index,
                            seconds=(-1, 1, 0)[index % 3])).isoformat(),
         str(100 + index))
        for index in range(36)
    ]
    source = tmp_path / "surface-jitter.csv"
    write_rows(source, rows)

    inspected = inspect_dataset(
        str(source), time_column="timestamp", target_column="value")
    assert inspected["data_quality"]["status"] == "repaired_safe"
    assert inspected["data_quality"]["repairs"][0]["metrics"][
        "tolerance_seconds"] == 12.0

    mcp_inspected = runner_for("gnomon_inspect")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value",
    })
    assert mcp_inspected["data_quality"]["status"] == "repaired_safe"

    mcp_forecast = runner_for("gnomon_forecast")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": 3,
        "output_dir": str(tmp_path / "mcp-output"),
    })
    assert mcp_forecast["status"] == "complete"
    assert any("timestamp_jitter_aligned" in warning
               for warning in mcp_forecast["results"][0]["warnings"])

    assert main([
        "inspect", str(source), "--time", "timestamp", "--target", "value",
    ]) == 0
    cli_inspected = json.loads(capsys.readouterr().out)
    assert cli_inspected["data_quality"]["status"] == "repaired_safe"

    assert main([
        "forecast", str(source), "--time", "timestamp", "--target", "value",
        "--horizon", "3", "--output", str(tmp_path / "cli-output"),
    ]) == 0
    cli_forecast = json.loads(capsys.readouterr().out)
    assert any("timestamp_jitter_aligned" in warning
               for warning in cli_forecast["results"][0]["warnings"])


# --- inspect: the guided last mile ------------------------------------------

def test_inspect_reports_data_quality_ladder(tmp_path: Path) -> None:
    clean = tmp_path / "clean.csv"
    write_rows(clean, daily_rows(20))
    assert inspect_dataset(str(clean), time_column="timestamp", target_column="value")[
        "data_quality"]["status"] == "clean"

    safe = tmp_path / "safe.csv"
    rows = daily_rows(20)
    rows[3] = (rows[3][0], "$103")
    write_rows(safe, rows)
    payload = inspect_dataset(str(safe), time_column="timestamp", target_column="value")
    assert payload["data_quality"]["status"] == "repaired_safe"
    assert "--repair aggressive" not in payload["suggested_next"]

    aggressive = tmp_path / "aggressive.csv"
    rows = daily_rows(20)
    del rows[8]
    write_rows(aggressive, rows)
    payload = inspect_dataset(str(aggressive), time_column="timestamp", target_column="value")
    assert payload["data_quality"]["status"] == "repaired_aggressive"
    assert "--repair aggressive" in payload["suggested_next"]
    assert any(a["code"] == "gap_filled" for a in payload["data_quality"]["repairs"])


# --- the bundled filthy example ----------------------------------------------

def test_filthy_example_end_to_end(tmp_path: Path) -> None:
    source = REPO / "examples" / "filthy_requests.csv"
    payload = inspect_dataset(str(source), time_column="timestamp", target_column="requests")
    assert payload["data_quality"]["status"] == "repaired_aggressive"
    artifact, artifact_dir = forecast(
        str(source), time_column="timestamp", target_column="requests",
        horizon=7, output=str(tmp_path), clock=CLOCK, repair="aggressive",
    )
    payload = repair_evidence(artifact)
    codes = {action["code"] for action in payload["actions"]}
    assert {"numeric_format_normalised", "timestamp_format_normalised",
            "missing_value_dropped", "duplicate_row_collapsed",
            "conflicting_duplicate_resolved", "gap_filled",
            "blank_row_skipped"} <= codes
    result = artifact.results[0]
    assert result.support in ("supported", "weakly_supported", "supported_ensemble")
    assert any("repaired_data" in warning for warning in result.warnings)
    written = json.loads((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
    assert any(item["kind"] == "data_repair" for item in written["evidence"])
