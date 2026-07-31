"""The disclosed-repair layer: messy files parse under safe/aggressive
repair, every fix is logged, assumptive fixes downgrade support, and
excessive messiness is an honest refusal rather than an invented series."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aion.contracts import AionError
from aion.ids import FixedClock
from aion.repair import (
    AmbiguousDateOrder,
    parse_number,
    parse_timestamp_lenient,
    scan_day_first,
    scan_numeric_evidence,
)
from aion.runtime import forecast, inspect_dataset

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
    with pytest.raises(AionError) as caught:
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
    with pytest.raises(AionError) as caught:
        run(source, tmp_path)
    assert caught.value.code == "IRREGULAR_TIME_GRID"


def test_off_preserves_strict_errors(tmp_path: Path) -> None:
    rows = daily_rows(30)
    rows[5] = (rows[5][0], "$105")
    source = tmp_path / "messy.csv"
    write_rows(source, rows)
    with pytest.raises(AionError) as caught:
        run(source, tmp_path, repair="off")
    assert caught.value.code == "INVALID_TARGET"


def test_safe_ambiguous_date_order_is_an_error(tmp_path: Path) -> None:
    rows = [(f"0{1 + index}/03/2026", str(100 + index)) for index in range(9)]
    source = tmp_path / "ambiguous.csv"
    write_rows(source, rows)
    with pytest.raises(AionError) as caught:
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
    from aion.pipeline import load_stage
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
    with pytest.raises(AionError):
        run(source, tmp_path)             # safe refuses to choose
    artifact, _ = run(source, tmp_path, repair="aggressive")
    values = [item.value for item in _series_values(source, tmp_path)]
    assert values[5] == 105.0             # the last row in file order
    assert any(a["code"] == "conflicting_duplicate_resolved"
               for a in repair_evidence(artifact)["actions"])


def test_aggressive_snaps_jittered_timestamps(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1)
    rows = []
    for index in range(30):
        stamp = start + timedelta(days=index)
        if index in (4, 9, 17):
            stamp += timedelta(seconds=7 + index)
        rows.append((stamp.isoformat(), str(100 + index)))
    source = tmp_path / "jitter.csv"
    write_rows(source, rows)
    with pytest.raises(AionError):
        run(source, tmp_path)
    artifact, _ = run(source, tmp_path, repair="aggressive")
    snapped = [a for a in repair_evidence(artifact)["actions"] if a["code"] == "timestamp_snapped"]
    assert snapped and snapped[0]["count"] == 3


def test_aggressive_coerces_mixed_timezones(tmp_path: Path) -> None:
    rows = daily_rows(30)
    rows[3] = (rows[3][0] + "T00:00:00+00:00", rows[3][1])
    source = tmp_path / "mixed_tz.csv"
    write_rows(source, rows)
    with pytest.raises(AionError) as caught:
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
    with pytest.raises(AionError) as caught:
        run(source, tmp_path, repair="aggressive")
    assert caught.value.code == "EXCESSIVE_REPAIR"
    assert caught.value.to_dict()["error"]["repair_options"]


def test_invalid_repair_level_is_typed(tmp_path: Path) -> None:
    source = tmp_path / "clean.csv"
    write_rows(source, daily_rows(10))
    with pytest.raises(AionError) as caught:
        run(source, tmp_path, repair="yolo")
    assert caught.value.code == "INVALID_REPAIR_LEVEL"


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
