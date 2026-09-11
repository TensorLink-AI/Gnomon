"""The disclosed-repair layer: messy files parse under safe/aggressive
repair, every fix is logged, assumptive fixes are disclosed, and
excessive messiness is an honest refusal rather than an invented series."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.repair import (
    AmbiguousDateOrder,
    parse_number,
    parse_timestamp_lenient,
    scan_day_first,
    scan_numeric_evidence,
)

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



def inspect_rows(tmp_path, rows, **options):
    from gnomon import GnomonSession
    path = tmp_path / "data.csv"
    path.write_text("timestamp,value\n" + "\n".join(f"{stamp},{value}" for stamp, value in rows))
    with GnomonSession() as session:
        inspected = session.data.inspect(str(path), **options)
        request = session.data.request(inspected["data_ref"], horizon=1)
        return inspected, request


def daily_rows(count=30):
    return [((datetime(2026, 1, 1) + timedelta(days=i)).isoformat(), str(100+i)) for i in range(count)]


def test_strict_repair_rejects_and_the_safe_default_discloses_every_fix(tmp_path):
    # 1.2.0: safe is the default; strict rejection needs an explicit repair="off".
    rows = daily_rows()
    rows[5] = (rows[5][0], "$105")
    with pytest.raises(GnomonError) as caught:
        inspect_rows(tmp_path, rows, repair="off")
    assert caught.value.code == "INVALID_TARGET"
    inspected, request = inspect_rows(tmp_path, rows + [rows[10]])
    assert inspected["data_quality"]["status"] == "repaired_safe"
    assert request.history == tuple(float(100+i) for i in range(30))
    codes = {action["code"] for action in inspected["repairs"]}
    assert {"numeric_format_normalised", "duplicate_row_collapsed"} <= codes
    assert not any(action["assumptive"] for action in inspected["repairs"])


@pytest.mark.parametrize("level", ["typo", None, False, True, 0, 1, [], {}, "Safe", ""])
@pytest.mark.parametrize("entrypoint", ["file", "rows", "grid", "store", "session"])
def test_invalid_repair_policy_is_refused_before_io_or_mutation(tmp_path, level, entrypoint):
    from gnomon import GnomonSession
    from gnomon.data import load_observations, observations_from_rows
    from gnomon.datasets import load_stage
    from gnomon.repair import RepairLog, repair_observations
    path = tmp_path / "must-not-be-created"
    rows = [{"timestamp": "2026-01-01", "value": "$10"}]
    log = RepairLog()
    with pytest.raises(GnomonError) as caught:
        if entrypoint == "file":
            load_observations(str(path), "timestamp", "value", None, repair=level, repair_log=log)
        elif entrypoint == "rows":
            observations_from_rows(rows, ["timestamp", "value"], "timestamp", "value", None,
                                   repair=level, repair_log=log)
        elif entrypoint == "grid":
            repair_observations([], None, level, log)
        elif entrypoint == "store":
            load_stage("store:data", time_column="timestamp", target_column="value",
                       series_column=None, frequency=None, store_path=str(path), repair=level)
        else:
            with GnomonSession() as session:
                session.call("gnomon_inspect", {"input": str(path), "repair": level})
    assert caught.value.code == "INVALID_ARGUMENTS"
    assert "repair must be" in str(caught.value)
    assert rows == [{"timestamp": "2026-01-01", "value": "$10"}]
    assert log.actions() == [] and not path.exists()


def test_only_aggressive_repairs_fill_gaps_and_mark_assumptions(tmp_path):
    rows = daily_rows()
    del rows[12]
    with pytest.raises(GnomonError) as caught:
        inspect_rows(tmp_path, rows, repair="safe", frequency="D")
    assert caught.value.code == "IRREGULAR_TIME_GRID"
    inspected, request = inspect_rows(tmp_path, rows, repair="aggressive", frequency="D")
    assert request.history[12] == 112 and len(request.history) == 30
    filled = next(action for action in inspected["repairs"] if action["code"] == "gap_filled")
    assert filled["count"] == 1 and filled["assumptive"]


@pytest.mark.parametrize("policy", [False, True, 0, 1, "", [], {}, "typo"])
def test_invalid_regrid_policy_is_not_silently_ignored(tmp_path, policy):
    from gnomon import GnomonSession
    path = tmp_path / "must-not-be-created"
    with GnomonSession() as session, pytest.raises(GnomonError) as caught:
        session.call("gnomon_inspect", {"input": str(path), "regrid": policy})
    assert caught.value.code == "INVALID_ARGUMENTS"
    assert "regrid must be" in str(caught.value) and not path.exists()


def test_aggressive_duplicate_resolution_preserves_file_order(tmp_path):
    rows = daily_rows()
    rows.insert(5, (rows[5][0], "999"))
    with pytest.raises(GnomonError):
        inspect_rows(tmp_path, rows, repair="safe")
    inspected, request = inspect_rows(tmp_path, rows, repair="aggressive")
    assert request.history[5] == 105
    assert any(action["code"] == "conflicting_duplicate_resolved" for action in inspected["repairs"])


def test_safe_jitter_alignment_preserves_values_and_discloses_tolerance(tmp_path):
    start = datetime(2026, 1, 1, 0, 7)
    rows = [((start + timedelta(minutes=20*i, seconds=(-1, 1, 0)[i%3])).isoformat(), str(i)) for i in range(36)]
    with pytest.raises(GnomonError):
        inspect_rows(tmp_path, rows, repair="off")
    inspected, request = inspect_rows(tmp_path, rows, repair="safe")
    aligned = next(action for action in inspected["repairs"] if action["code"] == "timestamp_jitter_aligned")
    assert aligned["count"] == 24 and aligned["metrics"]["tolerance_seconds"] == 12
    assert request.history == tuple(range(36))
    assert request.timestamps == tuple((start + timedelta(minutes=20*i)).isoformat() for i in range(36))


@pytest.mark.parametrize("level", ["safe", "aggressive"])
def test_alignment_cannot_collapse_distinct_readings(tmp_path, level):
    start = datetime(2026, 1, 1, 0, 7)
    rows = [((start + timedelta(minutes=20*i)).isoformat(), str(i)) for i in range(30)]
    rows.insert(11, ((start + timedelta(minutes=200, seconds=5)).isoformat(), "999"))
    with pytest.raises(GnomonError) as caught:
        inspect_rows(tmp_path, rows, repair=level, frequency="20min")
    assert caught.value.code == "TIMESTAMP_ALIGNMENT_CONFLICT"


def test_excessive_invention_is_refused(tmp_path):
    rows = [row for i, row in enumerate(daily_rows(40)) if i % 4 != 1]
    with pytest.raises(GnomonError) as caught:
        inspect_rows(tmp_path, rows, repair="aggressive")
    assert caught.value.code == "EXCESSIVE_REPAIR"


def test_filthy_example_runs_with_disclosed_repairs(tmp_path):
    from gnomon import GnomonSession
    with GnomonSession.from_config() as session:
        inspected = session.call("gnomon_inspect", {"input": str(REPO / "examples/filthy_requests.csv"),
                                                   "target_column": "requests", "repair": "aggressive"})
        codes = {action["code"] for action in inspected["repairs"]}
        assert {"numeric_format_normalised", "timestamp_format_normalised", "gap_filled",
                "missing_value_dropped", "conflicting_duplicate_resolved"} <= codes
        result = session.call("gnomon_forecast", {"provider": "last_value",
            "data_ref": inspected["data_ref"], "horizon": 7})
        assert len(result["result"]["point"]) == 7
        assert result["evidence"] == "inference_only" and not result["action_authorized"]
