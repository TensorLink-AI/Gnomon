"""Calendar structure is not messiness.

The 2026-08 MCP evaluation had to forward-fill 64 years of Treasury
yields (16,135 trading days → 23,594 calendar days) and hand-restamp
monthly feeds before Gnomon would accept them — and the repair ladder
could never have done it, because weekends alone (~29% of a calendar
grid) sit at the MAX_ASSUMPTIVE_FRACTION ceiling that exists to stop
Gnomon inventing datasets. `regrid` is the structural answer: the caller
*declares* the calendar (business_daily, month_start), the fills and
restamps are disclosed as warnings and evidence, and the messiness
ceiling — which measures guessing, not calendars — is not charged.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        writer.writerows(rows)
    return path


def _business_days(path: Path, weeks: int) -> Path:
    rows = []
    day = date(2020, 1, 6)  # a Monday
    value = 4.0
    while len(rows) < weeks * 5:
        if day.weekday() < 5:
            value += 0.01
            rows.append((day.isoformat(), round(value, 3)))
        day += timedelta(days=1)
    return _write_csv(path, rows)


def _month_ends(path: Path, months: int) -> Path:
    import calendar
    rows = []
    year, month = 2015, 1
    for index in range(months):
        last = calendar.monthrange(year, month)[1]
        rows.append((date(year, month, last).isoformat(), 100.0 + index))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return _write_csv(path, rows)


def test_business_daily_regrid_forecasts_end_to_end(tmp_path: Path) -> None:
    """Mon-Fri data lands on the continuous daily grid with every fill
    disclosed — the workflow the evaluator had to hand-build upstream."""
    from gnomon.toolspec import runner_for

    source = _business_days(tmp_path / "yields.csv", weeks=30)
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 7, "regrid": "business_daily",
        "output_dir": str(tmp_path / "out"),
    })
    assert payload["status"] == "complete"
    warnings = payload["results"][0]["warnings"]
    assert any("regrid_business_daily" in warning for warning in warnings), warnings
    # 30 weeks of weekdays gain 29 weekends of fills.
    assert any("x58" in warning for warning in warnings), warnings


def test_business_daily_scale_beyond_the_repair_ceiling(tmp_path: Path) -> None:
    """The whole point: at multi-year scale the fills (~29% of the grid)
    exceed MAX_ASSUMPTIVE_FRACTION, so repair=aggressive could never do
    this — regrid must."""
    from gnomon.repair import MAX_ASSUMPTIVE_FRACTION
    from gnomon.toolspec import runner_for

    source = _business_days(tmp_path / "long.csv", weeks=520)  # ten years
    payload = runner_for("gnomon_inspect")({
        "input": str(source), "regrid": "business_daily",
    })
    assert payload["status"] == "valid"
    fills = 519 * 2
    grid = 520 * 5 + fills
    assert fills / grid > MAX_ASSUMPTIVE_FRACTION - 0.02  # near/above the cap


def test_month_start_regrid_accepts_month_end_stamps(tmp_path: Path) -> None:
    from gnomon.toolspec import runner_for

    source = _month_ends(tmp_path / "cpi.csv", months=48)
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 6, "regrid": "month_start",
        "output_dir": str(tmp_path / "out"),
    })
    assert payload["status"] == "complete"
    first = payload["results"][0]["forecast"][0]["timestamp"]
    assert first.endswith("-01T00:00:00"), first
    warnings = payload["results"][0]["warnings"]
    assert any("regrid_month_start" in warning for warning in warnings), warnings


def test_month_start_collision_refuses_loudly(tmp_path: Path) -> None:
    from gnomon.toolspec import runner_for

    source = _write_csv(tmp_path / "dup.csv", [
        ("2020-01-15", 1.0), ("2020-01-31", 2.0), ("2020-02-29", 3.0),
        ("2020-03-31", 4.0),
    ])
    with pytest.raises(GnomonError) as raised:
        runner_for("gnomon_forecast")({
            "input": str(source), "horizon": 2, "regrid": "month_start",
            "output_dir": str(tmp_path / "out"),
        })
    assert raised.value.code == "REGRID_CONFLICT"
    assert raised.value.details["series"]


def test_sparse_data_declared_business_daily_refuses(tmp_path: Path) -> None:
    """Weekly data declared business_daily would need ~86% of the grid
    invented; the declaration is implausible and the counts say so."""
    rows = [((date(2020, 1, 6) + timedelta(weeks=index)).isoformat(),
             float(index)) for index in range(30)]
    source = _write_csv(tmp_path / "weekly.csv", rows)
    from gnomon.toolspec import runner_for

    with pytest.raises(GnomonError) as raised:
        runner_for("gnomon_inspect")({
            "input": str(source), "regrid": "business_daily",
        })
    assert raised.value.code == "REGRID_IMPLAUSIBLE"
    assert raised.value.details["fill_fraction"] > 0.45


def test_regrid_conflicting_frequency_refuses(tmp_path: Path) -> None:
    source = _business_days(tmp_path / "b.csv", weeks=8)
    from gnomon.toolspec import runner_for

    with pytest.raises(GnomonError) as raised:
        runner_for("gnomon_inspect")({
            "input": str(source), "regrid": "business_daily",
            "frequency": "W",
        })
    assert raised.value.code == "INVALID_ARGUMENTS"
    assert "implies frequency D" in raised.value.message


def test_regrid_on_store_inputs_refuses(tmp_path: Path) -> None:
    from gnomon.pipeline import load_stage

    with pytest.raises(GnomonError) as raised:
        load_stage("store:whatever", time_column="t", target_column="v",
                   series_column=None, frequency=None,
                   store_path=str(tmp_path / "s.db"),
                   regrid="business_daily")
    assert raised.value.code == "INVALID_ARGUMENTS"
    assert "ingest" in raised.value.message


def test_regrid_fills_are_not_charged_against_the_repair_ceiling(
        tmp_path: Path) -> None:
    """A file that also needs aggressive repair keeps its own ceiling:
    the regrid fills must not push a modestly messy file into
    EXCESSIVE_REPAIR."""
    rows = []
    day = date(2021, 1, 4)
    value = 10.0
    count = 0
    while count < 200:
        if day.weekday() < 5:
            value += 0.05
            rows.append((day.isoformat(), round(value, 3)))
            count += 1
        day += timedelta(days=1)
    source = _write_csv(tmp_path / "mixed.csv", rows)
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 5, "regrid": "business_daily",
        "repair": "aggressive", "output_dir": str(tmp_path / "out"),
    })
    assert payload["status"] == "complete"


def test_macros_accept_regrid(tmp_path: Path) -> None:
    """The evaluator's business-day pain hit investigate too; every
    data-reading macro takes the declaration."""
    from gnomon.toolspec import runner_for

    source = _business_days(tmp_path / "macro.csv", weeks=30)
    investigation = runner_for("gnomon_investigate_change")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "regrid": "business_daily",
        "output_dir": str(tmp_path / "inv"),
    })
    assert investigation["status"] in ("complete", "ok")
    anomalies = runner_for("gnomon_detect_anomalies")({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "regrid": "business_daily",
        "output_dir": str(tmp_path / "anom"),
    })
    assert anomalies["status"] in ("complete", "ok")
