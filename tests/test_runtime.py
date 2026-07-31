from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aion.contracts import AionError
from aion.runtime import capabilities, forecast, inspect_dataset


def write_daily(path: Path, count: int, *, duplicate: bool = False) -> None:
    start = datetime(2026, 1, 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "value", "series"])
        writer.writeheader()
        for index in range(count):
            timestamp = start + timedelta(days=index)
            writer.writerow({"timestamp": timestamp.isoformat(), "value": 10 + 2 * index, "series": "alpha"})
        if duplicate:
            writer.writerow({"timestamp": start.isoformat(), "value": 10, "series": "alpha"})


def write_noisy(path: Path, count: int, *, minutes: int = 60, slope: float = 0.5) -> None:
    rng = random.Random(3)
    start = datetime(2026, 1, 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "value"])
        writer.writeheader()
        for index in range(count):
            timestamp = start + timedelta(minutes=minutes * index)
            writer.writerow({
                "timestamp": timestamp.isoformat(),
                "value": round(100 + slope * index + rng.gauss(0, 1), 4),
            })


def test_inspect_resolves_schema(tmp_path: Path) -> None:
    source = tmp_path / "daily.csv"
    write_daily(source, 30)
    result = inspect_dataset(
        str(source), time_column="timestamp", target_column="value", series_column="series"
    )
    assert result["status"] == "valid"
    assert result["schema"]["frequency"] == "D"  # type: ignore[index]
    assert result["series"][0]["observations"] == 30  # type: ignore[index]


def test_forecast_selects_drift_and_writes_complete_artifact(tmp_path: Path) -> None:
    source = tmp_path / "daily.csv"
    write_daily(source, 60)
    artifact, directory = forecast(
        str(source), time_column="timestamp", target_column="value", series_column="series",
        horizon=3, output=str(tmp_path / "output"),
    )
    result = artifact.results[0]
    assert result.support == "supported"
    assert result.selected_model == "drift"
    assert result.forecast[0]["point"] == pytest.approx(130.0)
    assert set(path.name for path in directory.iterdir()) == {
        "artifact.json", "forecast.csv", "evidence.jsonl", "summary.md",
        "lineage.json",
    }
    persisted = json.loads((directory / "artifact.json").read_text())
    assert persisted["task"]["schema"]["time_column"] == "timestamp"
    assert persisted["evidence"][0]["payload"]["partitioning"].startswith("selection")


def test_short_valid_series_uses_degraded_forecast(tmp_path: Path) -> None:
    source = tmp_path / "short.csv"
    write_daily(source, 15)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=3,
        output=str(tmp_path / "output"),
    )
    assert artifact.results[0].support == "degraded"
    assert len(artifact.results[0].forecast) == 3
    assert "single trailing" in artifact.results[0].warnings[0]


def test_short_valid_series_can_strictly_abstain(tmp_path: Path) -> None:
    source = tmp_path / "short.csv"
    write_daily(source, 15)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=3,
        output=str(tmp_path / "output"), strict_abstention=True,
    )
    assert artifact.results[0].support == "unsupported"
    assert artifact.results[0].forecast == []


def test_short_series_warning_reports_have_and_required_counts(tmp_path: Path) -> None:
    source = tmp_path / "short.csv"
    write_noisy(source, 84)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=24,
        frequency="h", output=str(tmp_path / "output"),
    )
    warning = artifact.results[0].warnings[0]
    assert "96 observations" in warning
    assert "(have 84)" in warning


def test_two_fold_history_degrades_instead_of_refusing(tmp_path: Path) -> None:
    source = tmp_path / "degraded.csv"
    write_noisy(source, 100)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=24,
        frequency="h", output=str(tmp_path / "output"),
    )
    result = artifact.results[0]
    assert result.support == "degraded"
    assert len(result.forecast) == 24
    assert result.interval_coverage is None
    assert any("Limited evaluation" in warning for warning in result.warnings)


def test_intervals_widen_with_the_horizon(tmp_path: Path) -> None:
    source = tmp_path / "hourly.csv"
    write_noisy(source, 400)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=24,
        output=str(tmp_path / "output"),
    )
    rows = artifact.results[0].forecast
    first_width = rows[0]["q90"] - rows[0]["q10"]
    last_width = rows[-1]["q90"] - rows[-1]["q10"]
    assert first_width > 0
    assert last_width == pytest.approx(first_width * 24 ** 0.5, rel=1e-6)


def test_threshold_analysis_reports_crossings(tmp_path: Path) -> None:
    source = tmp_path / "hourly.csv"
    write_noisy(source, 400, slope=1.0)
    artifact, directory = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=24,
        output=str(tmp_path / "output"), threshold=100 + 1.0 * 405,
    )
    result = artifact.results[0]
    assert result.threshold is not None
    assert result.threshold["value"] == pytest.approx(505.0)
    assert len(result.threshold["probability_above"]) == 24
    assert result.threshold["first_timestamp_point_above"] is not None
    assert all(0.0 <= p <= 1.0 for p in result.threshold["probability_above"])
    assert "Threshold" in (directory / "summary.md").read_text()


def test_minute_frequency_is_inferred_and_forecastable(tmp_path: Path) -> None:
    source = tmp_path / "five_minute.csv"
    write_noisy(source, 1000, minutes=5)
    inspected = inspect_dataset(str(source), time_column="timestamp", target_column="value")
    assert inspected["schema"]["frequency"] == "5min"  # type: ignore[index]
    assert "aion forecast" in inspected["suggested_next"]  # type: ignore[operator]
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=12,
        output=str(tmp_path / "output"),
    )
    assert artifact.results[0].support in {"supported", "weakly_supported"}
    assert len(artifact.results[0].forecast) == 12


def test_summary_notes_flat_forecast_and_coverage(tmp_path: Path) -> None:
    source = tmp_path / "flat.csv"
    start = datetime(2026, 1, 1)
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "value"])
        writer.writeheader()
        for index in range(60):
            writer.writerow({"timestamp": (start + timedelta(days=index)).isoformat(), "value": 42})
    artifact, directory = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=3,
        output=str(tmp_path / "output"),
    )
    assert artifact.results[0].selected_model == "last_value"
    summary = (directory / "summary.md").read_text()
    assert "flat line at the last observed value" in summary
    assert "interval coverage" in summary


def test_capabilities_report_new_models_and_frequencies() -> None:
    payload = capabilities()
    assert {"min", "5min", "15min", "30min"} <= set(payload["frequencies"])  # type: ignore[arg-type]
    assert payload["frequency_descriptions"]["MS"] == "month start"  # type: ignore[index]
    statistical = payload["models"]["statistical"]  # type: ignore[index]
    assert {"ets", "theta", "linear_trend", "drift", "window_average"} <= set(statistical)
    assert payload["features"]["threshold_analysis"] is True  # type: ignore[index]


def test_duplicate_timestamps_are_structured_input_error(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.csv"
    write_daily(source, 20, duplicate=True)
    # The strict path still rejects duplicates outright.
    with pytest.raises(AionError) as caught:
        forecast(str(source), time_column="timestamp", target_column="value",
                 series_column="series", horizon=3, output=str(tmp_path / "out"),
                 repair="off")
    assert caught.value.code == "DUPLICATE_TIMESTAMPS"
    # inspect diagnoses instead: the exact duplicate collapses under safe repair.
    payload = inspect_dataset(str(source), time_column="timestamp", target_column="value", series_column="series")
    quality = payload["data_quality"]
    assert quality["status"] == "repaired_safe"  # type: ignore[index]
    assert any(action["code"] == "duplicate_row_collapsed" for action in quality["repairs"])  # type: ignore[index]
