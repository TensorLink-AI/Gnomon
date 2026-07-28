from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from headwater.contracts import HeadwaterError
from headwater.runtime import forecast, inspect_dataset


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
        "artifact.json", "forecast.csv", "evidence.jsonl", "summary.md"
    }
    persisted = json.loads((directory / "artifact.json").read_text())
    assert persisted["task"]["schema"]["time_column"] == "timestamp"
    assert persisted["evidence"][0]["payload"]["partitioning"].startswith("selection")


def test_short_valid_series_abstains_without_error(tmp_path: Path) -> None:
    source = tmp_path / "short.csv"
    write_daily(source, 15)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value", horizon=3,
        output=str(tmp_path / "output"),
    )
    assert artifact.results[0].support == "unsupported"
    assert artifact.results[0].forecast == []
    assert "separated selection" in artifact.results[0].warnings[0]


def test_duplicate_timestamps_are_structured_input_error(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.csv"
    write_daily(source, 20, duplicate=True)
    with pytest.raises(HeadwaterError) as caught:
        inspect_dataset(str(source), time_column="timestamp", target_column="value", series_column="series")
    assert caught.value.code == "DUPLICATE_TIMESTAMPS"

