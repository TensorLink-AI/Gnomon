"""Perfectly regular data must never be called ambiguous.

The CiK abstention analysis found 45 task-seeds — solar irradiance at a
10-minute step, sensor pressure/speed at a 1-second step — refused with
AMBIGUOUS_FREQUENCY although every series had exactly one unique spacing.
The refusal came from a missing grid entry, not from any property of the
data. These tests pin the widened grid and keep the schema enums that
mirror it from drifting.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.temporal import (
    FREQUENCIES,
    FREQUENCY_DESCRIPTIONS,
    SEASONS,
    infer_frequency,
    normalise_frequency,
)


def _stamps(step: timedelta, count: int) -> list[datetime]:
    start = datetime(2026, 1, 1, 5, 14, 3)
    return [start + step * index for index in range(count)]


def test_ten_minute_spacing_is_inferred_not_refused() -> None:
    assert infer_frequency(_stamps(timedelta(minutes=10), 48)) == "10min"


def test_one_second_spacing_is_inferred_not_refused() -> None:
    assert infer_frequency(_stamps(timedelta(seconds=1), 48)) == "s"


def test_unsupported_regular_spacing_still_refuses() -> None:
    """Widening the grid must not turn inference into guessing: a step the
    grid does not carry (10 seconds) still fails loudly."""
    with pytest.raises(GnomonError) as raised:
        infer_frequency(_stamps(timedelta(seconds=10), 48))
    assert raised.value.code == "AMBIGUOUS_FREQUENCY"


@pytest.mark.parametrize(
    ("alias", "code"),
    [("S", "s"), ("1s", "s"), ("sec", "s"), ("second", "s"),
     ("10T", "10min"), ("10m", "10min"), ("10min", "10min")],
)
def test_new_codes_and_their_aliases_normalise(alias: str, code: str) -> None:
    assert normalise_frequency(alias) == code


def test_every_frequency_has_a_season_and_a_description() -> None:
    expected = set(FREQUENCIES) | {"MS"}
    assert set(SEASONS) == expected
    assert set(FREQUENCY_DESCRIPTIONS) == expected


def test_schema_enums_carry_the_whole_grid() -> None:
    """The tool schemas advertise the grid to agents; a code missing there
    is a code agents will never pass."""
    from gnomon.registry import _COMMON_INPUT
    from gnomon.toolspec import _INPUT_PROPERTIES

    assert set(_COMMON_INPUT["frequency"]["enum"]) == set(SEASONS)
    assert set(_INPUT_PROPERTIES["frequency"]["enum"]) == set(SEASONS)


def _write_regular(path: Path, step: timedelta, count: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "value"])
        writer.writeheader()
        for index, stamp in enumerate(_stamps(step, count)):
            writer.writerow({"timestamp": stamp.isoformat(),
                             "value": 100 + 0.5 * index})


def test_ten_minute_data_forecasts_end_to_end(tmp_path: Path) -> None:
    from gnomon.runtime import forecast

    source = tmp_path / "solar.csv"
    _write_regular(source, timedelta(minutes=10), 60)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=6, output=str(tmp_path / "output"),
    )
    assert artifact.task.schema.frequency == "10min"
    assert len(artifact.results[0].forecast) == 6


def test_one_second_data_forecasts_end_to_end(tmp_path: Path) -> None:
    from gnomon.runtime import forecast

    source = tmp_path / "pressure.csv"
    _write_regular(source, timedelta(seconds=1), 60)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=6, output=str(tmp_path / "output"),
    )
    assert artifact.task.schema.frequency == "s"
    assert len(artifact.results[0].forecast) == 6
