"""Perfectly regular data must never be called ambiguous.

The CiK abstention analysis found 45 task-seeds — solar irradiance at a
10-minute step, sensor pressure/speed at a 1-second step — refused with
AMBIGUOUS_FREQUENCY although every series had exactly one unique spacing.
The refusal came from a missing grid entry, not from any property of the
data. The named grid was first widened, then the mechanism generalised:
any strictly regular whole-second sub-daily step is representable as
``<N>s``/``<N>min``/``<N>h``, with the default season derived from the
same natural-cycle rule the curated table encodes. AMBIGUOUS_FREQUENCY
now means what it says — spacing that actually varies — and its details
say which case was hit.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gnomon.contracts import GnomonError
from gnomon.temporal import (
    FREQUENCIES,
    FREQUENCY_DESCRIPTIONS,
    SEASONS,
    canonical_code,
    default_season,
    frequency_step,
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


@pytest.mark.parametrize(
    ("step", "code"),
    [(timedelta(seconds=10), "10s"), (timedelta(minutes=7), "7min"),
     (timedelta(seconds=90), "90s"), (timedelta(hours=2), "2h"),
     (timedelta(minutes=20), "20min"), (timedelta(hours=12), "12h")],
)
def test_any_regular_sub_daily_step_is_inferred(step: timedelta, code: str) -> None:
    assert infer_frequency(_stamps(step, 48)) == code


def test_regular_but_unrepresentable_step_refuses_and_says_so() -> None:
    """Generalising must not shade into guessing: a two-day step is
    calendar-ambiguous (48 fixed hours vs every second calendar day), so it
    still fails — but the error now says the series was regular."""
    with pytest.raises(GnomonError) as raised:
        infer_frequency(_stamps(timedelta(days=2), 48))
    assert raised.value.code == "AMBIGUOUS_FREQUENCY"
    assert raised.value.details["regular"] is True
    assert "regular" in raised.value.message


def test_sub_second_step_refuses() -> None:
    with pytest.raises(GnomonError) as raised:
        infer_frequency(_stamps(timedelta(milliseconds=500), 48))
    assert raised.value.code == "AMBIGUOUS_FREQUENCY"
    assert raised.value.details["regular"] is True


def test_genuinely_mixed_spacing_refuses_and_says_so() -> None:
    stamps = _stamps(timedelta(minutes=7), 20)
    jittered = [stamp + timedelta(seconds=index % 3) for index, stamp in enumerate(stamps)]
    with pytest.raises(GnomonError) as raised:
        infer_frequency(jittered)
    assert raised.value.code == "AMBIGUOUS_FREQUENCY"
    assert raised.value.details["regular"] is False
    assert raised.value.details["distinct_spacings"] > 1


def test_gaps_at_an_unusual_step_still_refuse() -> None:
    """A named step tolerates gaps (the grid validator and repair own
    those); an unusual step with gaps is indistinguishable from a heavier
    grid with jitter, so inference refuses rather than guesses."""
    stamps = _stamps(timedelta(minutes=7), 30)
    del stamps[11]
    with pytest.raises(GnomonError) as raised:
        infer_frequency(stamps)
    assert raised.value.code == "AMBIGUOUS_FREQUENCY"


def _month_starts(count: int, skip: set[int] = frozenset()) -> list[datetime]:
    stamps = []
    year, month = 2010, 1
    for index in range(count):
        if index not in skip:
            stamps.append(datetime(year, month, 1))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return stamps


def test_month_start_with_a_missing_month_infers_ms() -> None:
    """Real monthly feeds drop the odd month; a single hole must route to
    the grid validator (which names the gap, with repair options) rather
    than demote the series to AMBIGUOUS_FREQUENCY."""
    assert infer_frequency(_month_starts(48, skip={11})) == "MS"


def test_month_start_with_several_missing_months_infers_ms() -> None:
    assert infer_frequency(_month_starts(60, skip={5, 20, 21, 40})) == "MS"


def test_yearly_january_data_is_not_mistaken_for_monthly() -> None:
    stamps = [datetime(2000 + index, 1, 1) for index in range(20)]
    with pytest.raises(GnomonError) as raised:
        infer_frequency(stamps)
    assert raised.value.code == "AMBIGUOUS_FREQUENCY"


def test_quarterly_data_is_not_mistaken_for_monthly() -> None:
    stamps = [datetime(2010 + index // 4, 3 * (index % 4) + 1, 1)
              for index in range(24)]
    with pytest.raises(GnomonError) as raised:
        infer_frequency(stamps)
    assert raised.value.code == "AMBIGUOUS_FREQUENCY"


def test_first_of_month_at_varying_times_is_not_a_month_grid() -> None:
    stamps = _month_starts(24, skip={7})
    stamps = [stamp + timedelta(hours=index % 3)
              for index, stamp in enumerate(stamps)]
    with pytest.raises(GnomonError) as raised:
        infer_frequency(stamps)
    assert raised.value.code == "AMBIGUOUS_FREQUENCY"


def _daily_observations(days: list[datetime]) -> list:
    from gnomon.data import Observation

    return [Observation(timestamp=stamp, value=float(index), series="yield")
            for index, stamp in enumerate(days)]


def test_business_day_gap_is_named_in_the_grid_refusal() -> None:
    """Business-day data (market series) infers a daily grid and then hits
    the weekend hole; the refusal must say the skipped days are weekends so
    the caller learns the shape of the data, not just its first gap."""
    from gnomon.temporal import validate_and_group

    weekdays = [stamp for stamp in _stamps(timedelta(days=1), 21)
                if stamp.weekday() < 5]
    with pytest.raises(GnomonError) as raised:
        validate_and_group(_daily_observations(weekdays), None)
    assert raised.value.code == "IRREGULAR_TIME_GRID"
    assert raised.value.details["gap_weekend_only"] is True
    assert "business-day" in raised.value.message


def test_midweek_gap_is_not_called_a_weekend() -> None:
    from gnomon.temporal import validate_and_group

    stamps = _stamps(timedelta(days=1), 14)
    removed = next(stamp for stamp in stamps if stamp.weekday() == 2)
    stamps.remove(removed)
    with pytest.raises(GnomonError) as raised:
        validate_and_group(_daily_observations(stamps), "D")
    assert raised.value.code == "IRREGULAR_TIME_GRID"
    assert "gap_weekend_only" not in raised.value.details
    assert "business-day" not in raised.value.message


@pytest.mark.parametrize(
    ("alias", "code"),
    [("S", "s"), ("1s", "s"), ("sec", "s"), ("second", "s"),
     ("10T", "10min"), ("10m", "10min"), ("10min", "10min"),
     # General codes canonicalise to one spelling per duration: named codes
     # win, then the largest unit that divides the step evenly.
     ("60s", "min"), ("600s", "10min"), ("120min", "2h"), ("3600s", "h"),
     ("90s", "90s"), ("7min", "7min"), ("45T", "45min"), ("24h", "D")],
)
def test_new_codes_and_their_aliases_normalise(alias: str, code: str) -> None:
    assert normalise_frequency(alias) == code


@pytest.mark.parametrize("bad", ["0s", "0min", "48h", "-5min", "1.5min", "10ms"])
def test_unrepresentable_requested_codes_refuse(bad: str) -> None:
    with pytest.raises(GnomonError) as raised:
        normalise_frequency(bad)
    assert raised.value.code == "UNSUPPORTED_FREQUENCY"


def test_frequency_error_lists_general_subdaily_patterns() -> None:
    with pytest.raises(GnomonError) as raised:
        normalise_frequency("quarter-hour-ish")
    supported = raised.value.details["supported"]
    assert {"<N>s", "<N>min", "<N>h"} <= set(supported)


def test_intraday_ambiguity_does_not_offer_month_end_restamping() -> None:
    start = datetime(2026, 1, 1)
    stamps = [
        start,
        start + timedelta(minutes=20),
        start + timedelta(minutes=41),
        start + timedelta(minutes=61),
    ]
    with pytest.raises(GnomonError) as raised:
        infer_frequency(stamps)
    actions = {item["action"] for item in raised.value.repair_options or []}
    assert "restamp_to_month_start" not in actions
    assert raised.value.details["observed_step"] == "0:20:00"


def test_every_frequency_has_a_season_and_a_description() -> None:
    expected = set(FREQUENCIES) | {"MS"}
    assert set(SEASONS) == expected
    assert set(FREQUENCY_DESCRIPTIONS) == expected


def test_default_seasons_follow_the_curated_rule() -> None:
    """One rule reproduces the whole curated table (daily cycle when it
    fits in 288 lags, else hourly, else a minute cycle) and extends it to
    general codes."""
    for code in FREQUENCIES:
        assert default_season(code) == SEASONS[code]
    assert default_season("2h") == 12       # daily cycle
    assert default_season("20min") == 72    # daily cycle
    assert default_season("7min") == 206    # ~daily cycle, rounded
    assert default_season("2min") == 30     # hourly: daily would be 720 lags
    assert default_season("10s") == 6       # minute: hourly would be 360 lags
    assert default_season("12h") == 2


def test_frequency_step_round_trips_canonical_codes() -> None:
    for code, duration in FREQUENCIES.items():
        assert frequency_step(code) == duration
        assert canonical_code(duration) == code
    assert frequency_step("MS") is None
    assert frequency_step("90s") == timedelta(seconds=90)
    assert canonical_code(timedelta(seconds=90)) == "90s"


def test_schema_patterns_carry_the_whole_grid() -> None:
    """The tool schemas advertise the grid to agents; a code their pattern
    rejects is a code agents can never pass."""
    from gnomon.registry import _COMMON_INPUT
    from gnomon.toolspec import _INPUT_PROPERTIES

    for properties in (_COMMON_INPUT, _INPUT_PROPERTIES):
        pattern = re.compile(properties["frequency"]["pattern"])
        for code in SEASONS:
            assert pattern.match(code), code
        for code in ("7min", "90s", "2h", "10s"):
            assert pattern.match(code), code
        for junk in ("0s", "monthly-ish", "1.5min", "48hh"):
            assert not pattern.match(junk), junk


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


def test_general_step_data_forecasts_end_to_end(tmp_path: Path) -> None:
    """A 7-minute grid — in no one's named list — loads, infers, validates,
    and forecasts, with the future timestamps on the same 7-minute grid."""
    from gnomon.runtime import forecast

    source = tmp_path / "telemetry.csv"
    _write_regular(source, timedelta(minutes=7), 60)
    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=4, output=str(tmp_path / "output"),
    )
    assert artifact.task.schema.frequency == "7min"
    rows = artifact.results[0].forecast
    assert len(rows) == 4
    stamps = [datetime.fromisoformat(str(row["timestamp"])) for row in rows]
    deltas = {right - left for left, right in zip(stamps, stamps[1:])}
    assert deltas == {timedelta(minutes=7)}


def test_detect_season_finds_a_period_beyond_the_frequency_default() -> None:
    """An hourly axis defaults to season 24, but a 50-step oscillation is
    real data (AnomLLM's synthetic sines run at period 33-50). The search
    must reach past twice the calendar default, or every season-aware
    consumer — per-phase z-scores, rolling windows, seasonal naive —
    runs on the wrong period; measured as mass false-positive anomaly
    flags on sine series."""
    import math

    from gnomon.temporal import detect_season

    values = [math.sin(2 * math.pi * 0.02 * index) for index in range(400)]
    season, strength, basis = detect_season(values, "h")
    assert basis == "autocorrelation"
    assert season == 50
    assert strength > 0.5


def test_detect_season_considers_period_at_two_cycle_boundary() -> None:
    """Exactly two cycles still provide a measured, correctly typed period."""
    import math

    from gnomon.temporal import detect_season

    for phase in (0.0, 0.2, 1.1):
        values = [
            100 + 5 * math.sin(2 * math.pi * index / 24 + phase)
            for index in range(48)
        ]
        season, strength, basis = detect_season(values, "h")
        assert (season, basis) == (24, "autocorrelation")
        assert strength > .3


def test_detect_season_does_not_round_hourly_period_down_to_23() -> None:
    """Unequal overlap lengths must not make 23 beat a perfect 24-hour cycle."""
    import math

    from gnomon.temporal import detect_season

    values = [math.sin(2 * math.pi * index / 24) for index in range(50)]
    season, strength, basis = detect_season(values, "h")
    assert (season, basis) == (24, "autocorrelation")
    assert strength > .75


def test_detect_season_measures_non_sinusoidal_two_cycle_repeat() -> None:
    """A clean repeated daily shape is evidence, not a zero-strength default."""
    from gnomon.fingerprint import series_fingerprint
    from gnomon.temporal import detect_season

    values = [float(index % 24) for index in range(48)]
    season, strength, basis = detect_season(values, "h")
    assert (season, basis) == (24, "autocorrelation")
    assert strength > .75
    fingerprint = series_fingerprint(values, "h")
    assert fingerprint["season_period"] == 24
    assert fingerprint["season_source"] == "autocorrelation"
    assert fingerprint["season_strength"] > .75


def test_frequency_prior_does_not_replace_a_measured_period_23() -> None:
    import math

    from gnomon.temporal import detect_season

    values = [math.sin(2 * math.pi * index / 23) for index in range(69)]
    season, strength, basis = detect_season(values, "h")
    assert (season, basis) == (23, "autocorrelation")
    assert strength > .3


def test_export_rounding_dust_does_not_become_a_short_season() -> None:
    """A rounded linear cron counter has trend, not a nine-step cycle."""
    from gnomon.temporal import detect_season

    values = [round(1060 + 325 * index / 74, 6) for index in range(75)]
    assert detect_season(values, "20min") == (
        72, 0.0, "frequency_default")


def test_small_real_cycle_survives_relative_residual_energy_floor() -> None:
    import math

    from gnomon.temporal import detect_season

    values = [
        1060 + 325 * index / 74 + .01 * math.sin(2 * math.pi * index / 9)
        for index in range(75)
    ]
    season, strength, basis = detect_season(values, "20min")
    assert (season, basis) == (9, "autocorrelation")
    assert strength > .5
