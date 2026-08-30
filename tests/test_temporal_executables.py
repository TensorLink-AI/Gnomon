import math
import random

from gnomon.temporal_executables import (
    _label, fit_dependence_executable, fit_future_seasonality_executable,
    fit_temporal_executable,
)


def _series(seed: int, *, slope: float = 0.0, season: int = 1,
            noise: float = .2) -> list[float]:
    rng = random.Random(seed)
    return [50 + slope * i + (4 * math.sin(2 * math.pi * i / season)
                              if season > 1 else 0) + rng.gauss(0, noise)
            for i in range(360)]


def test_all_univariate_properties_return_immutable_fitted_receipts() -> None:
    values = _series(7, slope=.05, season=24)
    for prop in ("level", "trend", "seasonality", "regime", "extreme"):
        first = fit_temporal_executable(
            values, property=prop, horizon=24, season=24).execute()
        second = fit_temporal_executable(
            values, property=prop, horizon=24, season=24).execute()
        assert first == second
        assert first["executable"]["property"] == prop
        assert first["diagnostics"]["primary_forecast_unchanged"] is True
        assert first["interval"]["lower"] <= first["estimate"] <= first["interval"]["upper"]


def test_selection_is_prefix_safe() -> None:
    values = _series(11, slope=.03)
    prefix = fit_temporal_executable(values[:300], property="trend", horizon=12)
    replay = fit_temporal_executable(
        (values[:300] + [10_000] * 20)[:300], property="trend", horizon=12)
    assert prefix == replay


def test_short_history_returns_best_estimate_without_inventing_support() -> None:
    fitted = fit_temporal_executable(
        [float(i) for i in range(20)], property="level", horizon=8)
    assert fitted.support == "weak"
    assert fitted.direction in {"lower", "similar", "higher"}
    assert fitted.execute()["automation_eligible"] is False


def test_direction_is_not_supported_when_interval_crosses_its_boundary() -> None:
    rng = random.Random(55606)
    values = [100 + rng.gauss(0, .6) for _ in range(420)]
    fitted = fit_temporal_executable(
        values, property="level", horizon=24)
    assert fitted.direction == "higher"
    assert fitted.lower < .25 < fitted.upper
    assert fitted.support == "weak"
    assert fitted.diagnostics["interval_direction_consistent"] is False


def test_seasonally_adjusted_trend_direction_matches_its_numeric_estimate() -> None:
    rng = random.Random(7301)
    values = [100 + .08 * index
              + 8 * math.sin(2 * math.pi * index / 24 + .4)
              + rng.gauss(0, .6) for index in range(240)]
    fitted = fit_temporal_executable(
        values, property="trend", horizon=48, season=24)
    assert fitted.direction == _label("trend", fitted.estimate)
    assert fitted.direction == "upward"
    assert fitted.support == "weak"
    assert fitted.diagnostics["folds"] == 2
    assert fitted.lower < fitted.estimate < fitted.upper
    assert fitted.diagnostics["calibration_direction_disagreement"] in {
        True, False}


def test_insufficient_seasonal_cycles_return_typed_trend_abstention() -> None:
    values = [100 + .08 * index
              + 8 * math.sin(2 * math.pi * index / 24)
              for index in range(30)]
    fitted = fit_temporal_executable(
        values, property="trend", horizon=12, season=24)
    answer = fitted.execute()
    assert answer["direction"] == "uncertain"
    assert answer["estimate"] is None
    assert answer["interval"] is None
    assert answer["support"] == "abstained"
    assert answer["automation_eligible"] is False
    assert fitted.diagnostics["reason"] == \
        "insufficient_cycles_for_seasonally_adjusted_trend"


def test_unadmitted_visible_seasonality_never_earns_trend_automation() -> None:
    rng = random.Random(7302)
    values = [100 + 8 * math.sin(2 * math.pi * index / 24 + .2)
              + rng.gauss(0, .4) for index in range(240)]
    fitted = fit_temporal_executable(
        values, property="trend", horizon=48, season=1)
    evidence = fitted.diagnostics["unmodelled_seasonality"]
    assert evidence["detected_period"] == 24
    assert evidence["period_was_not_silently_admitted"] is True
    assert fitted.support != "supported"
    assert fitted.execute()["automation_eligible"] is False


def test_no_observations_still_abstains() -> None:
    fitted = fit_temporal_executable([], property="level", horizon=8)
    assert fitted.support == "abstained"
    assert fitted.direction == "uncertain"


def test_dependence_uses_paired_differences_and_calibrates() -> None:
    rng = random.Random(31)
    left, right, x, y = [], [], 0.0, 0.0
    for _ in range(500):
        shock = rng.gauss(0, 1)
        x += shock
        y += .9 * shock + rng.gauss(0, .1)
        left.append(x)
        right.append(y)
    answer = fit_dependence_executable(left, right, horizon=16).execute()
    assert answer["estimate"] > .8
    assert answer["direction"] == "positive"
    assert answer["support"] == "supported"
    assert answer["diagnostics"]["primary_forecast_unchanged"] is True


def test_future_seasonality_distinguishes_fixed_from_repeatable_phase_shift() -> None:
    fixed = [math.sin(2 * math.pi * index / 12) for index in range(360)]
    shifting = []
    for block in range(15):
        shifting.extend(math.sin(2 * math.pi * (index + 3 * block) / 12)
                        for index in range(24))

    fixed_answer = fit_future_seasonality_executable(
        fixed, horizon=24, season=12).execute()
    shifting_answer = fit_future_seasonality_executable(
        shifting, horizon=24, season=12).execute()

    assert fixed_answer["direction"] == "fixed"
    assert shifting_answer["direction"] == "shifting"
    assert fixed_answer["diagnostics"]["point_forecast_used"] is False
    assert shifting_answer["diagnostics"]["folds"] >= 5


def test_future_seasonality_preserves_phase_for_nonperiod_window_width() -> None:
    fixed = [math.sin(2 * math.pi * index / 12) for index in range(600)]

    answer = fit_future_seasonality_executable(
        fixed, horizon=31, season=12).execute()

    assert answer["direction"] == "fixed"
    assert answer["support"] == "supported"
