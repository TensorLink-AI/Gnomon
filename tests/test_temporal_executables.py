import math
import random

from gnomon.temporal_executables import (
    fit_dependence_executable, fit_temporal_executable,
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
