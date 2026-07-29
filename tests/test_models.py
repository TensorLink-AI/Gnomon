from __future__ import annotations

import math
import random

import pytest

from aion.models import MODELS, ets, linear_trend, theta, window_average


def trend_series(count: int, slope: float = 2.0, noise: float = 0.0) -> list[float]:
    rng = random.Random(11)
    return [100 + slope * index + rng.gauss(0, noise) for index in range(count)]


def test_linear_trend_recovers_slope_from_all_points() -> None:
    history = trend_series(80, slope=3.0, noise=0.5)
    forecast = linear_trend(history, 5, 7)
    assert len(forecast) == 5
    assert forecast[1] - forecast[0] == pytest.approx(3.0, abs=0.2)


def test_linear_trend_beats_two_point_drift_on_curved_data() -> None:
    # A level shift at the start fools first-to-last drift but not OLS.
    history = [500.0] + trend_series(60, slope=1.0)
    forecast = linear_trend(history, 1, 7)
    assert forecast[0] == pytest.approx(100 + 61, abs=30)


def test_window_average_is_flat_mean_of_recent_window() -> None:
    history = [1.0] * 50 + [3.0] * 7
    forecast = window_average(history, 4, 7)
    assert forecast == [pytest.approx(3.0)] * 4


def test_theta_extrapolates_half_the_trend() -> None:
    history = trend_series(60, slope=2.0)
    forecast = theta(history, 10, 7)
    assert len(forecast) == 10
    assert forecast[-1] > forecast[0]
    assert forecast[1] - forecast[0] == pytest.approx(1.0, abs=0.1)


def test_ets_captures_trend_and_seasonality() -> None:
    season = 12
    history = [
        50 + 0.5 * index + 10 * math.sin(2 * math.pi * index / season)
        for index in range(6 * season)
    ]
    forecast = ets(history, season, season)
    actual = [
        50 + 0.5 * index + 10 * math.sin(2 * math.pi * index / season)
        for index in range(6 * season, 7 * season)
    ]
    error = sum(abs(a - f) for a, f in zip(actual, forecast)) / season
    assert error < 2.0


def test_ets_requires_minimal_history() -> None:
    with pytest.raises(ValueError):
        ets([1.0, 2.0], 4, 7)


def test_every_model_returns_horizon_length_finite_values() -> None:
    history = trend_series(120, slope=1.0, noise=1.0)
    for name, model in MODELS.items():
        forecast = model(history, 6, 7)
        assert len(forecast) == 6, name
        assert all(math.isfinite(value) for value in forecast), name
