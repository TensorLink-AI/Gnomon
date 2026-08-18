from __future__ import annotations

from statistics import mean
from typing import Callable


def last_value(history: list[float], horizon: int, season: int) -> list[float]:
    return [history[-1]] * horizon


def seasonal_naive(history: list[float], horizon: int, season: int) -> list[float]:
    if len(history) < season:
        raise ValueError("insufficient seasonal history")
    return [history[-season + (index % season)] for index in range(horizon)]


def drift(history: list[float], horizon: int, season: int) -> list[float]:
    if len(history) < 2:
        raise ValueError("insufficient history")
    slope = (history[-1] - history[0]) / (len(history) - 1)
    return [history[-1] + slope * step for step in range(1, horizon + 1)]


def _ols_line(history: list[float]) -> tuple[float, float]:
    count = len(history)
    if count < 2:
        raise ValueError("insufficient history")
    x_mean = (count - 1) / 2
    y_mean = mean(history)
    denominator = sum((index - x_mean) ** 2 for index in range(count))
    slope = sum(
        (index - x_mean) * (value - y_mean) for index, value in enumerate(history)
    ) / denominator
    return y_mean - slope * x_mean, slope


def linear_trend(history: list[float], horizon: int, season: int) -> list[float]:
    intercept, slope = _ols_line(history)
    origin = len(history) - 1
    return [intercept + slope * (origin + step) for step in range(1, horizon + 1)]


def window_average(history: list[float], horizon: int, season: int) -> list[float]:
    window = history[-min(len(history), max(2, season)):]
    return [mean(window)] * horizon


def theta(history: list[float], horizon: int, season: int) -> list[float]:
    """Classic Theta(0, 2): simple exponential smoothing plus half the OLS slope."""
    _, slope = _ols_line(history)
    best_level, best_sse = history[0], float("inf")
    for alpha in (0.1, 0.3, 0.5, 0.7, 0.9):
        level, sse = history[0], 0.0
        for value in history[1:]:
            sse += (value - level) ** 2
            level = alpha * value + (1 - alpha) * level
        if sse < best_sse:
            best_level, best_sse = level, sse
    return [best_level + (slope / 2) * step for step in range(1, horizon + 1)]


def _holt_winters(
    history: list[float], horizon: int, season: int,
    alpha: float, beta: float, gamma: float, seasonal: bool,
) -> tuple[list[float], float]:
    """One additive ETS fit; returns the forecast and the one-step-ahead SSE."""
    if seasonal:
        first_cycle, second_cycle = history[:season], history[season : 2 * season]
        level = mean(first_cycle)
        trend = (mean(second_cycle) - level) / season
        seasonals = [value - level for value in first_cycle]
        start = season
    else:
        level, trend = history[0], history[1] - history[0]
        seasonals = [0.0]
        start = 2
    sse = 0.0
    for index in range(start, len(history)):
        seasonal_index = index % len(seasonals)
        prediction = level + trend + seasonals[seasonal_index]
        error = history[index] - prediction
        sse += error * error
        previous_level = level
        level = alpha * (history[index] - seasonals[seasonal_index]) + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
        if seasonal:
            seasonals[seasonal_index] = (
                gamma * (history[index] - level) + (1 - gamma) * seasonals[seasonal_index]
            )
    origin = len(history)
    forecast = [
        level + trend * step + seasonals[(origin + step - 1) % len(seasonals)]
        for step in range(1, horizon + 1)
    ]
    return forecast, sse


def ets(history: list[float], horizon: int, season: int) -> list[float]:
    """Additive Holt-Winters (or Holt's linear when seasonal history is short),
    with smoothing parameters chosen by one-step-ahead error on the history."""
    if len(history) < 4:
        raise ValueError("insufficient history")
    seasonal = season > 1 and len(history) >= 2 * season + 4
    best_forecast, best_sse = None, float("inf")
    for alpha in (0.2, 0.5, 0.8):
        for beta in (0.05, 0.2):
            for gamma in (0.05, 0.3) if seasonal else (0.0,):
                forecast, sse = _holt_winters(
                    history, horizon, season, alpha, beta, gamma, seasonal
                )
                if sse < best_sse:
                    best_forecast, best_sse = forecast, sse
    assert best_forecast is not None
    return best_forecast


MODELS: dict[str, Callable[[list[float], int, int], list[float]]] = {
    "last_value": last_value,
    "seasonal_naive": seasonal_naive,
    "drift": drift,
    "linear_trend": linear_trend,
    "window_average": window_average,
    "theta": theta,
    "ets": ets,
}
BASELINES = {"last_value", "seasonal_naive"}


def predict(name: str, history: list[float], horizon: int, season: int) -> list[float]:
    return MODELS[name](history, horizon, season)
