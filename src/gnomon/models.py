from __future__ import annotations

from statistics import mean
from math import exp, log
from typing import Callable


def last_value(history: list[float], horizon: int, season: int) -> list[float]:
    return [history[-1]] * horizon


def seasonal_naive(history: list[float], horizon: int, season: int) -> list[float]:
    if len(history) < season:
        raise ValueError("insufficient seasonal history")
    return [history[-season + (index % season)] for index in range(horizon)]


def historical_mean(history: list[float], horizon: int, season: int) -> list[float]:
    """Repeat the expanding-prefix level as an assumption-light baseline."""
    if not history:
        raise ValueError("insufficient history")
    return [mean(history)] * horizon


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


def _theta_raw(history: list[float], horizon: int, season: int) -> list[float]:
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


def _nested_transform_choice(
    history: list[float], season: int,
    raw: Callable[[list[float], int, int], list[float]],
) -> bool:
    """Whether log scale wins a training-internal trailing validation.

    This chooses one representative inside a model family. The chosen family
    member alone enters Gnomon's outer rolling contest, avoiding the candidate
    multiplicity that would result from publishing raw/log variants as
    separate models.
    """
    if len(history) < 12 or any(value <= 0 for value in history):
        return False
    # A log transform is a model of multiplicative scale. Do not select it
    # merely because it happens to shave validation loss from an additive or
    # linear series: require changes to grow with the series level. This is a
    # training-only structural diagnostic, not a dataset or benchmark rule.
    levels = history[:-1]
    changes = [abs(right - left) for left, right in zip(history, history[1:])]
    level_mean, change_mean = mean(levels), mean(changes)
    covariance = sum((level - level_mean) * (change - change_mean)
                     for level, change in zip(levels, changes))
    level_ss = sum((level - level_mean) ** 2 for level in levels)
    change_ss = sum((change - change_mean) ** 2 for change in changes)
    if level_ss <= 1e-12 or change_ss <= 1e-12:
        return False
    correlation = covariance / (level_ss * change_ss) ** .5
    if correlation < .35:
        return False
    holdout = max(3, min(len(history) // 5, 12))
    train, actual = history[:-holdout], history[-holdout:]
    try:
        raw_points = raw(train, holdout, season)
        log_points = [exp(value) for value in raw(
            [log(value) for value in train], holdout, season)]
    except (ValueError, ArithmeticError, OverflowError):
        return False
    denominator = sum(abs(value) for value in actual)
    if denominator <= 0:
        return False
    raw_loss = sum(abs(a - p) for a, p in zip(actual, raw_points)) / denominator
    log_loss = sum(abs(a - p) for a, p in zip(actual, log_points)) / denominator
    return log_loss < raw_loss


def theta(history: list[float], horizon: int, season: int) -> list[float]:
    """Theta with a nested raw-vs-log family choice.

    Alpha variants have equal parameter counts, so minimising their one-step
    SSE is equivalent to AICc selection within that fixed family. A held-out
    slice of the training history chooses raw or log scale before the outer
    rolling fold scores the single resulting Theta candidate.
    """
    if _nested_transform_choice(history, season, _theta_raw):
        return [exp(value) for value in _theta_raw(
            [log(value) for value in history], horizon, season)]
    return _theta_raw(history, horizon, season)


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


def _ets_raw(history: list[float], horizon: int, season: int) -> list[float]:
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


def ets(history: list[float], horizon: int, season: int) -> list[float]:
    """ETS family representative selected internally, then outer-backtested."""
    if _nested_transform_choice(history, season, _ets_raw):
        return [exp(value) for value in _ets_raw(
            [log(value) for value in history], horizon, season)]
    return _ets_raw(history, horizon, season)


def croston_sba(history: list[float], horizon: int, season: int) -> list[float]:
    """Syntetos-Boylan corrected Croston forecast for intermittent demand.

    Smoothing variants are selected by a training-internal trailing holdout;
    only this family representative enters the outer model contest.
    """
    if len(history) < 8 or any(value < 0 for value in history):
        raise ValueError("Croston-SBA requires non-negative history")
    nonzero = [index for index, value in enumerate(history) if value > 0]
    if len(nonzero) < 2:
        raise ValueError("too few non-zero demands")

    def fit(values: list[float], alpha: float) -> float:
        occurrences = [(index, value) for index, value in enumerate(values)
                       if value > 0]
        if len(occurrences) < 2:
            raise ValueError("too few non-zero demands")
        level = occurrences[0][1]
        interval = float(max(1, occurrences[1][0] - occurrences[0][0]))
        previous = occurrences[0][0]
        for index, value in occurrences[1:]:
            gap = max(1, index - previous)
            level = alpha * value + (1 - alpha) * level
            interval = alpha * gap + (1 - alpha) * interval
            previous = index
        return (1 - alpha / 2) * level / max(interval, 1e-12)

    holdout = max(2, min(len(history) // 4, 8))
    train, actual = history[:-holdout], history[-holdout:]
    best_alpha, best_loss = None, float("inf")
    for alpha in (.1, .2, .3, .5):
        try:
            point = fit(train, alpha)
        except ValueError:
            continue
        loss = sum(abs(value - point) for value in actual)
        if loss < best_loss:
            best_alpha, best_loss = alpha, loss
    if best_alpha is None:
        raise ValueError("too few pre-holdout non-zero demands")
    point = fit(history, best_alpha)
    return [point] * horizon


MODELS: dict[str, Callable[[list[float], int, int], list[float]]] = {
    "last_value": last_value,
    "seasonal_naive": seasonal_naive,
    "historical_mean": historical_mean,
    "drift": drift,
    "linear_trend": linear_trend,
    "window_average": window_average,
    "theta": theta,
    "ets": ets,
    "croston_sba": croston_sba,
}
BASELINES = {"last_value", "seasonal_naive", "historical_mean"}


def predict(name: str, history: list[float], horizon: int, season: int) -> list[float]:
    return MODELS[name](history, horizon, season)
