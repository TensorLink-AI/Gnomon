from __future__ import annotations

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


MODELS: dict[str, Callable[[list[float], int, int], list[float]]] = {
    "last_value": last_value,
    "seasonal_naive": seasonal_naive,
    "drift": drift,
}
BASELINES = {"last_value", "seasonal_naive"}


def predict(name: str, history: list[float], horizon: int, season: int) -> list[float]:
    return MODELS[name](history, horizon, season)

