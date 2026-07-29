"""The context-aware candidate: a drift base plus a measured event effect.

Deliberately the simplest defensible use of context — an additive
intervention effect estimated from detrended history. The event effect is
the mean detrended level during event-active periods minus the mean during
inactive periods, applied to future periods flagged active. If the event
never occurred in training history the effect is unmeasurable and the model
refuses, which the ablation layer reports as an honest exclusion.
"""

from __future__ import annotations

from statistics import mean

from .models import drift


def event_adjusted(
    history: list[float],
    horizon: int,
    season: int,
    active_history: list[bool],
    active_future: list[bool],
) -> list[float]:
    if len(history) < 2:
        raise ValueError("insufficient history")
    if len(active_history) != len(history) or len(active_future) != horizon:
        raise ValueError("event flags do not align with the time grid")
    if not any(active_history):
        raise ValueError("event has no occurrences in training history")
    if all(active_history):
        raise ValueError("event is active for the entire training history")

    slope = (history[-1] - history[0]) / (len(history) - 1)
    detrended = [value - (history[0] + slope * index) for index, value in enumerate(history)]
    active_level = mean(d for d, active in zip(detrended, active_history) if active)
    inactive_level = mean(d for d, active in zip(detrended, active_history) if not active)
    effect = active_level - inactive_level

    base = drift(history, horizon, season)
    return [point + (effect if active else 0.0) for point, active in zip(base, active_future)]
