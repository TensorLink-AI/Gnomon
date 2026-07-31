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


def measure_effect(values: list[float], active: list[bool]) -> float:
    """The additive level shift attributable to the flagged periods: the mean
    while active minus the mean while inactive.

    Applied to detrended history this is the context candidate's event
    effect; applied to another model's in-sample residuals it is the part of
    the event that model has *not* already explained, which is what the
    adjudication ladder's joint candidate needs."""
    if len(active) != len(values):
        raise ValueError("event flags do not align with the time grid")
    if not any(active):
        raise ValueError("event has no occurrences in training history")
    if all(active):
        raise ValueError("event is active for the entire training history")
    active_level = mean(value for value, flag in zip(values, active) if flag)
    inactive_level = mean(value for value, flag in zip(values, active) if not flag)
    return active_level - inactive_level


def event_effect(history: list[float], active_history: list[bool]) -> float:
    """The measured additive event effect over detrended history.

    Isolated from ``event_adjusted`` so any base forecast — not just drift —
    can carry a measured effect."""
    if len(history) < 2:
        raise ValueError("insufficient history")
    if len(active_history) != len(history):
        raise ValueError("event flags do not align with the time grid")
    slope = (history[-1] - history[0]) / (len(history) - 1)
    detrended = [value - (history[0] + slope * index) for index, value in enumerate(history)]
    return measure_effect(detrended, active_history)


def apply_event_effect(
    base: list[float], effect: float, active_future: list[bool]
) -> list[float]:
    """Add the measured effect to the periods a known event marks active."""
    return [point + (effect if active else 0.0) for point, active in zip(base, active_future)]


def event_adjusted(
    history: list[float],
    horizon: int,
    season: int,
    active_history: list[bool],
    active_future: list[bool],
) -> list[float]:
    if len(active_future) != horizon:
        raise ValueError("event flags do not align with the time grid")
    effect = event_effect(history, active_history)
    return apply_event_effect(drift(history, horizon, season), effect, active_future)
