"""The context-aware candidate: a drift base plus a measured event effect.

Deliberately the simplest defensible use of context — an additive
intervention effect estimated from detrended history. The event effect is
the mean detrended level during event-active periods minus the mean during
inactive periods, applied to future periods flagged active. If the event
never occurred in training history the effect is unmeasurable and the model
refuses, which the ablation layer reports as an honest exclusion.

**Effect shapes.** Not every intervention holds its level for as long as it
is flagged. A promotion often spikes and decays; a launch often ramps in.
The shape family here is deliberately small and every member is a pure
function of the history and the flags:

``level``       the effect applies in full for every active period.
``decay``       the effect is largest at onset and decays geometrically,
                which is what a pull-forward looks like.
``ramp``        the effect builds linearly across the active window.

The shape is **chosen by the same identical-fold ablation that decides
whether context is admitted at all**, never asserted by the caller — a
caller who could pick the winner could fit a story to the data, and the
whole point of the gate is that context earns its place by measurement.
What a proposer *may* do is nominate one via the event attribute
`expected_shape`: the contest then narrows to that shape alone — one
comparison instead of three, strictly less room to look good by luck — and
the nominated shape still has to beat the history-only baseline or the
event is excluded outright, never silently switched to a shape that fits.
Because three shapes on the same folds is three chances to look good by
luck, the winning shape must beat the *history-only* baseline, not merely
beat the other shapes; `context_eval` enforces that, and applies its
single-fold guard to the winner.
"""

from __future__ import annotations

from statistics import mean

from .models import drift


def event_effect(history: list[float], active_history: list[bool]) -> float:
    """The measured additive effect: mean detrended level during event-active
    periods minus the mean during inactive periods. Raises when the effect is
    unmeasurable from the given history."""
    if len(history) < 2:
        raise ValueError("insufficient history")
    if len(active_history) != len(history):
        raise ValueError("event flags do not align with the time grid")
    if not any(active_history):
        raise ValueError("event has no occurrences in training history")
    if all(active_history):
        raise ValueError("event is active for the entire training history")

    slope = (history[-1] - history[0]) / (len(history) - 1)
    detrended = [value - (history[0] + slope * index) for index, value in enumerate(history)]
    active_level = mean(d for d, active in zip(detrended, active_history) if active)
    inactive_level = mean(d for d, active in zip(detrended, active_history) if not active)
    return active_level - inactive_level


#: Shapes the ablation may choose between, in a fixed order so selection is
#: deterministic when two shapes score identically.
EFFECT_SHAPES = ("level", "decay", "ramp")

#: Geometric decay factor per period after onset. One value, not a fitted
#: one: fitting a decay rate on the same folds that choose the shape would
#: add a second free parameter to a comparison already short of folds.
DECAY_RATE = 0.6


def shape_weights(active: list[bool], shape: str) -> list[float]:
    """Per-period multipliers on the measured effect.

    Each run of consecutive active periods is shaped independently, so a
    second occurrence inside the horizon gets its own onset rather than
    inheriting the first one's decay.
    """
    if shape not in EFFECT_SHAPES:
        raise ValueError(f"unknown effect shape {shape!r}")
    weights = [0.0] * len(active)
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and active[end]:
            end += 1
        span = end - index
        for offset in range(span):
            if shape == "level":
                weights[index + offset] = 1.0
            elif shape == "decay":
                weights[index + offset] = DECAY_RATE ** offset
            else:  # ramp
                weights[index + offset] = (offset + 1) / span
        index = end
    return weights


def event_effect_shaped(
    history: list[float], active_history: list[bool], shape: str,
) -> float:
    """The effect size for a given shape.

    The mean weight of the active periods says how much of a full-strength
    effect the shape actually delivered over the history it was measured on;
    dividing by it keeps the *area* under the shaped effect equal to the
    measured total, so the shapes differ in timing rather than in magnitude.
    Otherwise `decay` would win comparisons simply by being smaller.
    """
    effect = event_effect(history, active_history)
    if shape == "level":
        return effect
    weights = shape_weights(active_history, shape)
    active_weights = [weight for weight, flag in zip(weights, active_history) if flag]
    mean_weight = sum(active_weights) / len(active_weights) if active_weights else 0.0
    if mean_weight <= 1e-12:
        raise ValueError(f"shape {shape!r} delivers no effect over this history")
    return effect / mean_weight


def event_adjusted(
    history: list[float],
    horizon: int,
    season: int,
    active_history: list[bool],
    active_future: list[bool],
    shape: str = "level",
    base_points: list[float] | None = None,
) -> list[float]:
    """The base path with the measured event effect added.

    ``base_points`` is the path the effect is applied to; without it the base
    is ``drift``, which is what the admission ablation uses.

    It is tempting to pass the *selected* model here so that the ablation's
    improvement isolates the event rather than also reflecting a change of
    base model. That is wrong, and was measured to be wrong:
    ``event_effect`` is the raw active-minus-inactive level difference, so
    adding it to a base that already models the event — any seasonal model,
    when the events recur on a period — counts the same bump twice and fold
    improvements collapse toward −1. Layering is only sound on a base whose
    residuals do *not* already contain the event.
    """
    if len(history) < 2:
        raise ValueError("insufficient history")
    if len(active_history) != len(history) or len(active_future) != horizon:
        raise ValueError("event flags do not align with the time grid")
    effect = event_effect_shaped(history, active_history, shape)
    weights = shape_weights(active_future, shape)

    base = base_points if base_points is not None else drift(history, horizon, season)
    if len(base) != horizon:
        raise ValueError("base path does not span the horizon")
    return [point + effect * weight for point, weight in zip(base, weights)]
