"""Conditional forecasts: answering *if* an event happens, without pretending.

The admission gate in `context_eval` requires a verifiable source, because
an event whose `known_at` rests only on someone's say-so cannot be shown not
to leak the future into a backtest. That is the right rule for anything
claiming to be the forecast. But it turns a large class of real question —
"what if we run the promotion in March?" — into an abstention, when the
honest answer is a clearly separated conditional one.

A conditional forecast is never the forecast. It lives in its own
`conditional_forecasts` list, carries its own `conditional_on_event`
support, and states the assumptions it rests on. Every existing field keeps
its unconditional value, so a client that ignores the new key sees exactly
what it saw before.

Two things keep this from becoming a way to launder assertions into numbers:

- The effect size is *measured*, never supplied. It comes from periods in
  the observed history when the same event was active
  (`context_model.event_effect`). An event with no precedent in the history
  has no measurable effect, and the conditional forecast is refused with
  that reason rather than invented from the event's own `attributes`.
- The interval is widened by the standard error of that effect estimate, so
  a magnitude measured from three past occurrences produces a visibly less
  certain answer than one measured from thirty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any

from .context import ContextEvent, backtest_admissible, event_applies, validate_context_event
from .context_model import event_effect
from .evaluation import interval_from_spread

CONDITIONAL_SUPPORT = "conditional_on_event"

#: Two-sided normal quantile for the 80% interval the rest of the pipeline
#: reports. Used only to convert the effect's standard error into the same
#: units as the conformal half-widths it widens.
_Z80 = 1.2815515655446004


@dataclass
class ConditionalForecast:
    events: list[str]
    support: str
    assumptions: list[str]
    forecast: list[dict[str, Any]]
    effect: float
    effect_standard_error: float
    occurrences_in_history: int

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "support": self.support,
            "assumptions": self.assumptions,
            "forecast": self.forecast,
            "measured_effect": self.effect,
            "effect_standard_error": self.effect_standard_error,
            "occurrences_in_history": self.occurrences_in_history,
        }


def _effect_standard_error(
    history: list[float], active_history: list[bool]
) -> tuple[float, int]:
    """Standard error of the active-minus-inactive difference of means.

    The effect is a difference between two sample means of the detrended
    series, so its uncertainty is the usual pooled standard error. Reporting
    it is what stops "measured from three occurrences" and "measured from
    thirty" from looking equally confident.
    """
    slope = (history[-1] - history[0]) / (len(history) - 1)
    detrended = [value - (history[0] + slope * index)
                 for index, value in enumerate(history)]
    active = [d for d, flag in zip(detrended, active_history) if flag]
    inactive = [d for d, flag in zip(detrended, active_history) if not flag]
    if len(active) < 2 or len(inactive) < 2:
        # One observation on either side gives a difference with no measurable
        # spread. Report the count so the caller can see how thin it is.
        return float("inf"), len(active)
    active_mean, inactive_mean = mean(active), mean(inactive)
    active_var = sum((d - active_mean) ** 2 for d in active) / (len(active) - 1)
    inactive_var = sum((d - inactive_mean) ** 2 for d in inactive) / (len(inactive) - 1)
    return (active_var / len(active) + inactive_var / len(inactive)) ** 0.5, len(active)


def conditional_candidates(
    events: list[ContextEvent], series_name: str
) -> tuple[list[ContextEvent], list[dict[str, str]]]:
    """Events eligible for a conditional answer but not for the real gate.

    An event with a verifiable source is not here: it goes through the
    admission gate, and if the gate rejects it that is a measured verdict,
    not a licence to report it conditionally instead.
    """
    candidates: list[ContextEvent] = []
    excluded: list[dict[str, str]] = []
    for event in events:
        if validate_context_event(event):
            excluded.append({"event_id": event.event_id,
                             "reason": "event does not satisfy the contract"})
        elif not event_applies(event, series_name):
            continue
        elif event.status == "cancelled":
            excluded.append({"event_id": event.event_id,
                             "reason": "cancelled event cannot define a scenario"})
        elif backtest_admissible(event):
            continue
        else:
            candidates.append(event)
    return candidates, excluded


def assess_conditional(
    values: list[float],
    timestamps: list[datetime],
    future_timestamps: list[datetime],
    events: list[ContextEvent],
    series_name: str,
    horizon: int,
    season: int,
    spreads: dict[int, tuple[float, float, float]],
    points: list[float],
) -> tuple[list[ConditionalForecast], list[dict[str, str]]]:
    """One conditional forecast per unsourced event type with a measurable effect.

    ``points`` is the forecast that was actually published, and the effect is
    added to it. Rebuilding a base from scratch — `context_model` uses drift —
    would mean the conditional and unconditional answers differ by a change of
    model as well as by the event, so their difference would no longer be the
    thing the caller asked about.
    """
    from .context_eval import event_flags

    candidates, excluded = conditional_candidates(events, series_name)
    forecasts: list[ConditionalForecast] = []
    if not candidates or not values or not timestamps:
        return forecasts, excluded
    if timestamps[0].tzinfo is None:
        for event in candidates:
            excluded.append({
                "event_id": event.event_id,
                "reason": "dataset timestamps are timezone-naive; context events "
                          "require timezone-aware data",
            })
        return forecasts, excluded

    cutoff = timestamps[-1]
    # Grouped by event_type, because that is the unit an effect is measurable
    # on. A promotion planned for next month is a distinct record from the
    # three promotions already in the history, and it is those three that say
    # what a promotion does to this series. Conditioning each record on only
    # itself would make every forward-looking event unmeasurable by
    # construction — which is the abstention this exists to convert.
    by_type: dict[str, list[ContextEvent]] = {}
    for event in candidates:
        by_type.setdefault(event.event_type, []).append(event)

    for event_type, group in sorted(by_type.items()):
        active_history = event_flags(group, timestamps, cutoff)
        active_future = event_flags(group, future_timestamps, cutoff)
        identifiers = sorted(event.event_id for event in group)
        future_ids = sorted(
            event.event_id for event in group
            if event_flags([event], future_timestamps, cutoff) != [False] * horizon
        )
        if not any(active_future):
            excluded.append({
                "event_id": ", ".join(identifiers),
                "reason": f"no {event_type!r} event is active during the forecast horizon",
            })
            continue
        try:
            effect = event_effect(values, active_history)
        except ValueError as exc:
            # No precedent in the observed history, so no measurable
            # magnitude. Inventing one from the event's own attributes is
            # exactly what this module exists not to do.
            excluded.append({"event_id": ", ".join(identifiers), "reason": str(exc)})
            continue
        standard_error, occurrences = _effect_standard_error(values, active_history)
        if standard_error == float("inf"):
            excluded.append({
                "event_id": ", ".join(identifiers),
                "reason": f"effect measured from {occurrences} active period(s); "
                          f"too few to estimate its uncertainty",
            })
            continue

        widening = _Z80 * standard_error
        rows: list[dict[str, Any]] = []
        for step, (timestamp, base_point) in enumerate(zip(future_timestamps, points), 1):
            spread = spreads.get(step)
            if spread is None:
                continue
            active = active_future[step - 1]
            point = base_point + (effect if active else 0.0)
            low, centre, high = interval_from_spread(point, spread)
            # Only steps the event actually touches inherit its estimation
            # uncertainty; the rest are the unconditional interval.
            extra = widening if active else 0.0
            rows.append({
                "timestamp": timestamp.isoformat(), "point": point,
                "q10": low - extra, "q50": centre, "q90": high + extra,
            })
        if not rows:
            excluded.append({
                "event_id": ", ".join(identifiers),
                "reason": "no calibrated interval was available to condition",
            })
            continue
        forecasts.append(ConditionalForecast(
            events=future_ids or identifiers,
            support=CONDITIONAL_SUPPORT,
            assumptions=[
                f"conditional: assumes {', '.join(future_ids or identifiers)} "
                f"occurs as described",
                f"the {event_type!r} effect is measured from {occurrences} "
                f"active period(s) in the observed history, not from the event "
                f"description",
                "these events have no verifiable source, so they were not "
                "admissible for backtesting; the unconditional forecast is "
                "unaffected by them",
                "the unconditional forecast is the base; the measured effect is "
                "added at event-active steps, so the difference between the two "
                "is the event and nothing else",
                "interval width is the unconditional calibration widened by the "
                "standard error of the measured effect at event-active steps",
            ],
            forecast=rows,
            effect=effect,
            effect_standard_error=standard_error,
            occurrences_in_history=occurrences,
        ))
    return forecasts, excluded
