"""Hourly net load with a weather driver and renewable feed-in: choose
a schedule position for the delivery day under asymmetric imbalance
prices.

Series: hourly net load (consumption minus renewable generation) —
a daily cycle, a weekday factor, temperature response (heating and
cooling raise load as the day-mean temperature departs from the
seasonal norm), solar-shaped feed-in with cloud variability, and noise.
The horizon is the next delivery day (24 hours).

Decision: the total MWh position to schedule for the delivery day.
Being short (realized load above the position) pays the short imbalance
price; being long pays the long price. Short ≫ long, so the critical
fractile is short/(short+long) = 0.8: a point-forecast position is
systematically short of optimal.

Context facts (`known_at` doing real work — a forecast of a forecast):
- `temp_norm`: the seasonal normal day-mean temperature, known from the
  start;
- `temp_forecast`: day-mean temperature forecast *vintages* for the
  delivery day — an early vintage issued days out and a later, more
  accurate vintage that revises it, each with its own `known_at`;
- `outage_mw`: a notified feed-in outage (MW off the renewable fleet
  for a window of delivery hours), which raises net load
  mechanistically.

Trap flavor (~15%, disclosed): the temperature forecast is revised
between the early vintage and the cutoff ("gate closure"), and the
revision moves the optimal position materially — the realized future
runs on a true temperature near the revised vintage, the counterfactual
on the stale vintage, and the two optimal positions differ by
construction.

Engine mapping (documented): Gnomon forecasts the net-load series; the
position is the per-step quantile path at the critical fractile
(interpolated between q50 and q90), plus deterministic adjustments from
structured facts — the as-of temperature deviation (forecast minus
norm) times the disclosed calibrated sensitivity, and notified outage
MW times overlap hours.

Parameters and grounding: 300–900 MWh-scale base loads with 25–45%
daily swing mirror a mid-size balancing portfolio; temperature
sensitivity of 0.8–2% of base load per degree matches published
demand-temperature elasticities; solar feed-in up to 15–35% of base
with cloud noise matches distribution-level penetration; short/long
imbalance at 8/2 per MWh mirrors real single-pricing asymmetry. Load
values pass per-case seeded positive affine anonymization (levels
``a*x+b``, MW deltas ``a*x``); temperatures stay in real degrees
Celsius (disclosed — they are drivers, not the priced series), and
decision structure is verified invariant on the rounded shown numbers.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

from benchmarks.enterprisebench.harness import (
    Case,
    ContextItem,
    CostModel,
    DomainPack,
    register,
    tail_shock,
)
from benchmarks.enterprisebench.textgen import register_templates

HISTORY = 14 * 24
HORIZON = 24
PRICE_SHORT = 8.0
PRICE_LONG = 2.0
CONFIG: dict[str, Any] = {
    "history_hours": HISTORY, "horizon_hours": HORIZON, "frequency": "h",
    "units": "shown_load_units_short_8_long_2_per_unit",
    "base_load": (300.0, 900.0), "daily_swing": (0.25, 0.45),
    "weekday_uplift": (0.05, 0.15),
    "temp_sensitivity_fraction_per_degree": (0.008, 0.02),
    "temp_norm": (10.0, 22.0), "temp_ar": 0.7,
    "solar_share": (0.15, 0.35), "noise_fraction": 0.015,
    #: Deep-solar portfolios: enough midday feed-in that *net* load
    #: shows the duck curve — midday dip, evening peak — the regime a
    #: modern balancing desk actually schedules against. The remainder
    #: keep the consumption-dominated midday-peak profile. Achieved mix
    #: disclosed in provenance.
    "deep_solar_case_share": 0.5,
    "deep_solar_share": (0.45, 0.7),
    "evening_peak_hour": 19, "midday_peak_hour": 12,
    "outage_share": 0.35,
    "trap_position_shift_min_fraction": 0.06,
    "outcome_targets": {"plain": 0.85, "trap": 0.15},
}


def _load_series(rng: random.Random, hours: int, base: float,
                 swing: float, weekday: float, sensitivity: float,
                 day_temps: list[float], norm: float, solar: float,
                 peak_hour: int,
                 outages: list[dict[str, Any]]) -> list[float]:
    values = []
    for hour in range(hours):
        day, hour_of_day = divmod(hour, 24)
        level = base * (1.0 + swing * math.cos(
            (hour_of_day - peak_hour) / 24.0 * 2.0 * math.pi))
        if day % 7 < 5:
            level *= 1.0 + weekday
        level += base * sensitivity * abs(day_temps[day] - norm)
        sun = max(0.0, math.sin((hour_of_day - 6) / 12.0 * math.pi))
        cloud = 0.6 + 0.4 * rng.random()
        feed_in = base * solar * sun * cloud
        for outage in outages:
            if outage["from"] <= hour <= outage["to"]:
                feed_in = max(0.0, feed_in - outage["mw"])
        level -= feed_in
        values.append(level + tail_shock(rng, CONFIG["noise_fraction"]
                                         * base))
    return values


def simulate(seed: int, count: int) -> tuple[list[Case], dict[str, Any]]:
    caps = {cell: int(count * fraction) + 1
            for cell, fraction in CONFIG["outcome_targets"].items()}
    cases: list[Case] = []
    cell_counts: dict[str, int] = {}
    skipped = {"cell_full": 0, "trap_shape": 0, "rounding_flip": 0}
    attempts = 0
    balanced_limit = 300 * count
    hours = HISTORY + HORIZON
    days = hours // 24
    while len(cases) < count and attempts < 2 * balanced_limit:
        attempts += 1
        balanced_phase = attempts <= balanced_limit
        rng = random.Random(f"enterprisebench:energy:{seed}:{attempts}")
        trap = balanced_phase and cell_counts.get("trap", 0) < caps["trap"]
        cell = "trap" if trap else "plain"
        if balanced_phase and cell_counts.get(cell, 0) >= caps[cell]:
            skipped["cell_full"] += 1
            continue

        base = rng.uniform(*CONFIG["base_load"])
        swing = rng.uniform(*CONFIG["daily_swing"])
        weekday = rng.uniform(*CONFIG["weekday_uplift"])
        sensitivity = rng.uniform(
            *CONFIG["temp_sensitivity_fraction_per_degree"])
        norm = rng.uniform(*CONFIG["temp_norm"])
        deep_solar = rng.random() < CONFIG["deep_solar_case_share"]
        if deep_solar:
            solar = rng.uniform(*CONFIG["deep_solar_share"])
            peak_hour = CONFIG["evening_peak_hour"]
        else:
            solar = rng.uniform(*CONFIG["solar_share"])
            peak_hour = CONFIG["midday_peak_hour"]

        day_temps = [norm]
        for _ in range(days - 1):
            day_temps.append(norm + CONFIG["temp_ar"]
                             * (day_temps[-1] - norm) + rng.gauss(0, 2.5))
        delivery_day = days - 1
        if trap:
            # A weather event the early vintage missed: the true
            # delivery-day temperature departs hard from the norm; the
            # stale vintage stays near the norm and is revised before
            # gate closure to (approximately) the truth.
            departure = rng.choice((-1.0, 1.0)) * rng.uniform(9.0, 16.0)
            true_temp = norm + departure
            stale_forecast = norm + rng.uniform(-2.0, 2.0)
            revised_forecast = true_temp + rng.gauss(0.0, 0.8)
        else:
            true_temp = day_temps[delivery_day - 1] * 0.5 + norm * 0.5 \
                + rng.gauss(0.0, 2.0)
            stale_forecast = true_temp + rng.gauss(0.0, 3.0)
            revised_forecast = true_temp + rng.gauss(0.0, 1.0)
        day_temps[delivery_day] = true_temp

        outages: list[dict[str, Any]] = []
        items: list[ContextItem] = [ContextItem(
            "temp-norm", "temp_norm", norm, 0, 0, hours - 1)]
        if rng.random() < CONFIG["outage_share"]:
            start = HISTORY + rng.randrange(8, 16)
            outage = {"from": start,
                      "to": min(start + rng.randrange(3, 9), hours - 1),
                      "mw": rng.uniform(0.05, 0.2) * base}
            outages.append(outage)
            items.append(ContextItem(
                "outage-0", "outage_mw", outage["mw"],
                HISTORY - rng.randrange(12, 72), outage["from"],
                outage["to"], text_only=rng.random() < 0.3))
        items.append(ContextItem(
            "temp-fc-a", "temp_forecast", stale_forecast,
            HISTORY - rng.randrange(48, 80), HISTORY, hours - 1))
        items.append(ContextItem(
            "temp-fc-b", "temp_forecast", revised_forecast,
            HISTORY - rng.randrange(2, 12), HISTORY, hours - 1,
            revises="temp-fc-a", trap=trap))
        # A post-cutoff vintage nobody could know at gate closure: the
        # as-of resolver must excise it and the lint must prove it.
        if rng.random() < 0.5:
            items.append(ContextItem(
                "temp-fc-c", "temp_forecast",
                true_temp + rng.gauss(0.0, 0.3),
                HISTORY + rng.randrange(1, 12), HISTORY, hours - 1,
                revises="temp-fc-b"))

        series_rng = random.Random(
            f"enterprisebench:energy:{seed}:{attempts}:series")
        values = _load_series(series_rng, hours, base, swing, weekday,
                              sensitivity, day_temps, norm, solar,
                              peak_hour, outages)
        if trap:
            stale_temps = list(day_temps)
            stale_temps[delivery_day] = stale_forecast
            stale_rng = random.Random(
                f"enterprisebench:energy:{seed}:{attempts}:series")
            stale_values = _load_series(
                stale_rng, hours, base, swing, weekday, sensitivity,
                stale_temps, norm, solar, peak_hour, outages)
        history, future = values[:HISTORY], values[HISTORY:]
        real_total = sum(future)
        if trap:
            stale_total = sum(stale_values[HISTORY:])
            if abs(real_total - stale_total) < max(
                    1.0, CONFIG["trap_position_shift_min_fraction"]
                    * abs(real_total)):
                skipped["trap_shape"] += 1
                continue

        # Load levels a*x+b, MW deltas a*x, temperatures untouched.
        a = rng.uniform(0.6, 2.4)
        b = rng.uniform(20, 400) - a * statistics.median(history)
        shown_history = tuple(round(a * v + b, 4) for v in history)
        shown_future = tuple(round(a * v + b, 4) for v in future)
        shown_real_total = sum(shown_future)

        def transform(item: ContextItem) -> ContextItem:
            value = (round(a * item.value, 4)
                     if item.kind == "outage_mw"
                     else round(item.value, 4))
            return ContextItem(item.item_id, item.kind, value,
                               item.known_at, item.effective_from,
                               item.effective_to, item.revises,
                               item.text_only, item.trap, item.aux)

        shown_items = tuple(transform(item) for item in items)
        trap_optimal = stale_optimal = None
        if trap:
            shown_stale_total = sum(
                round(a * v + b, 4) for v in stale_values[HISTORY:])
            if abs(shown_real_total - shown_stale_total) < 1e-6:
                skipped["rounding_flip"] += 1
                continue
            trap_optimal = {"schedule_mwh": round(shown_real_total, 4)}
            stale_optimal = {"schedule_mwh": round(shown_stale_total, 4)}
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        cases.append(Case(
            case_id=f"en{seed}-{len(cases):04d}", domain="energy",
            frequency="h", values=shown_history, future=shown_future,
            horizon=HORIZON, items=shown_items, threshold=None,
            trap=trap, trap_optimal=trap_optimal,
            stale_optimal=stale_optimal,
            series_id=f"portfolio-{len(cases):04d}",
            meta={"truth_event": True, "truth_first_step": None,
                  "outcome_cell": cell,
                  "regime": ("deep_solar_duck" if deep_solar
                             else "consumption_dominated"),
                  "realized_total": round(shown_real_total, 4)}))
    if len(cases) < count:
        raise ValueError(
            f"energy: only {len(cases)}/{count} cases after "
            f"{attempts} attempts; skipped={skipped}")
    provenance = {
        "generator": "mechanistic_weather_driven_net_load",
        "config": CONFIG,
        "attempts": attempts, "skipped": skipped,
        "outcome_distribution": dict(sorted(cell_counts.items())),
        "trap_share": cell_counts.get("trap", 0) / len(cases),
        "regime_mix": {
            regime: sum(1 for case in cases
                        if case.meta["regime"] == regime)
            for regime in ("deep_solar_duck", "consumption_dominated")},
        "cases_per_series": {case.series_id: 1 for case in cases},
        "independence": (
            "one independent simulated series per case; futures cannot "
            "overlap; labels can still co-move through shared parameter "
            "ranges — a caveat, not an independence claim"),
        "anonymization": (
            "per_case_seeded_positive_affine;load_levels_ax_plus_b;"
            "mw_deltas_ax;temperatures_untouched_disclosed;"
            "decision_invariance_verified_on_rounded_shown_numbers"),
    }
    return cases, provenance


def _score(decision: dict[str, Any], case: Case) -> dict[str, float]:
    realized = float(case.meta["realized_total"])
    position = float(decision["schedule_mwh"])
    cost = (PRICE_SHORT * max(0.0, realized - position)
            + PRICE_LONG * max(0.0, position - realized))
    return {"cost": cost, "regret": cost}


def _no_action(case: Case) -> dict[str, Any]:
    """The desk's default: schedule yesterday's same-day total."""
    return {"schedule_mwh": float(sum(case.values[-HORIZON:]))}


def _adjustment(case: Case, facts: list[ContextItem]) -> float:
    """Deterministic structured-context adjustment in shown units: the
    as-of temperature deviation times the disclosed calibrated
    sensitivity, plus notified outage MW times overlap hours."""
    norm = next((item.value for item in facts
                 if item.kind == "temp_norm"), None)
    forecast = next((item.value for item in facts
                     if item.kind == "temp_forecast"), None)
    mid = sum(CONFIG["temp_sensitivity_fraction_per_degree"]) / 2.0
    # Base load in shown units, estimated from the daily swing amplitude
    # (offset-free under affine anonymization: max-min scales by a).
    amplitude = (max(case.values) - min(case.values)) if case.values \
        else 0.0
    base_proxy = amplitude / 0.7
    adjustment = 0.0
    if norm is not None and forecast is not None:
        # Hourly abs-deviation response times 24 delivery hours,
        # differenced against the ~3-degree typical deviation already
        # embedded in the history the forecast extrapolates.
        adjustment += (abs(forecast - norm) - 3.0) * mid * base_proxy \
            * HORIZON
    for item in facts:
        if item.kind != "outage_mw":
            continue
        overlap_from = max(item.effective_from, case.cutoff)
        overlap_to = min(item.effective_to,
                         case.cutoff + case.horizon - 1)
        if overlap_to >= overlap_from:
            adjustment += item.value * (overlap_to - overlap_from + 1)
    return adjustment


def _engine_inputs(case: Case, facts: list[ContextItem]) -> dict[str, Any]:
    return {"threshold": None, "basis": "shown_series",
            "adjustment_mwh": _adjustment(case, facts)}


def _engine_decision(case: Case, packet: dict[str, Any],
                     inputs: dict[str, Any]) -> dict[str, Any]:
    """Position at the critical fractile: q50 + 0.75 * (q90 - q50) per
    step (tau = 0.8 interpolated), summed, plus the structured-context
    adjustment."""
    rows = packet.get("forecast") or []
    if len(rows) != case.horizon:
        return _no_action(case)
    position = sum(float(row["q50"])
                   + 0.75 * (float(row["q90"]) - float(row["q50"]))
                   for row in rows)
    return {"schedule_mwh": round(position + inputs["adjustment_mwh"], 4)}


def _decision_from_forecast(case: Case, path: list[float],
                            inputs: dict[str, Any]) -> dict[str, Any]:
    return {"schedule_mwh": round(
        sum(float(v) for v in path) + inputs["adjustment_mwh"], 4)}


def _parse_decision(payload: dict[str, Any],
                    case: Case) -> dict[str, Any] | None:
    value = payload.get("schedule_mwh")
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value):
        return None
    return {"schedule_mwh": float(value)}


def _question(case: Case) -> str:
    return (
        "Decision: schedule the total position for the next delivery "
        "day (the 24 unshown hours), in the shown load units. Every "
        "unit of realized load above the position pays the short "
        "imbalance price; every unit below pays the long price. "
        "Temperature forecast memos are vintages for the delivery day; "
        "use the vintage in force as of gate closure (the cutoff).")


PACK = DomainPack(
    name="energy",
    version="0.2",
    decision_kind="quantity",
    simulate=simulate,
    cost_model=CostModel(
        names={"short_per_unit": PRICE_SHORT,
               "long_per_unit": PRICE_LONG},
        break_even=PRICE_SHORT / (PRICE_SHORT + PRICE_LONG),
        score=_score,
        no_action=_no_action,
        optimal=lambda case: {"schedule_mwh":
                              float(case.meta["realized_total"])},
    ),
    decision_schema={
        "instruction": 'Return {"schedule_mwh": <number>}.',
        "fields": {"schedule_mwh": "total delivery-day position"},
    },
    context_kinds={
        "temp_norm": {"unit": "celsius", "bounds": (-40.0, 60.0),
                      "max_span": 500},
        "temp_forecast": {"unit": "celsius", "bounds": (-40.0, 60.0),
                          "max_span": 500},
        "outage_mw": {"unit": "shown_load_units",
                      "bounds": (0.0, 100_000.0), "max_span": 72},
    },
    question=_question,
    engine_inputs=_engine_inputs,
    engine_decision=_engine_decision,
    decision_from_forecast=_decision_from_forecast,
    constant_policies=lambda case: {
        "schedule_zero": {"schedule_mwh": 0.0},
        "schedule_double_naive": {
            "schedule_mwh": 2.0 * float(sum(case.values[-HORIZON:]))},
    },
    parse_decision=_parse_decision,
    decision_scalar=lambda decision: float(
        decision.get("schedule_mwh", 0.0)),
    config=CONFIG,
    season_length=24,
    extra_metrics=None,
)

register(PACK)

register_templates("temp_norm", base=(
    "Climatology note {ref}: the seasonal normal day-mean temperature "
    "for this portfolio is {value} degrees.",
    "Reference sheet ({ref}): normal daily mean temperature {value} C.",
))
register_templates("temp_forecast", base=(
    "Weather desk {ref}: the delivery-day mean temperature is forecast "
    "at {value} degrees (vintage issued {known_date}).",
    "Met bulletin ({ref}, issued {known_date}): day-mean {value} C "
    "expected for the delivery day.",
), revision=(
    "Forecast update {ref} issued {known_date}: the delivery-day mean, "
    "earlier put at {prev_value} degrees, is now expected at {value} "
    "degrees.",
    "Revised met bulletin ({ref}): {value} C for the delivery day, "
    "superseding the {prev_value} C vintage.",
))
register_templates("outage_mw", base=(
    "Grid notice {ref}: planned feed-in outage of {value} (shown load "
    "units) from {from_date} to {to_date}.",
    "Maintenance advisory ({ref}): {value} of renewable capacity "
    "offline {from_date} through {to_date}.",
), revision=(
    "Amended grid notice {ref}: the outage first sized at {prev_value} "
    "is now {value}, {from_date} to {to_date}.",
))
