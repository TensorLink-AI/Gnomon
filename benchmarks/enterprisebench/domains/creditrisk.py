"""Monthly delinquency roll-rates with a lagged macro driver: tighten
underwriting or hold, against a provision threshold.

Series: a cohort's monthly roll-rate (the shown units are anonymized
index points). The mechanism is a base rate, a mild annual seasonal, an
AR noise term, and a macro driver that acts with a one-month lag —
month m's roll-rate responds to month m-1's macro value.

The macro series itself arrives as context with a *publication lag*:
the value for month m is released mid-month m+1, so its ``known_at`` is
one month after its effective month — March's value is known in
mid-April, and the as-of resolver enforces exactly that. Later
restatements revise earlier releases.

Decision: will the roll-rate exceed the provision threshold (a dated
fact) within the horizon — tighten now (forgone revenue, fully avoids
the breach) or hold (a capital breach costs multiples). Break-even
probability = tighten/breach = 0.2, base rate held near it.

Trap flavor (~15%, disclosed): a *restated macro series* — the release
for the latest published month is restated before the cutoff, and the
restatement moves the mechanistically implied future roll-rate path
across the provision threshold; the counterfactual future simulated
from the stale value verifies the flip by construction.

Engine mapping: Gnomon forecasts the roll-rate series against the as-of
provision threshold through the governed breach ladder. The engine
consumes structured facts for the threshold only; the macro releases
reach it solely through the history they already shaped — recovering
the restatement's forward implication is precisely the value the
context arms are being tested for.

Parameters and grounding: base roll-rates of 2–6 index points with
macro betas of 0.3–0.9 points per macro unit mirror consumer-credit
roll-rate elasticities to unemployment-style indices; the one-month
mechanism lag plus the one-month publication lag reproduce how macro
data actually reaches a provisioning desk. Roll-rates and the threshold
pass per-case affine anonymization (``a*x+b``, deltas ``a*x``); the
macro index passes its own independent affine transform (different
unit, disclosed).
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from benchmarks.enterprisebench.harness import (
    Case,
    ContextItem,
    DomainPack,
    as_of,
    binary_cost_model,
    binary_decision_schema,
    crossing_decision,
    governed_engine_decision,
    parse_binary_decision,
    register,
)
from benchmarks.enterprisebench.textgen import register_templates

HISTORY = 60
HORIZON = 6
COST_TIGHTEN = 4.0
COST_BREACH = 20.0
CONFIG: dict[str, Any] = {
    "history_months": HISTORY, "horizon_months": HORIZON,
    "frequency": "MS",
    "units": "cost_units_tighten_4_capital_breach_20",
    "base_roll": (2.0, 6.0), "seasonal_amplitude": (0.05, 0.25),
    "macro_mu": (4.0, 9.0), "macro_rho": 0.85, "macro_sigma": 0.35,
    "beta": (0.3, 0.9), "noise_sigma": (0.08, 0.2),
    "publication_lag_months": 1, "mechanism_lag_months": 1,
    "released_months": 8,
    "outcome_targets": {"no_breach": 0.55, "breach": 0.30, "trap": 0.15},
}


def _macro_path(rng: random.Random, months: int, mu: float) -> list[float]:
    values = [mu + rng.gauss(0.0, CONFIG["macro_sigma"])]
    for _ in range(months - 1):
        values.append(mu + CONFIG["macro_rho"] * (values[-1] - mu)
                      + rng.gauss(0.0, CONFIG["macro_sigma"]))
    return values


def _roll_path(rng: random.Random, months: int, base: float,
               amplitude: float, beta: float, sigma: float, mu: float,
               macro: list[float]) -> list[float]:
    import math
    values = []
    for month in range(months):
        seasonal = amplitude * math.sin(month / 12.0 * 2.0 * math.pi)
        driver = macro[max(0, month - CONFIG["mechanism_lag_months"])]
        values.append(base * (1.0 + seasonal) + beta * (driver - mu)
                      + rng.gauss(0.0, sigma))
    return values


def simulate(seed: int, count: int) -> tuple[list[Case], dict[str, Any]]:
    caps = {cell: int(count * fraction) + 1
            for cell, fraction in CONFIG["outcome_targets"].items()}
    cases: list[Case] = []
    cell_counts: dict[str, int] = {}
    skipped = {"cell_full": 0, "rounding_flip": 0, "trap_shape": 0}
    attempts = 0
    balanced_limit = 300 * count
    months = HISTORY + HORIZON
    lag = CONFIG["publication_lag_months"]
    while len(cases) < count and attempts < 2 * balanced_limit:
        attempts += 1
        balanced_phase = attempts <= balanced_limit
        rng = random.Random(
            f"enterprisebench:creditrisk:{seed}:{attempts}")
        trap = balanced_phase and cell_counts.get("trap", 0) < caps["trap"]
        cell = "trap" if trap else None

        base = rng.uniform(*CONFIG["base_roll"])
        amplitude = rng.uniform(*CONFIG["seasonal_amplitude"])
        beta = rng.uniform(*CONFIG["beta"])
        sigma = rng.uniform(*CONFIG["noise_sigma"])
        mu = rng.uniform(*CONFIG["macro_mu"])
        macro_rng = random.Random(
            f"enterprisebench:creditrisk:{seed}:{attempts}:macro")
        macro = _macro_path(macro_rng, months, mu)

        stale_macro = None
        restated_month = HISTORY - 1 - lag
        if trap:
            # The initial release for the latest published month was
            # wrong; the restatement (known before the cutoff) is the
            # true value the future mechanism runs on.
            shift = rng.choice((-1.0, 1.0)) * rng.uniform(1.2, 2.2)
            stale_macro = list(macro)
            stale_macro[restated_month] = macro[restated_month] - shift
            # Both continuations share the AR innovations: rebuild the
            # stale continuation from the stale level with the same
            # residual draws.
            for month in range(restated_month + 1, months):
                innovation = macro[month] - (
                    mu + CONFIG["macro_rho"] * (macro[month - 1] - mu))
                stale_macro[month] = mu + CONFIG["macro_rho"] * (
                    stale_macro[month - 1] - mu) + innovation

        roll_rng = random.Random(
            f"enterprisebench:creditrisk:{seed}:{attempts}:roll")
        roll = _roll_path(roll_rng, months, base, amplitude, beta, sigma,
                          mu, macro)
        history, future = roll[:HISTORY], roll[HISTORY:]
        if trap:
            stale_roll_rng = random.Random(
                f"enterprisebench:creditrisk:{seed}:{attempts}:roll")
            stale_future = _roll_path(
                stale_roll_rng, months, base, amplitude, beta, sigma, mu,
                stale_macro)[HISTORY:]
            low = min(max(future), max(stale_future))
            high = max(max(future), max(stale_future))
            if high - low < 3.0 * sigma:
                skipped["trap_shape"] += 1
                continue
            threshold = low + rng.uniform(0.35, 0.65) * (high - low)
        else:
            scale = max(sigma, 1e-6)
            breach = (cell_counts.get("breach", 0) < caps["breach"]
                      if balanced_phase else rng.random() < 0.3)
            margin = (rng.uniform(0.3, 2.0) if breach
                      else rng.uniform(0.5, 5.0)) * scale
            threshold = max(future) - margin if breach \
                else max(future) + margin
            stale_future = None
            cell = "breach" if breach else "no_breach"
        if balanced_phase and cell_counts.get(cell, 0) >= caps[cell]:
            skipped["cell_full"] += 1
            continue

        items: list[ContextItem] = [ContextItem(
            "provision", "provision_threshold", threshold, 0, 0,
            months - 1)]
        # Macro releases with the publication lag: month m's value is
        # known in month m+1. The most recent months are unreleased.
        first_released = max(0, restated_month
                             - CONFIG["released_months"] + 1)
        for month in range(first_released, restated_month + 1):
            items.append(ContextItem(
                f"macro-{month:02d}", "macro_release", macro[month],
                month + lag, month, month,
                text_only=rng.random() < 0.2 and month != restated_month))
        if trap:
            # The initial (stale) release becomes version A; the
            # restatement revises it, still pre-cutoff.
            stale_value = stale_macro[restated_month]
            items = [item for item in items
                     if item.item_id != f"macro-{restated_month:02d}"]
            items.append(ContextItem(
                f"macro-{restated_month:02d}", "macro_release",
                stale_value, restated_month + lag, restated_month,
                restated_month))
            items.append(ContextItem(
                f"macro-{restated_month:02d}-restated", "macro_release",
                macro[restated_month],
                HISTORY, restated_month, restated_month,
                revises=f"macro-{restated_month:02d}", trap=True))
        # The freshest month releases exactly at the cutoff (month m,
        # published month m+1) — visible, and the tightest test of the
        # publication lag.
        freshest = HISTORY - lag
        items.append(ContextItem(
            f"macro-{freshest:02d}", "macro_release", macro[freshest],
            freshest + lag, freshest, freshest))
        # A post-cutoff restatement of it the resolver must excise: the
        # statistics office corrects the print after the decision was
        # already made.
        items.append(ContextItem(
            f"macro-{freshest:02d}-post", "macro_release",
            macro[freshest] + rng.gauss(0.0, 0.4), HISTORY + 1,
            freshest, freshest, revises=f"macro-{freshest:02d}"))

        # Roll-rates and threshold share one affine transform; the
        # macro index gets its own (it is a different unit).
        a = rng.uniform(0.6, 2.4)
        b = rng.uniform(5, 60) - a * statistics.median(history)
        a2 = rng.uniform(0.5, 3.0)
        b2 = rng.uniform(10, 200) - a2 * mu
        shown_history = tuple(round(a * v + b, 4) for v in history)
        shown_future = tuple(round(a * v + b, 4) for v in future)
        shown_threshold = round(a * threshold + b, 4)

        def transform(item: ContextItem) -> ContextItem:
            if item.kind == "provision_threshold":
                value = round(a * item.value + b, 4)
            else:
                value = round(a2 * item.value + b2, 4)
            return ContextItem(item.item_id, item.kind, value,
                               item.known_at, item.effective_from,
                               item.effective_to, item.revises,
                               item.text_only, item.trap, item.aux)

        shown_items = tuple(transform(item) for item in items)
        breach_steps = [step for step, value
                        in enumerate(shown_future, 1)
                        if value > shown_threshold]
        expected = (max(future) > threshold if trap
                    else cell == "breach")
        if bool(breach_steps) != expected:
            skipped["rounding_flip"] += 1
            continue
        trap_optimal = stale_optimal = None
        if trap:
            stale_breach = any(
                round(a * v + b, 4) > shown_threshold
                for v in stale_future)
            if stale_breach == bool(breach_steps):
                skipped["rounding_flip"] += 1
                continue
            trap_optimal = {"action": "act" if breach_steps
                            else "monitor"}
            stale_optimal = {"action": "act" if stale_breach
                             else "monitor"}
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        cases.append(Case(
            case_id=f"cr{seed}-{len(cases):04d}", domain="creditrisk",
            frequency="MS", values=shown_history, future=shown_future,
            horizon=HORIZON, items=shown_items,
            threshold=shown_threshold, trap=trap,
            trap_optimal=trap_optimal, stale_optimal=stale_optimal,
            series_id=f"cohort-{len(cases):04d}",
            meta={"truth_event": bool(breach_steps),
                  "truth_first_step": (breach_steps[0]
                                       if breach_steps else None),
                  "outcome_cell": cell}))
    if len(cases) < count:
        raise ValueError(
            f"creditrisk: only {len(cases)}/{count} cases after "
            f"{attempts} attempts; skipped={skipped}")
    provenance = {
        "generator": "mechanistic_lagged_macro_roll_rates",
        "config": CONFIG,
        "attempts": attempts, "skipped": skipped,
        "outcome_distribution": dict(sorted(cell_counts.items())),
        "trap_share": cell_counts.get("trap", 0) / len(cases),
        "breach_base_rate": statistics.mean(
            case.meta["truth_event"] for case in cases),
        "cases_per_series": {case.series_id: 1 for case in cases},
        "independence": (
            "one independent simulated series per case; futures cannot "
            "overlap; labels can still co-move through shared parameter "
            "ranges — a caveat, not an independence claim"),
        "anonymization": (
            "per_case_seeded_positive_affine;roll_rates_and_threshold_"
            "share_one_transform;macro_index_transformed_independently;"
            "decision_invariance_verified_on_rounded_shown_numbers"),
    }
    return cases, provenance


def _engine_inputs(case: Case, facts: list[ContextItem]) -> dict[str, Any]:
    threshold = next((item.value for item in facts
                      if item.kind == "provision_threshold"),
                     max(case.values))
    return {"threshold": threshold, "basis": "shown_series"}


def _question(case: Case) -> str:
    return (
        "Decision: will the cohort roll-rate exceed the provision "
        "threshold (the provision_threshold fact) in any of the unshown "
        "months? Macro releases in the context carry a one-month "
        "publication lag and act on roll-rates with a one-month "
        "mechanism lag; use each release in the version known as of the "
        "cutoff. Acting (tightening underwriting) costs forgone revenue "
        "and fully avoids a breach; holding costs nothing unless the "
        "threshold is breached.")


PACK = DomainPack(
    name="creditrisk",
    version="0.1",
    decision_kind="binary",
    simulate=simulate,
    cost_model=binary_cost_model(COST_TIGHTEN, COST_BREACH,
                                 "tighten_forgone_revenue",
                                 "capital_breach"),
    decision_schema=binary_decision_schema(HORIZON),
    context_kinds={
        "provision_threshold": {"unit": "index_points",
                                "bounds": (-1_000_000.0, 1_000_000.0),
                                "max_span": 200},
        "macro_release": {"unit": "macro_index",
                          "bounds": (-1_000_000.0, 1_000_000.0),
                          "max_span": 2},
    },
    question=_question,
    engine_inputs=_engine_inputs,
    engine_decision=lambda case, packet, inputs: governed_engine_decision(
        case, packet, COST_TIGHTEN, COST_BREACH),
    decision_from_forecast=lambda case, path, inputs: crossing_decision(
        case, path, inputs["threshold"]),
    constant_policies=lambda case: {
        "always_act": {"action": "act", "event_expected": True,
                       "first_event_step": 1},
        "never_act": {"action": "monitor", "event_expected": False,
                      "first_event_step": None}},
    parse_decision=parse_binary_decision,
    decision_scalar=lambda decision: 1.0
    if decision.get("action") == "act" else 0.0,
    config=CONFIG,
    season_length=12,
)

register(PACK)

register_templates("provision_threshold", base=(
    "Risk committee minute {ref}: the provisioning trigger for this "
    "cohort stands at {value} index points.",
    "Policy memo ({ref}): tighten when the roll-rate approaches "
    "{value}; that is the provision threshold.",
))
register_templates("macro_release", base=(
    "Statistics office release {ref} (published {known_date}): the "
    "macro index for the month starting {from_date} printed at "
    "{value}.",
    "Macro wire ({ref}): {from_date} index value {value}, released "
    "{known_date}.",
), revision=(
    "Restatement {ref} (published {known_date}): the macro index for "
    "the month starting {from_date}, first released at {prev_value}, "
    "is restated to {value}.",
    "Data correction ({ref}): {from_date} macro print revised from "
    "{prev_value} to {value}.",
))
