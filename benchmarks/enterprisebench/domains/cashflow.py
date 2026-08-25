"""Cash balance vs minimum-balance floor: draw the credit line or not.

Series: daily cash balance. The mechanism is invoices issued and paid at
terms plus behavioral lateness (inflow jumps), payroll every two weeks
and daily opex with a weekday cycle (outflows), mild growth/churn drift
on invoice amounts, and noise. Invoices whose payments land inside the
horizon are dated context facts — the same objects that mechanically
drive the simulated future, so causal ground truth is exact.

Decision: will the balance cross the minimum-balance floor within the
horizon — draw the credit line now (carry cost, fully covers any
shortfall) or not (a realized shortfall costs multiples). Break-even
probability = carry/shortfall = 0.2; the shortfall base rate is held
near it and disclosed.

Trap flavor (~15%, disclosed): an invoice amount is corrected after
issue but before the cutoff, and the correction moves the post-payment
balance trough across the floor — the stale amount and the corrected
amount imply opposite optimal decisions, verified by construction
against the counterfactual future simulated with the stale amount.

A separate, disclosed share of non-trap cases carries a post-cutoff
correction that does drive the realized future: irreducible uncertainty
nobody at the cutoff could know. The leakage lint proves its value is
absent from every prompt; an arm beating the as-of information bound on
those cases would be exhibiting leakage.

Engine mapping (documented approximation): Gnomon forecasts the shown
balance series; a below-floor breach is a threshold breach on the
negated series. Known invoice inflows due inside the horizon raise the
future balance above a pure extrapolation, so the effective floor is
adjusted by the resolved as-of sum of those inflows before the governed
breach ladder prices the decision.

Parameters and grounding: opening balances 50–400k with payroll at
6–12% of balance every 14 days mirror a small-to-mid company's treasury;
invoice terms of 14/30/45 days with 0–7 days behavioral lateness match
observed B2B payment behavior; invoice sizes 2–12% of the opening
balance keep single invoices material but not solitary drivers. Shown
numbers pass per-case seeded positive affine anonymization (balances and
the floor ``a*x+b``, flows ``a*x``); decision structure is verified
invariant on the rounded shown numbers and flipping cases discarded.
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

HISTORY = 112
HORIZON = 14
COST_DRAW = 3.0
COST_SHORTFALL = 15.0
CONFIG: dict[str, Any] = {
    "history": HISTORY, "horizon": HORIZON, "frequency": "D",
    "units": "cost_units_draw_3_shortfall_15",
    "opening_balance": (50_000.0, 400_000.0),
    "invoice_every": (3, 8), "invoice_fraction": (0.02, 0.12),
    "terms": (14, 30, 45), "lateness": (0, 7),
    "payroll_every": 14, "payroll_fraction": (0.06, 0.12),
    "opex_fraction": (0.003, 0.008), "noise_fraction": 0.003,
    "amount_drift": (0.995, 1.002),
    "outcome_targets": {"no_shortfall": 0.55, "shortfall": 0.30,
                        "trap": 0.15},
    "post_cutoff_correction_share": 0.3,
}


def _mechanism(rng: random.Random, length: int
               ) -> tuple[list[float], list[dict[str, Any]], float]:
    """Daily net flows (before any trap invoice) plus the invoice book.

    Returns (net_flows, invoices, opening). Each invoice records issue,
    terms, lateness, pay step, and amount; paid-in-window inflows are
    already inside net_flows, flagged so a trap can be layered on top.
    """
    opening = rng.uniform(*CONFIG["opening_balance"])
    payroll = rng.uniform(*CONFIG["payroll_fraction"]) * opening
    payroll_offset = rng.randrange(CONFIG["payroll_every"])
    opex = rng.uniform(*CONFIG["opex_fraction"]) * opening
    drift = rng.uniform(*CONFIG["amount_drift"])
    noise_sigma = CONFIG["noise_fraction"] * opening
    invoices: list[dict[str, Any]] = []
    step = rng.randrange(1, 6)
    while step < length:
        terms = rng.choice(CONFIG["terms"])
        invoices.append({
            "issue": step, "terms": terms,
            "lateness": rng.randint(*CONFIG["lateness"]),
            "amount": rng.uniform(*CONFIG["invoice_fraction"]) * opening
            * (drift ** step),
        })
        step += rng.randint(*CONFIG["invoice_every"])
    for invoice in invoices:
        invoice["pay"] = invoice["issue"] + invoice["terms"] \
            + invoice["lateness"]
    flows = []
    for index in range(length):
        net = -opex * (1.35 if index % 7 < 5 else 0.6)
        if index % CONFIG["payroll_every"] == payroll_offset:
            net -= payroll
        net += sum(invoice["amount"] for invoice in invoices
                   if invoice["pay"] == index)
        net += rng.gauss(0.0, noise_sigma)
        flows.append(net)
    return flows, invoices, opening


def _balance(opening: float, flows: list[float]) -> list[float]:
    values = []
    level = opening
    for net in flows:
        level += net
        values.append(level)
    return values


def _robust_scale(values: list[float]) -> float:
    diffs = [abs(right - left) for left, right in zip(values, values[1:])]
    return max(statistics.median(diffs), 1e-6) if diffs else 1.0


def simulate(seed: int, count: int) -> tuple[list[Case], dict[str, Any]]:
    targets = CONFIG["outcome_targets"]
    caps = {cell: int(count * fraction) + 1
            for cell, fraction in targets.items()}
    cases: list[Case] = []
    cell_counts: dict[str, int] = {}
    post_cutoff_corrections = 0
    skipped = {"cell_full": 0, "rounding_flip": 0, "trap_shape": 0}
    attempts = 0
    balanced_limit = 300 * count
    length = HISTORY + HORIZON
    while len(cases) < count and attempts < 2 * balanced_limit:
        attempts += 1
        balanced_phase = attempts <= balanced_limit
        rng = random.Random(f"enterprisebench:cashflow:{seed}:{attempts}")
        flows, invoices, opening = _mechanism(rng, length)
        base_future_flows = flows[HISTORY:]
        history = _balance(opening, flows[:HISTORY])
        scale = _robust_scale(history)

        trap = balanced_phase and cell_counts.get("trap", 0) < caps["trap"]
        trap_invoice: dict[str, Any] | None = None
        if trap:
            # One extra invoice issued in history, paid mid-horizon,
            # whose amount was corrected before the cutoff. The future
            # is realized with the corrected amount; the counterfactual
            # future with the stale amount verifies the flip.
            pay = HISTORY + rng.randrange(2, HORIZON - 3)
            revised_down = rng.random() < 0.5
            magnitude = rng.uniform(4.0, 10.0) * scale
            stale = rng.uniform(0.05, 0.10) * opening + (
                magnitude if revised_down else 0.0)
            corrected = stale - magnitude if revised_down \
                else stale + magnitude
            issue = rng.randrange(HISTORY // 2, HISTORY - 20)
            trap_invoice = {"issue": issue, "pay": pay,
                            "stale": stale, "corrected": corrected,
                            "known_correction": rng.randrange(
                                issue + 3, HISTORY)}
        realized_flows = list(base_future_flows)
        stale_flows = list(base_future_flows)
        if trap_invoice is not None:
            offset = trap_invoice["pay"] - HISTORY
            realized_flows[offset] += trap_invoice["corrected"]
            stale_flows[offset] += trap_invoice["stale"]
        future = _balance(history[-1], realized_flows)
        stale_future = _balance(history[-1], stale_flows)

        if trap_invoice is not None:
            # The flip must be decided by the correction: the realized
            # trough and the counterfactual trough must sit on opposite
            # sides of a floor with real margin between them.
            low = min(min(future), min(stale_future))
            high = max(min(future), min(stale_future))
            if high - low < 1.0 * scale:
                skipped["trap_shape"] += 1
                continue
            floor = low + rng.uniform(0.3, 0.7) * (high - low)
            cell = "trap"
        else:
            shortfall = (cell_counts.get("shortfall", 0)
                         < caps["shortfall"]
                         if balanced_phase else rng.random() < 0.3)
            margin = (rng.uniform(0.2, 1.5) if shortfall
                      else rng.uniform(0.3, 4.0)) * scale
            floor = min(future) + margin if shortfall \
                else min(future) - margin
            cell = "shortfall" if shortfall else "no_shortfall"
        if balanced_phase and cell_counts.get(cell, 0) >= caps[cell]:
            skipped["cell_full"] += 1
            continue

        items: list[ContextItem] = [ContextItem(
            "cash-floor", "cash_floor", floor, 0, 0, length - 1)]
        emitted = 0
        for invoice in invoices:
            if invoice["pay"] <= HISTORY - 20 or invoice["pay"] >= length:
                continue
            emitted += 1
            items.append(ContextItem(
                f"inv-{emitted:03d}", "invoice_due", invoice["amount"],
                invoice["issue"], invoice["pay"], invoice["pay"],
                aux=(("terms_days", invoice["terms"]),)))
        if trap_invoice is not None:
            items.append(ContextItem(
                "inv-trap-a", "invoice_due", trap_invoice["stale"],
                trap_invoice["issue"], trap_invoice["pay"],
                trap_invoice["pay"]))
            items.append(ContextItem(
                "inv-trap-b", "invoice_due", trap_invoice["corrected"],
                trap_invoice["known_correction"], trap_invoice["pay"],
                trap_invoice["pay"], revises="inv-trap-a", trap=True))
        elif rng.random() < CONFIG["post_cutoff_correction_share"]:
            # Irreducible: a correction known only after the cutoff.
            # It does not change the realized path here (the book was
            # already final); it exists so leakage has a target to miss.
            future_invoices = [item for item in items
                               if item.kind == "invoice_due"
                               and item.effective_from > HISTORY]
            if future_invoices:
                target = future_invoices[0]
                post_cutoff_corrections += 1
                items.append(ContextItem(
                    target.item_id + "-post", "invoice_due",
                    target.value * rng.uniform(0.5, 0.9),
                    HISTORY + rng.randrange(1, HORIZON),
                    target.effective_from, target.effective_to,
                    revises=target.item_id))

        a = rng.uniform(0.6, 2.4)
        b = rng.uniform(1000, 20000) - a * statistics.median(history)
        shown_history = tuple(round(a * v + b, 4) for v in history)
        shown_future = tuple(round(a * v + b, 4) for v in future)
        shown_stale_future = [round(a * v + b, 4) for v in stale_future]
        shown_floor = round(a * floor + b, 4)

        def transform(item: ContextItem) -> ContextItem:
            value = (round(a * item.value + b, 4)
                     if item.kind == "cash_floor"
                     else round(a * item.value, 4))
            return ContextItem(item.item_id, item.kind, value,
                               item.known_at, item.effective_from,
                               item.effective_to, item.revises,
                               item.text_only, item.trap, item.aux)

        shown_items = tuple(transform(item) for item in items)
        shortfall_steps = [step for step, value
                           in enumerate(shown_future, 1)
                           if value < shown_floor]
        expected_event = cell == "shortfall" or (
            cell == "trap" and min(future) < floor)
        if bool(shortfall_steps) != expected_event:
            skipped["rounding_flip"] += 1
            continue
        trap_optimal = stale_optimal = None
        if cell == "trap":
            stale_event = any(value < shown_floor
                              for value in shown_stale_future)
            if stale_event == bool(shortfall_steps):
                skipped["rounding_flip"] += 1
                continue
            trap_optimal = {"action": "act" if shortfall_steps
                            else "monitor"}
            stale_optimal = {"action": "act" if stale_event
                             else "monitor"}
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        cases.append(Case(
            case_id=f"cf{seed}-{len(cases):04d}", domain="cashflow",
            frequency="D", values=shown_history, future=shown_future,
            horizon=HORIZON, items=shown_items, threshold=shown_floor,
            trap=cell == "trap", trap_optimal=trap_optimal,
            stale_optimal=stale_optimal,
            series_id=f"treasury-{len(cases):04d}",
            meta={"truth_event": bool(shortfall_steps),
                  "truth_first_step": (shortfall_steps[0]
                                       if shortfall_steps else None),
                  "outcome_cell": cell}))
    if len(cases) < count:
        raise ValueError(
            f"cashflow: only {len(cases)}/{count} cases after "
            f"{attempts} attempts; skipped={skipped}")
    provenance = {
        "generator": "mechanistic_invoice_driven_balance",
        "config": CONFIG,
        "attempts": attempts, "skipped": skipped,
        "outcome_distribution": dict(sorted(cell_counts.items())),
        "trap_share": cell_counts.get("trap", 0) / len(cases),
        "shortfall_base_rate": statistics.mean(
            case.meta["truth_event"] for case in cases),
        "irreducible_post_cutoff_corrections": post_cutoff_corrections,
        "cases_per_series": {case.series_id: 1 for case in cases},
        "independence": (
            "one independent simulated series per case; futures cannot "
            "overlap; labels can still co-move through shared parameter "
            "ranges — a caveat, not an independence claim"),
        "anonymization": (
            "per_case_seeded_positive_affine;balances_and_floor_ax_plus_b;"
            "flows_ax;decision_invariance_verified_on_rounded_shown_numbers"),
    }
    return cases, provenance


def _engine_inputs(case: Case, facts: list[ContextItem]) -> dict[str, Any]:
    floor = next((item.value for item in facts
                  if item.kind == "cash_floor"), min(case.values))
    known_inflows = sum(
        item.value for item in facts
        if item.kind == "invoice_due"
        and case.cutoff < item.effective_from <= case.cutoff + case.horizon)
    floor_adjusted = floor - known_inflows
    return {"series": [-float(v) for v in case.values],
            "threshold": -floor_adjusted,
            "floor_adjusted": floor_adjusted,
            "basis": "negated_balance_below_floor_with_known_inflow_adjustment"}


def _question(case: Case) -> str:
    return (
        "Decision: will the cash balance cross below the minimum-balance "
        "floor (the cash_floor fact) at any point in the unshown "
        "horizon? Invoice payments listed in the context arrive on their "
        "effective date in the version known as of the cutoff. Acting "
        "(drawing the credit line) costs the carry and fully covers any "
        "shortfall; not drawing costs nothing unless the floor is "
        "crossed.")


PACK = DomainPack(
    name="cashflow",
    version="0.1",
    decision_kind="binary",
    simulate=simulate,
    cost_model=binary_cost_model(COST_DRAW, COST_SHORTFALL,
                                 "credit_line_carry", "shortfall"),
    decision_schema=binary_decision_schema(HORIZON),
    context_kinds={
        "cash_floor": {"unit": "currency_level",
                       "bounds": (-10_000_000.0, 10_000_000.0),
                       "max_span": 400},
        "invoice_due": {"unit": "currency_inflow",
                        "bounds": (0.0, 5_000_000.0), "max_span": 3},
    },
    question=_question,
    engine_inputs=_engine_inputs,
    engine_decision=lambda case, packet, inputs: governed_engine_decision(
        case, packet, COST_DRAW, COST_SHORTFALL),
    decision_from_forecast=lambda case, path, inputs: crossing_decision(
        case, path, inputs["floor_adjusted"], above=False),
    constant_policies=lambda case: {
        "always_act": {"action": "act", "event_expected": True,
                       "first_event_step": 1},
        "never_act": {"action": "monitor", "event_expected": False,
                      "first_event_step": None}},
    parse_decision=parse_binary_decision,
    decision_scalar=lambda decision: 1.0
    if decision.get("action") == "act" else 0.0,
    config=CONFIG,
    season_length=7,
)

register(PACK)
