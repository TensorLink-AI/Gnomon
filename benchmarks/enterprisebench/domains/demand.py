"""Intermittent SKU demand with promotions, holidays, and stockout
censoring: choose an order-up-to quantity per SKU and a coherent
category total.

Series: the shown series is daily *category sales* (the sum of two SKU
sales series, both shown in the question block). Sales are not demand:
a periodic replenishment policy runs in history, and when a SKU stocks
out, sales are censored at available stock — the stockout days are
disclosed per SKU, and scoring uses the uncensored realized demand, so
an arm that reads sales as demand under-orders exactly the way real
replenishment systems do.

Decision: order-up-to quantities for the horizon, per SKU plus a
category total. Costs per unit: stockout 9 ≫ holding 1, so the critical
fractile is 0.9 — ordering the naive point sum systematically loses to
an arm that carries uncertainty. Hierarchical coherence
(|category − ΣSKU|) is scored as a secondary metric on every arm.

Context facts: promotions (`promo_uplift`, percent, dated window on one
SKU — they multiply demand mechanistically in history and future),
holiday spikes (`holiday_spike`, percent, single day, category-wide).
Trap flavor (~15%, disclosed): a promo is *rescheduled* after
announcement — same uplift, moved window, revised before the cutoff;
the realized future runs on the new dates and the counterfactual future
on the stale dates differs enough to flip the optimal order materially.

No affine anonymization: quantities are counts and the decision is in
the same units (disclosed; the corpus is synthetic, so memorization is
not a threat and MASE remains the affine-invariant secondary).

Parameters and grounding: base demand rates 4–30 units/day with
zero-inflation 10–35% mirror intermittent mid-tail SKUs; promo uplifts
of 40–150% decaying over one to three weeks match observed promotion
lifts; weekly seasonality peaks on weekends; replenishment every 7 days
to a 1.2–1.6× cover level produces realistic stockout episodes rather
than wall-to-wall censoring.
"""

from __future__ import annotations

import json
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
    seasonal_naive_path,
)
from benchmarks.enterprisebench.textgen import register_templates

HISTORY = 112
HORIZON = 14
SKUS = ("sku_a", "sku_b")
COST_HOLDING = 1.0
COST_STOCKOUT = 9.0
CONFIG: dict[str, Any] = {
    "history": HISTORY, "horizon": HORIZON, "frequency": "D",
    "units": "units_holding_1_stockout_9_per_unit",
    "skus": list(SKUS),
    "base_rate": (4.0, 30.0), "zero_inflation": (0.10, 0.35),
    "weekend_uplift": (0.1, 0.5),
    "promo_uplift_percent": (40.0, 150.0), "promo_length": (7, 21),
    "holiday_spike_percent": (30.0, 120.0),
    "replenish_every": 7, "cover_factor": (1.2, 1.6),
    "shock_day_probability": 0.02, "shock_multiplier": (2.0, 4.0),
    "trap_total_shift_min_fraction": 0.12,
    "outcome_targets": {"plain": 0.85, "trap": 0.15},
}


def _sku_demand(rng: random.Random, length: int, base: float,
                zero_p: float, weekend: float,
                promos: list[dict[str, Any]],
                holidays: list[dict[str, Any]]) -> list[int]:
    values = []
    for step in range(length):
        rate = base * (1.0 + weekend if step % 7 >= 5 else 1.0)
        for promo in promos:
            if promo["from"] <= step <= promo["to"]:
                age = step - promo["from"]
                decay = max(0.25, 1.0 - age / max(1, promo["length"]))
                rate *= 1.0 + promo["uplift"] / 100.0 * decay
        for holiday in holidays:
            if step == holiday["step"]:
                rate *= 1.0 + holiday["uplift"] / 100.0
        if rng.random() < zero_p:
            values.append(0)
        else:
            quantity = rng.gauss(rate, max(1.0, rate ** 0.5))
            # Rare demand shocks (a viral mention, a weather day):
            # real intermittent series carry occasional multiples of
            # the base rate that Gaussian noise never produces.
            if rng.random() < CONFIG["shock_day_probability"]:
                quantity *= rng.uniform(*CONFIG["shock_multiplier"])
            values.append(max(0, round(quantity)))
    return values


def _censor(demand: list[int], rng: random.Random,
            cover: float) -> tuple[list[int], list[int]]:
    """Replenish every 7 days to mean-demand * cover; sales are censored
    at available stock. Returns (sales, stockout_steps)."""
    typical = max(1.0, statistics.mean(demand[:28]) if demand else 1.0)
    level = typical * CONFIG["replenish_every"] * cover
    stock = level
    sales, stockouts = [], []
    for step, quantity in enumerate(demand):
        if step % CONFIG["replenish_every"] == 0:
            stock = level
        sold = min(quantity, int(stock))
        if sold < quantity:
            stockouts.append(step)
        stock -= sold
        sales.append(sold)
    return sales, stockouts


def _optimal_orders(demand_totals: dict[str, int]) -> dict[str, Any]:
    orders = {sku: float(total) for sku, total in demand_totals.items()}
    return {"orders": orders,
            "category_total": float(sum(demand_totals.values()))}


def simulate(seed: int, count: int) -> tuple[list[Case], dict[str, Any]]:
    caps = {cell: int(count * fraction) + 1
            for cell, fraction in CONFIG["outcome_targets"].items()}
    cases: list[Case] = []
    cell_counts: dict[str, int] = {}
    skipped = {"cell_full": 0, "trap_shape": 0, "degenerate": 0}
    attempts = 0
    balanced_limit = 300 * count
    length = HISTORY + HORIZON
    while len(cases) < count and attempts < 2 * balanced_limit:
        attempts += 1
        balanced_phase = attempts <= balanced_limit
        rng = random.Random(f"enterprisebench:demand:{seed}:{attempts}")
        trap = balanced_phase and cell_counts.get("trap", 0) < caps["trap"]
        cell = "trap" if trap else "plain"
        if balanced_phase and cell_counts.get(cell, 0) >= caps[cell]:
            skipped["cell_full"] += 1
            continue

        holidays = ([{"step": rng.randrange(HISTORY - 30, length - 1),
                      "uplift": rng.uniform(
                          *CONFIG["holiday_spike_percent"])}]
                    if rng.random() < 0.5 else [])
        promos: dict[str, list[dict[str, Any]]] = {sku: [] for sku in SKUS}
        items: list[ContextItem] = []
        trap_detail = None
        if trap:
            # A promo announced for one window, rescheduled to another
            # before the cutoff. Realized demand runs on the new dates.
            uplift = rng.uniform(80.0, 150.0)
            promo_length = rng.randint(10, 18)
            stale_from = rng.randrange(HISTORY + 2, length - 4)
            move = rng.choice((-1, 1)) * rng.randint(
                HORIZON, HORIZON + 10)
            new_from = stale_from + move
            if not HISTORY - 20 <= new_from <= length - 4:
                new_from = stale_from - move
            if not 0 <= new_from <= length - 2:
                skipped["trap_shape"] += 1
                continue
            announced = rng.randrange(HISTORY // 2, HISTORY - 10)
            revised = rng.randrange(announced + 3, HISTORY)
            trap_detail = {"uplift": uplift, "length": promo_length,
                           "stale_from": stale_from, "new_from": new_from,
                           "announced": announced, "revised": revised}
            promos["sku_a"].append({
                "from": new_from, "to": new_from + promo_length,
                "uplift": uplift, "length": promo_length})
        elif rng.random() < 0.6:
            sku = SKUS[rng.randrange(len(SKUS))]
            uplift = rng.uniform(*CONFIG["promo_uplift_percent"])
            promo_length = rng.randint(*CONFIG["promo_length"])
            start = rng.randrange(HISTORY // 2, length - 4)
            promos[sku].append({"from": start, "to": start + promo_length,
                                "uplift": uplift, "length": promo_length})
            items.append(ContextItem(
                f"promo-{sku}", "promo_uplift", uplift,
                max(0, start - rng.randrange(3, 12)), start,
                min(start + promo_length, length - 1),
                aux=(("sku", sku),)))

        demand: dict[str, list[int]] = {}
        sales: dict[str, list[int]] = {}
        stockouts: dict[str, list[int]] = {}
        stale_demand: dict[str, list[int]] = {}
        for index, sku in enumerate(SKUS):
            sku_rng = random.Random(
                f"enterprisebench:demand:{seed}:{attempts}:{sku}")
            base = sku_rng.uniform(*CONFIG["base_rate"])
            zero_p = sku_rng.uniform(*CONFIG["zero_inflation"])
            weekend = sku_rng.uniform(*CONFIG["weekend_uplift"])
            demand[sku] = _sku_demand(sku_rng, length, base, zero_p,
                                      weekend, promos[sku], holidays)
            if trap_detail is not None:
                # Counterfactual with identical draws under stale dates.
                stale_rng = random.Random(
                    f"enterprisebench:demand:{seed}:{attempts}:{sku}")
                stale_promos = ([{
                    "from": trap_detail["stale_from"],
                    "to": trap_detail["stale_from"]
                    + trap_detail["length"],
                    "uplift": trap_detail["uplift"],
                    "length": trap_detail["length"]}]
                    if sku == "sku_a" else [])
                stale_demand[sku] = _sku_demand(
                    stale_rng, length, base, zero_p, weekend,
                    stale_promos, holidays)
            censor_rng = random.Random(
                f"enterprisebench:demand:{seed}:{attempts}:{sku}:stock")
            sales[sku], stockouts[sku] = _censor(
                demand[sku], censor_rng,
                censor_rng.uniform(*CONFIG["cover_factor"]))

        if trap_detail is not None:
            real_total = sum(sum(demand[sku][HISTORY:]) for sku in SKUS)
            stale_total = sum(sum(stale_demand[sku][HISTORY:])
                              for sku in SKUS)
            if real_total <= 0 or abs(real_total - stale_total) < max(
                    8.0, CONFIG["trap_total_shift_min_fraction"]
                    * real_total):
                skipped["trap_shape"] += 1
                continue
            items.append(ContextItem(
                "promo-a", "promo_uplift", trap_detail["uplift"],
                trap_detail["announced"], trap_detail["stale_from"],
                min(trap_detail["stale_from"] + trap_detail["length"],
                    length - 1),
                aux=(("sku", "sku_a"),)))
            items.append(ContextItem(
                "promo-b", "promo_uplift", trap_detail["uplift"],
                trap_detail["revised"], trap_detail["new_from"],
                min(trap_detail["new_from"] + trap_detail["length"],
                    length - 1),
                revises="promo-a", trap=True,
                aux=(("sku", "sku_a"),)))
        for index, holiday in enumerate(holidays):
            items.append(ContextItem(
                f"holiday-{index}", "holiday_spike", holiday["uplift"],
                max(0, holiday["step"] - rng.randrange(10, 25)),
                holiday["step"], holiday["step"],
                text_only=rng.random() < 0.4))

        category_sales = [sum(sales[sku][step] for sku in SKUS)
                          for step in range(length)]
        history = tuple(float(v) for v in category_sales[:HISTORY])
        if max(history) == min(history):
            skipped["degenerate"] += 1
            continue
        future = tuple(float(v) for v in category_sales[HISTORY:])
        demand_totals = {sku: sum(demand[sku][HISTORY:]) for sku in SKUS}
        trap_optimal = stale_optimal = None
        if trap_detail is not None:
            trap_optimal = _optimal_orders(demand_totals)
            stale_optimal = _optimal_orders(
                {sku: sum(stale_demand[sku][HISTORY:]) for sku in SKUS})
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        cases.append(Case(
            case_id=f"dm{seed}-{len(cases):04d}", domain="demand",
            frequency="D", values=history, future=future,
            horizon=HORIZON, items=tuple(items), threshold=None,
            trap=trap_detail is not None, trap_optimal=trap_optimal,
            stale_optimal=stale_optimal,
            series_id=f"category-{len(cases):04d}",
            meta={
                "truth_event": any(
                    stockouts[sku] and stockouts[sku][-1] >= HISTORY
                    for sku in SKUS),
                "truth_first_step": None,
                "outcome_cell": cell,
                "demand_totals": demand_totals,
                "sku_sales": {sku: sales[sku][:HISTORY] for sku in SKUS},
                "sku_stockout_steps": {
                    sku: [s for s in stockouts[sku] if s < HISTORY]
                    for sku in SKUS},
                "extra_futures": {
                    f"{sku}_demand": demand[sku][HISTORY:]
                    for sku in SKUS},
            }))
    if len(cases) < count:
        raise ValueError(
            f"demand: only {len(cases)}/{count} cases after "
            f"{attempts} attempts; skipped={skipped}")
    provenance = {
        "generator": "mechanistic_intermittent_demand_with_censoring",
        "config": CONFIG,
        "attempts": attempts, "skipped": skipped,
        "outcome_distribution": dict(sorted(cell_counts.items())),
        "trap_share": cell_counts.get("trap", 0) / len(cases),
        "future_stockout_rate": statistics.mean(
            case.meta["truth_event"] for case in cases),
        "cases_per_series": {case.series_id: 1 for case in cases},
        "independence": (
            "one independent simulated series per case; futures cannot "
            "overlap; labels can still co-move through shared parameter "
            "ranges — a caveat, not an independence claim"),
        "anonymization": (
            "per_case_seeded_positive_affine_not_applied:counts_domain;"
            "synthetic_corpus_no_memorization_surface;disclosed"),
    }
    return cases, provenance


def _score(decision: dict[str, Any], case: Case) -> dict[str, float]:
    totals = case.meta["demand_totals"]
    cost = 0.0
    for sku in SKUS:
        order = float(decision["orders"].get(sku, 0.0))
        realized = float(totals[sku])
        cost += COST_HOLDING * max(0.0, order - realized)
        cost += COST_STOCKOUT * max(0.0, realized - order)
    return {"cost": cost, "regret": cost}


def _no_action(case: Case) -> dict[str, Any]:
    """The ERP's default: naive seasonal replenishment of shown sales."""
    orders = {}
    for sku in SKUS:
        path = seasonal_naive_path(case.meta["sku_sales"][sku], 7,
                                   case.horizon)
        orders[sku] = float(sum(path))
    return {"orders": orders,
            "category_total": float(sum(orders.values()))}


def _promo_multiplier(case: Case, facts: list[ContextItem]) -> float:
    """Known promo/holiday uplift over the horizon, decay-averaged, as a
    category-level multiplier on expected demand."""
    boost = 0.0
    for item in facts:
        if item.kind not in ("promo_uplift", "holiday_spike"):
            continue
        overlap_from = max(item.effective_from, case.cutoff)
        overlap_to = min(item.effective_to, case.cutoff + case.horizon - 1)
        if overlap_to < overlap_from:
            continue
        overlap = (overlap_to - overlap_from + 1) / case.horizon
        share = 0.5 if item.kind == "promo_uplift" else 1.0
        boost += (item.value / 100.0) * overlap * 0.5 * share
    return 1.0 + boost


def _split_by_share(case: Case, category_total: float) -> dict[str, Any]:
    trailing = {sku: sum(case.meta["sku_sales"][sku][-28:])
                for sku in SKUS}
    denominator = max(1.0, float(sum(trailing.values())))
    orders = {sku: round(category_total * trailing[sku] / denominator, 2)
              for sku in SKUS}
    return {"orders": orders,
            "category_total": round(float(sum(orders.values())), 2)}


def _engine_inputs(case: Case, facts: list[ContextItem]) -> dict[str, Any]:
    return {"threshold": None, "basis": "shown_series",
            "promo_multiplier": _promo_multiplier(case, facts)}


def _engine_decision(case: Case, packet: dict[str, Any],
                     inputs: dict[str, Any]) -> dict[str, Any]:
    """Order-up-to at the critical fractile (0.9): the q90 path sum,
    lifted by the as-of known promo multiplier, split across SKUs by
    trailing sales share (coherent by construction)."""
    rows = packet.get("forecast") or []
    if len(rows) != case.horizon:
        return _no_action(case)
    q90_sum = sum(float(row["q90"]) for row in rows)
    return _split_by_share(case, q90_sum * inputs["promo_multiplier"])


def _decision_from_forecast(case: Case, path: list[float],
                            inputs: dict[str, Any]) -> dict[str, Any]:
    return _split_by_share(
        case, sum(float(v) for v in path) * inputs["promo_multiplier"])


def _parse_decision(payload: dict[str, Any],
                    case: Case) -> dict[str, Any] | None:
    orders_raw = payload.get("orders")
    if not isinstance(orders_raw, dict):
        return None
    orders = {}
    for sku in SKUS:
        value = orders_raw.get(sku)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            return None
        orders[sku] = float(value)
    total = payload.get("category_total")
    if isinstance(total, bool) or not isinstance(total, (int, float)) \
            or not math.isfinite(total) or total < 0:
        return None
    return {"orders": orders, "category_total": float(total)}


def _question(case: Case) -> str:
    sku_blocks = []
    for sku in SKUS:
        sku_blocks.append(
            f"{sku} sales oldest first: "
            + json.dumps(case.meta["sku_sales"][sku],
                         separators=(",", ":"))
            + f"; stockout days (sales were censored at available "
            f"stock, demand ran higher): "
            + json.dumps(case.meta["sku_stockout_steps"][sku]))
    return (
        "Per-SKU detail (the shown series above is their category sum):\n"
        + "\n".join(sku_blocks) + "\n"
        "Decision: set order-up-to quantities covering total demand "
        "(not sales) over the unshown horizon for each SKU, plus the "
        "category total, which must equal the sum of the SKU orders. "
        "Each unit ordered above realized demand costs holding; each "
        "unit of realized demand above the order costs stockout.")


def _extra_metrics(decision: dict[str, Any],
                   case: Case) -> dict[str, float]:
    total = float(decision.get("category_total", 0.0))
    sku_sum = float(sum(decision.get("orders", {}).get(sku, 0.0)
                        for sku in SKUS))
    return {"coherence_error": abs(total - sku_sum)}


PACK = DomainPack(
    name="demand",
    version="0.2",
    decision_kind="quantity",
    simulate=simulate,
    cost_model=CostModel(
        names={"holding_per_unit": COST_HOLDING,
               "stockout_per_unit": COST_STOCKOUT},
        break_even=COST_STOCKOUT / (COST_STOCKOUT + COST_HOLDING),
        score=_score,
        no_action=_no_action,
        optimal=lambda case: _optimal_orders(case.meta["demand_totals"]),
    ),
    decision_schema={
        "instruction": (
            'Return {"orders": {"sku_a": <units>, "sku_b": <units>}, '
            '"category_total": <units>}.'),
        "fields": {"orders": "per-SKU non-negative units",
                   "category_total": "sum of SKU orders"},
    },
    context_kinds={
        "promo_uplift": {"unit": "percent", "bounds": (0.0, 500.0),
                         "max_span": 45},
        "holiday_spike": {"unit": "percent", "bounds": (0.0, 500.0),
                          "max_span": 3},
    },
    question=_question,
    engine_inputs=_engine_inputs,
    engine_decision=_engine_decision,
    decision_from_forecast=_decision_from_forecast,
    constant_policies=lambda case: {
        "order_zero": {"orders": {sku: 0.0 for sku in SKUS},
                       "category_total": 0.0},
        "order_double_naive": {
            "orders": {sku: 2.0 * value for sku, value
                       in _no_action(case)["orders"].items()},
            "category_total": 2.0 * _no_action(case)["category_total"]},
    },
    parse_decision=_parse_decision,
    decision_scalar=lambda decision: float(
        decision.get("category_total", 0.0)),
    config=CONFIG,
    season_length=7,
    extra_metrics=_extra_metrics,
)

register(PACK)

register_templates("promo_uplift", base=(
    "Marketing brief {ref}: the {sku} promotion runs {from_date} "
    "through {to_date}; planning assumes roughly {value} percent lift "
    "at launch, decaying over the window.",
    "Trade calendar ({ref}): {sku} promo scheduled {from_date} to "
    "{to_date} with an expected {value} percent uplift.",
), revision=(
    "Promo reschedule {ref}: the {sku} campaign originally announced "
    "for other dates now runs {from_date} through {to_date}; the "
    "expected lift stays near {value} percent.",
    "Updated trade note ({ref}): {sku} promotion moved — new window "
    "{from_date} to {to_date}, uplift still about {value} percent.",
))
register_templates("holiday_spike", base=(
    "Seasonal note {ref}: the holiday on {from_date} typically adds "
    "around {value} percent across the category.",
    "Calendar memo ({ref}): expect a one-day spike near {value} "
    "percent on {from_date}.",
))
