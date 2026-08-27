"""Cloud spend vs committed budget: act (rightsize/alert) or monitor.

Series: daily spend for one account — a base level with a weekday cycle,
mild noise, and causal context effects (deploy uplifts, announced
migrations that ramp spend down). The commit level is context, not part
of the series: a ``commit_base`` fact plus zero or more ``commit_change``
deltas, each dated and revisable. The decision threshold is whatever the
commit facts resolve to *as of the cutoff* — deriving it is part of the
job.

Costs: an overage (missed breach) costs far more than an intervention,
so the break-even probability is intervention/overage = 0.2, and the
breach base rate is held near it (verified at generation, achieved mix
disclosed) so constant policies cannot masquerade as skill.

Trap flavor (~15%, disclosed): a ``commit_change`` is revised before the
cutoff and the revision moves the effective threshold across the
realized future's peak — the stale version and the correct version imply
opposite optimal decisions, by construction.

Simulator parameters and their real-world grounding: base spend spans
small-team to mid-size accounts (800–6000 $/day); weekday load is 10–35%
above weekends, matching business-hours compute; deploy uplifts of
5–25% for one to four weeks mirror feature-launch capacity bumps;
migration ramp-downs of 15–45% over three weeks mirror announced
workload moves. All shown numbers pass per-case seeded positive affine
anonymization (levels ``a*x+b``, deltas ``a*x``, percentages untouched);
decision structure is verified invariant on the rounded shown numbers
and flipping cases are discarded.
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

HISTORY = 112
HORIZON = 14
COST_INTERVENE = 3.0
COST_OVERAGE = 15.0
CONFIG: dict[str, Any] = {
    "history": HISTORY, "horizon": HORIZON, "frequency": "D",
    "units": "cost_units_intervention_3_overage_15",
    "base_spend": (800.0, 6000.0), "weekday_uplift": (0.10, 0.35),
    "noise_sigma_fraction": 0.04, "trend_per_step_fraction": (-.0015, .002),
    "deploy_uplift_fraction": (0.05, 0.25), "deploy_duration": (7, 28),
    "migration_depth_fraction": (0.15, 0.45), "migration_ramp": 21,
    "outcome_targets": {"no_breach": 0.55, "breach": 0.30, "trap": 0.15},
}


def _spend_series(rng: random.Random, length: int,
                  effects: list[dict[str, Any]]) -> list[float]:
    """The mechanistic generator: every context effect acts causally on
    the series, history and future alike, so ground truth is known."""
    base = rng.uniform(*CONFIG["base_spend"])
    weekday_uplift = rng.uniform(*CONFIG["weekday_uplift"])
    trend = rng.uniform(*CONFIG["trend_per_step_fraction"]) * base
    sigma = CONFIG["noise_sigma_fraction"] * base
    values = []
    for step in range(length):
        level = base + trend * step
        if step % 7 < 5:
            level *= 1.0 + weekday_uplift
        for effect in effects:
            if effect["kind"] == "deploy" and \
                    effect["from"] <= step <= effect["to"]:
                level += effect["uplift"]
            if effect["kind"] == "migration" and step >= effect["from"]:
                ramp = min(1.0, (step - effect["from"] + 1)
                           / CONFIG["migration_ramp"])
                level *= 1.0 - effect["depth"] * ramp
        values.append(max(1.0, level + rng.gauss(0.0, sigma)))
    return values


def _robust_scale(values: list[float]) -> float:
    diffs = [abs(right - left) for left, right in zip(values, values[1:])]
    return max(statistics.median(diffs), 1e-6) if diffs else 1.0


def _threshold_from_items(items: list[ContextItem]) -> float | None:
    base = None
    delta = 0.0
    for item in items:
        if item.kind == "commit_base":
            base = item.value
        elif item.kind == "commit_change":
            delta += item.value
    return None if base is None else base + delta


def simulate(seed: int, count: int) -> tuple[list[Case], dict[str, Any]]:
    targets = CONFIG["outcome_targets"]
    caps = {cell: int(count * fraction) + 1
            for cell, fraction in targets.items()}
    cases: list[Case] = []
    cell_counts: dict[str, int] = {}
    skipped = {"cell_full": 0, "rounding_flip": 0, "degenerate": 0}
    attempts = 0
    balanced_limit = 200 * count
    while len(cases) < count and attempts < 2 * balanced_limit:
        attempts += 1
        balanced_phase = attempts <= balanced_limit
        rng = random.Random(f"enterprisebench:cloudcost:{seed}:{attempts}")
        length = HISTORY + HORIZON

        effects: list[dict[str, Any]] = []
        items: list[ContextItem] = []
        probe = random.Random(rng.random())
        if rng.random() < 0.7:
            start = rng.randrange(HISTORY // 3, length - 3)
            uplift_fraction = rng.uniform(*CONFIG["deploy_uplift_fraction"])
            effects.append({"kind": "deploy", "from": start,
                            "to": start + rng.randint(
                                *CONFIG["deploy_duration"]),
                            "uplift_fraction": uplift_fraction})
        if rng.random() < 0.25:
            start = rng.randrange(HISTORY // 2, length - 5)
            effects.append({"kind": "migration", "from": start,
                            "depth": rng.uniform(
                                *CONFIG["migration_depth_fraction"])})
        # Resolve fractional uplifts against a probe draw of the base so
        # effect sizes are in currency, then simulate for real.
        probe_base = probe.uniform(*CONFIG["base_spend"])
        for effect in effects:
            if effect["kind"] == "deploy":
                effect["uplift"] = effect.pop("uplift_fraction") * probe_base
        values = _spend_series(rng, length, effects)
        history, future = values[:HISTORY], values[HISTORY:]
        if max(history) == min(history):
            skipped["degenerate"] += 1
            continue
        scale = _robust_scale(history)
        peak = max(future)

        trap = balanced_phase and cell_counts.get("trap", 0) < caps["trap"]
        if trap:
            # Construct the flip: the stale delta puts the threshold on
            # one side of the realized peak, the revision on the other.
            breach_as_of = rng.random() < 0.5
            margin = rng.uniform(0.8, 2.5) * scale
            threshold_as_of = peak - margin if breach_as_of \
                else peak + margin
            threshold_stale = peak + margin if breach_as_of \
                else peak - margin
            base_level = min(threshold_as_of, threshold_stale) \
                - rng.uniform(0.5, 3.0) * scale
            known_v0 = rng.randrange(HISTORY // 2, HISTORY - 14)
            known_v1 = rng.randrange(known_v0 + 3, HISTORY)
            items.append(ContextItem(
                "commit-base", "commit_base", base_level, 0, 0, length - 1))
            items.append(ContextItem(
                "commit-chg-a", "commit_change",
                threshold_stale - base_level, known_v0,
                min(HISTORY, known_v0 + 7), length - 1))
            items.append(ContextItem(
                "commit-chg-b", "commit_change",
                threshold_as_of - base_level, known_v1,
                min(HISTORY, known_v0 + 7), length - 1,
                revises="commit-chg-a", trap=True))
            cell = "trap"
        else:
            breach = (cell_counts.get("breach", 0) < caps["breach"]
                      if balanced_phase else rng.random() < 0.3)
            margin = (rng.uniform(0.2, 1.5) if breach
                      else rng.uniform(0.3, 4.0)) * scale
            threshold = peak - margin if breach else peak + margin
            breach_as_of = breach
            if rng.random() < 0.5:
                delta = rng.uniform(-1.5, 1.5) * scale
                base_level = threshold - delta
                known = rng.randrange(HISTORY // 2, HISTORY)
                items.append(ContextItem(
                    "commit-base", "commit_base", base_level,
                    0, 0, length - 1))
                items.append(ContextItem(
                    "commit-chg-a", "commit_change", delta, known,
                    min(HISTORY, known + 7), length - 1))
            else:
                items.append(ContextItem(
                    "commit-base", "commit_base", threshold,
                    0, 0, length - 1))
            cell = "breach" if breach else "no_breach"
        if balanced_phase and cell_counts.get(cell, 0) >= caps[cell]:
            skipped["cell_full"] += 1
            continue

        for index, effect in enumerate(effects):
            known = max(0, effect["from"] - rng.randrange(2, 10))
            if effect["kind"] == "deploy":
                # Half the deploy notices exist only as memos
                # (disclosed): the structured record never sees them,
                # and only extraction can recover them.
                items.append(ContextItem(
                    f"deploy-{index}", "deploy_uplift", effect["uplift"],
                    known, effect["from"],
                    min(effect["to"], length - 1),
                    text_only=rng.random() < 0.5))
            else:
                items.append(ContextItem(
                    f"migration-{index}", "migration_reduction",
                    effect["depth"] * 100.0, known, effect["from"],
                    length - 1))

        # Post-cutoff noise the as-of resolver must excise: a late
        # correction to the commit record nobody could have known at the
        # cutoff. It never changes the truth (decisions are scored
        # against the commitment in effect when the decision was made);
        # an arm that follows it anyway is exhibiting leakage.
        if rng.random() < 0.4:
            changes = [item for item in items
                       if item.kind == "commit_change"]
            target = changes[-1] if changes else None
            if target is not None:
                by_id = {item.item_id: item for item in items}
                stale = by_id.get(target.revises or "")
                if stale is not None:
                    # On trap chains the late correction reverts toward
                    # the superseded figure — a hidden reversal, the
                    # marker the trap-integrity split measures leakage
                    # against.
                    post_value = stale.value + rng.uniform(0.1, 0.4) \
                        * (target.value - stale.value)
                else:
                    post_value = target.value * rng.uniform(0.4, 0.9)
                items.append(ContextItem(
                    target.item_id + "-post", "commit_change",
                    post_value, HISTORY + rng.randrange(1, HORIZON),
                    target.effective_from, length - 1,
                    revises=target.item_id))

        # Per-case seeded positive affine anonymization: levels a*x+b,
        # deltas/uplifts a*x, percentages untouched. Decision structure
        # verified on the rounded shown numbers below.
        a = rng.uniform(0.6, 2.4)
        b = rng.uniform(50, 900) - a * statistics.median(history)
        shown_history = tuple(round(a * v + b, 4) for v in history)
        shown_future = tuple(round(a * v + b, 4) for v in future)

        def transform(item: ContextItem) -> ContextItem:
            if item.kind == "commit_base":
                value = round(a * item.value + b, 4)
            elif item.kind in ("commit_change", "deploy_uplift"):
                value = round(a * item.value, 4)
            else:
                value = round(item.value, 4)
            return ContextItem(item.item_id, item.kind, value,
                               item.known_at, item.effective_from,
                               item.effective_to, item.revises,
                               item.text_only, item.trap, item.aux)

        shown_items = tuple(transform(item) for item in items)
        resolved = as_of(shown_items, HISTORY)
        shown_threshold = _threshold_from_items(resolved)
        breach_steps = [step for step, value
                        in enumerate(shown_future, 1)
                        if value > shown_threshold]
        if bool(breach_steps) != breach_as_of:
            skipped["rounding_flip"] += 1
            continue
        trap_optimal = stale_optimal = None
        if cell == "trap":
            stale_threshold = _threshold_from_items([
                item for item in as_of(
                    tuple(item for item in shown_items
                          if item.item_id != "commit-chg-b"), HISTORY)])
            stale_breach = any(value > stale_threshold
                               for value in shown_future)
            if stale_breach == bool(breach_steps):
                skipped["rounding_flip"] += 1
                continue
            trap_optimal = {"action": "act" if breach_steps else "monitor"}
            stale_optimal = {"action": "act" if stale_breach
                             else "monitor"}
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        case = Case(
            case_id=f"cc{seed}-{len(cases):04d}", domain="cloudcost",
            frequency="D", values=shown_history, future=shown_future,
            horizon=HORIZON, items=shown_items,
            threshold=shown_threshold, trap=cell == "trap",
            trap_optimal=trap_optimal, stale_optimal=stale_optimal,
            series_id=f"account-{len(cases):04d}",
            meta={"truth_event": bool(breach_steps),
                  "truth_first_step": (breach_steps[0]
                                       if breach_steps else None),
                  "outcome_cell": cell})
        cases.append(case)
    if len(cases) < count:
        raise ValueError(
            f"cloudcost: only {len(cases)}/{count} cases after "
            f"{attempts} attempts; skipped={skipped}")
    provenance = {
        "generator": "mechanistic_spend_with_causal_context_effects",
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
            "per_case_seeded_positive_affine;levels_ax_plus_b;"
            "deltas_ax;percent_untouched;"
            "decision_invariance_verified_on_rounded_shown_numbers"),
    }
    return cases, provenance


def _engine_inputs(case: Case, facts: list[ContextItem]) -> dict[str, Any]:
    threshold = _threshold_from_items(facts)
    if threshold is None:
        threshold = max(case.values)
    return {"threshold": threshold, "basis": "shown_series"}


def _question(case: Case) -> str:
    return (
        "Decision: will daily spend breach the committed budget level in "
        "effect over the unshown horizon? The effective commit level is "
        "the commit_base plus every commit_change, each taken in the "
        "version known as of the cutoff. Acting (rightsize "
        "and alert) costs the intervention and fully mitigates any "
        "overage; monitoring costs nothing unless a breach occurs.")


PACK = DomainPack(
    name="cloudcost",
    version="0.1",
    decision_kind="binary",
    simulate=simulate,
    cost_model=binary_cost_model(COST_INTERVENE, COST_OVERAGE,
                                 "intervention", "overage"),
    decision_schema=binary_decision_schema(HORIZON),
    context_kinds={
        "commit_base": {"unit": "currency_per_day",
                        "bounds": (0.0, 10_000_000.0), "max_span": 400},
        "commit_change": {"unit": "currency_per_day_delta",
                          "bounds": (-1_000_000.0, 1_000_000.0),
                          "max_span": 400},
        "deploy_uplift": {"unit": "currency_per_day_delta",
                          "bounds": (0.0, 1_000_000.0), "max_span": 60},
        "migration_reduction": {"unit": "percent", "bounds": (0.0, 100.0),
                                "max_span": 400},
    },
    question=_question,
    engine_inputs=_engine_inputs,
    engine_decision=lambda case, packet, inputs: governed_engine_decision(
        case, packet, COST_INTERVENE, COST_OVERAGE),
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
    season_length=7,
)

register(PACK)

register_templates("commit_base", base=(
    "Finance memo {ref}: the committed spend budget for this account is "
    "set at {value} per day, effective {from_date}.",
    "Per the annual commit review ({ref}), the daily budget stands at "
    "{value} from {from_date} onward.",
    "Commitment letter {ref} pegs the account's spend ceiling at "
    "{value} a day.",
))
register_templates("commit_change", base=(
    "Budget change {ref}: the daily commit moves by {value}, effective "
    "{from_date}.",
    "Commit amendment ({ref}) approved on {known_date}: adjust the "
    "ceiling by {value} starting {from_date}.",
    "FinOps note {ref}: a commit delta of {value} takes effect "
    "{from_date}.",
), revision=(
    "Correction {ref} to the earlier commit amendment: the adjustment "
    "initially filed as {prev_value} is now {value}, effective "
    "{from_date}.",
    "Updated commit note {ref}: the change first estimated at "
    "{prev_value} has been restated to {value}, still effective "
    "{from_date}.",
))
register_templates("deploy_uplift", base=(
    "Deploy notice {ref}: the rollout starting {from_date} is expected "
    "to add {value} per day of spend through {to_date}.",
    "Capacity note ({ref}): the feature launch adds roughly {value} a "
    "day from {from_date} until {to_date}.",
), revision=(
    "Deploy revision {ref}: the added spend first sized at {prev_value} "
    "per day is now put at {value}, {from_date} to {to_date}.",
))
register_templates("migration_reduction", base=(
    "Migration bulletin {ref}: workloads move off this account from "
    "{from_date}; spend should ramp down by about {value} percent.",
    "Platform memo ({ref}): the announced migration beginning "
    "{from_date} trims spend around {value} percent once complete.",
), revision=(
    "Migration update {ref}: the reduction first put at {prev_value} "
    "percent is now expected at {value} percent from {from_date}.",
))
