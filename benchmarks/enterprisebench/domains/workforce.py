"""Intraday contact volume with campaigns and outages: staff up an
extra shift or hold, against the staffed capacity.

Series: hourly inbound contact volume — a business-hours daily profile,
a weekday factor, noise, and causal context effects: marketing
campaigns (a send at a stated hour produces a surge that decays over
the following hours) and service outages (a burst of contacts while
service is down). Both act mechanistically on history and future alike.

Decision: will any hour of the next day exceed the staffed capacity
(the ``staffed_capacity`` fact) — staff up now (extra shift, fully
absorbs the surge) or hold (an SLA breach costs multiples). Break-even
probability = labor/penalty = 0.25, base rate held near it and
disclosed.

Trap flavor (~15%, disclosed): a *campaign send-time is moved* after
announcement — the revision, known before the cutoff, moves the surge
into or out of the delivery day, and the counterfactual future
simulated at the stale send-time flips the optimal decision by
construction.

Engine mapping: Gnomon forecasts the volume series against an effective
capacity threshold through the governed breach ladder. Structured facts
adjust the threshold: a campaign whose as-of window overlaps the
horizon lowers the effective capacity by its expected peak surge (the
surge the extrapolated history cannot know about), so the send-time
trap flows through the engine's structured path too.

Parameters and grounding: 40–400 contacts/hour base with a 9am–6pm
concentration mirrors mid-size contact centers; campaign surges of
60–180% of base decaying over 4–10 hours match send-driven traffic;
outage bursts of 80–250% for 1–4 hours match incident-driven volume.
Volumes and capacity pass per-case seeded positive affine anonymization
(levels ``a*x+b``, surge sizes ``a*x``); decision structure is verified
invariant on the rounded shown numbers.
"""

from __future__ import annotations

import math
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

HISTORY = 14 * 24
HORIZON = 24
COST_STAFF = 3.0
COST_SLA = 12.0
CONFIG: dict[str, Any] = {
    "history_hours": HISTORY, "horizon_hours": HORIZON, "frequency": "h",
    "units": "cost_units_staff_up_3_sla_breach_12",
    "base_rate": (40.0, 400.0), "weekday_uplift": (0.1, 0.4),
    "noise_fraction": 0.06,
    "campaign_surge_fraction": (0.6, 1.8), "campaign_decay_hours": (4, 10),
    "outage_surge_fraction": (0.8, 2.5), "outage_hours": (1, 4),
    "outcome_targets": {"no_breach": 0.55, "breach": 0.30, "trap": 0.15},
}


def _volume(rng: random.Random, hours: int, base: float, weekday: float,
            campaigns: list[dict[str, Any]],
            outages: list[dict[str, Any]]) -> list[float]:
    values = []
    for hour in range(hours):
        day, hour_of_day = divmod(hour, 24)
        profile = max(0.08, math.sin((hour_of_day - 8) / 11.0 * math.pi)
                      if 8 <= hour_of_day <= 19 else 0.08)
        level = base * profile
        if day % 7 < 5:
            level *= 1.0 + weekday
        for campaign in campaigns:
            if hour >= campaign["send"]:
                age = hour - campaign["send"]
                level += base * campaign["surge"] * math.exp(
                    -age / campaign["decay"])
        for outage in outages:
            if outage["from"] <= hour <= outage["to"]:
                level += base * outage["surge"]
        values.append(max(0.0, level + rng.gauss(
            0.0, CONFIG["noise_fraction"] * base)))
    return values


def simulate(seed: int, count: int) -> tuple[list[Case], dict[str, Any]]:
    caps = {cell: int(count * fraction) + 1
            for cell, fraction in CONFIG["outcome_targets"].items()}
    cases: list[Case] = []
    cell_counts: dict[str, int] = {}
    skipped = {"cell_full": 0, "rounding_flip": 0, "trap_shape": 0,
               "degenerate": 0}
    attempts = 0
    balanced_limit = 300 * count
    hours = HISTORY + HORIZON
    while len(cases) < count and attempts < 2 * balanced_limit:
        attempts += 1
        balanced_phase = attempts <= balanced_limit
        rng = random.Random(
            f"enterprisebench:workforce:{seed}:{attempts}")
        trap = balanced_phase and cell_counts.get("trap", 0) < caps["trap"]

        base = rng.uniform(*CONFIG["base_rate"])
        weekday = rng.uniform(*CONFIG["weekday_uplift"])
        campaigns: list[dict[str, Any]] = []
        outages: list[dict[str, Any]] = []
        items: list[ContextItem] = []
        trap_detail = None
        if trap:
            surge = rng.uniform(1.0, 1.8)
            decay = rng.randint(*CONFIG["campaign_decay_hours"])
            stale_send = HISTORY + rng.randrange(8, 18)
            move = rng.choice((-1, 1)) * rng.randrange(24, 60)
            new_send = stale_send + move
            if not HISTORY - 48 <= new_send <= hours - 3:
                new_send = stale_send - move
            if not 0 <= new_send <= hours - 3:
                skipped["trap_shape"] += 1
                continue
            announced = rng.randrange(HISTORY - 96, HISTORY - 24)
            revised = rng.randrange(announced + 6, HISTORY)
            trap_detail = {"surge": surge, "decay": decay,
                           "stale_send": stale_send,
                           "new_send": new_send,
                           "announced": announced, "revised": revised}
            campaigns.append({"send": new_send, "surge": surge,
                              "decay": decay})
        else:
            if rng.random() < 0.55:
                send = rng.randrange(HISTORY // 2, hours - 3)
                campaign = {
                    "send": send,
                    "surge": rng.uniform(
                        *CONFIG["campaign_surge_fraction"]),
                    "decay": rng.randint(
                        *CONFIG["campaign_decay_hours"])}
                campaigns.append(campaign)
                items.append(ContextItem(
                    "campaign-0", "campaign_surge",
                    campaign["surge"] * base,
                    max(0, send - rng.randrange(12, 72)), send,
                    min(send + 3 * campaign["decay"], hours - 1)))
            if rng.random() < 0.3:
                start = rng.randrange(HISTORY // 2, HISTORY - 2)
                outage = {"from": start,
                          "to": start + rng.randint(
                              *CONFIG["outage_hours"]),
                          "surge": rng.uniform(
                              *CONFIG["outage_surge_fraction"])}
                outages.append(outage)
                items.append(ContextItem(
                    "outage-0", "outage_surge", outage["surge"] * base,
                    max(0, start - rng.randrange(0, 3)), outage["from"],
                    min(outage["to"], hours - 1),
                    text_only=rng.random() < 0.3))

        series_rng = random.Random(
            f"enterprisebench:workforce:{seed}:{attempts}:series")
        values = _volume(series_rng, hours, base, weekday, campaigns,
                         outages)
        history, future = values[:HISTORY], values[HISTORY:]
        if max(history) == min(history):
            skipped["degenerate"] += 1
            continue
        if trap_detail is not None:
            stale_rng = random.Random(
                f"enterprisebench:workforce:{seed}:{attempts}:series")
            stale_values = _volume(
                stale_rng, hours, base, weekday,
                [{"send": trap_detail["stale_send"],
                  "surge": trap_detail["surge"],
                  "decay": trap_detail["decay"]}], outages)
            stale_future = stale_values[HISTORY:]
            low = min(max(future), max(stale_future))
            high = max(max(future), max(stale_future))
            if high - low < 0.35 * base:
                skipped["trap_shape"] += 1
                continue
            capacity = low + rng.uniform(0.35, 0.65) * (high - low)
            cell = "trap"
        else:
            stale_future = None
            noise = CONFIG["noise_fraction"] * base
            breach = (cell_counts.get("breach", 0) < caps["breach"]
                      if balanced_phase else rng.random() < 0.3)
            margin = (rng.uniform(0.3, 2.0) if breach
                      else rng.uniform(0.5, 6.0)) * noise
            capacity = max(future) - margin if breach \
                else max(future) + margin
            cell = "breach" if breach else "no_breach"
        if balanced_phase and cell_counts.get(cell, 0) >= caps[cell]:
            skipped["cell_full"] += 1
            continue

        items.append(ContextItem(
            "capacity", "staffed_capacity", capacity, 0, 0, hours - 1))
        if trap_detail is not None:
            items.append(ContextItem(
                "campaign-0", "campaign_surge",
                trap_detail["surge"] * base, trap_detail["announced"],
                trap_detail["stale_send"],
                min(trap_detail["stale_send"]
                    + 3 * trap_detail["decay"], hours - 1)))
            items.append(ContextItem(
                "campaign-0-moved", "campaign_surge",
                trap_detail["surge"] * base, trap_detail["revised"],
                trap_detail["new_send"],
                min(trap_detail["new_send"]
                    + 3 * trap_detail["decay"], hours - 1),
                revises="campaign-0", trap=True))

        a = rng.uniform(0.6, 2.4)
        b = rng.uniform(10, 300) - a * statistics.median(history)
        shown_history = tuple(round(a * v + b, 4) for v in history)
        shown_future = tuple(round(a * v + b, 4) for v in future)
        shown_capacity = round(a * capacity + b, 4)

        def transform(item: ContextItem) -> ContextItem:
            value = (round(a * item.value + b, 4)
                     if item.kind == "staffed_capacity"
                     else round(a * item.value, 4))
            return ContextItem(item.item_id, item.kind, value,
                               item.known_at, item.effective_from,
                               item.effective_to, item.revises,
                               item.text_only, item.trap, item.aux)

        shown_items = tuple(transform(item) for item in items)
        breach_steps = [step for step, value
                        in enumerate(shown_future, 1)
                        if value > shown_capacity]
        expected = (max(future) > capacity if trap
                    else cell == "breach")
        if bool(breach_steps) != expected:
            skipped["rounding_flip"] += 1
            continue
        trap_optimal = stale_optimal = None
        if trap:
            stale_breach = any(round(a * v + b, 4) > shown_capacity
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
            case_id=f"wf{seed}-{len(cases):04d}", domain="workforce",
            frequency="h", values=shown_history, future=shown_future,
            horizon=HORIZON, items=shown_items,
            threshold=shown_capacity, trap=trap,
            trap_optimal=trap_optimal, stale_optimal=stale_optimal,
            series_id=f"queue-{len(cases):04d}",
            meta={"truth_event": bool(breach_steps),
                  "truth_first_step": (breach_steps[0]
                                       if breach_steps else None),
                  "outcome_cell": cell}))
    if len(cases) < count:
        raise ValueError(
            f"workforce: only {len(cases)}/{count} cases after "
            f"{attempts} attempts; skipped={skipped}")
    provenance = {
        "generator": "mechanistic_intraday_contacts_with_campaigns",
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
            "per_case_seeded_positive_affine;volumes_and_capacity_"
            "ax_plus_b;surge_sizes_ax;"
            "decision_invariance_verified_on_rounded_shown_numbers"),
    }
    return cases, provenance


def _engine_inputs(case: Case, facts: list[ContextItem]) -> dict[str, Any]:
    capacity = next((item.value for item in facts
                     if item.kind == "staffed_capacity"),
                    max(case.values))
    surge = 0.0
    for item in facts:
        if item.kind not in ("campaign_surge", "outage_surge"):
            continue
        overlap_from = max(item.effective_from, case.cutoff)
        overlap_to = min(item.effective_to,
                         case.cutoff + case.horizon - 1)
        if overlap_to >= overlap_from:
            surge = max(surge, item.value)
    return {"threshold": capacity - surge, "basis": "shown_series",
            "effective_capacity": capacity - surge}


def _question(case: Case) -> str:
    return (
        "Decision: will hourly contact volume exceed the staffed "
        "capacity (the staffed_capacity fact) in any of the unshown "
        "hours of the next day? Campaign memos state the send time; a "
        "surge follows the send and decays over the stated window — use "
        "each memo in the version known as of the cutoff. Acting "
        "(staffing an extra shift) costs the labor and fully absorbs "
        "any surge; holding costs nothing unless the SLA is breached.")


PACK = DomainPack(
    name="workforce",
    version="0.1",
    decision_kind="binary",
    simulate=simulate,
    cost_model=binary_cost_model(COST_STAFF, COST_SLA,
                                 "extra_shift_labor", "sla_breach"),
    decision_schema=binary_decision_schema(HORIZON),
    context_kinds={
        "staffed_capacity": {"unit": "contacts_per_hour",
                             "bounds": (-1_000_000.0, 1_000_000.0),
                             "max_span": 500},
        "campaign_surge": {"unit": "contacts_per_hour_delta",
                           "bounds": (0.0, 1_000_000.0), "max_span": 60},
        "outage_surge": {"unit": "contacts_per_hour_delta",
                         "bounds": (0.0, 1_000_000.0), "max_span": 24},
    },
    question=_question,
    engine_inputs=_engine_inputs,
    engine_decision=lambda case, packet, inputs: governed_engine_decision(
        case, packet, COST_STAFF, COST_SLA),
    decision_from_forecast=lambda case, path, inputs: crossing_decision(
        case, path, inputs["effective_capacity"]),
    constant_policies=lambda case: {
        "always_act": {"action": "act", "event_expected": True,
                       "first_event_step": 1},
        "never_act": {"action": "monitor", "event_expected": False,
                      "first_event_step": None}},
    parse_decision=parse_binary_decision,
    decision_scalar=lambda decision: 1.0
    if decision.get("action") == "act" else 0.0,
    config=CONFIG,
    season_length=24,
)

register(PACK)

register_templates("staffed_capacity", base=(
    "Ops plan {ref}: the staffed handling capacity is {value} contacts "
    "per hour on the current roster.",
    "WFM sheet ({ref}): current shift coverage handles {value} an hour "
    "before the SLA is at risk.",
))
register_templates("campaign_surge", base=(
    "Campaign brief {ref}: the send goes out at {from_date}; expect a "
    "surge peaking around {value} extra contacts an hour, decaying "
    "through {to_date}.",
    "Marketing ops note ({ref}): blast scheduled {from_date}, projected "
    "peak of {value} additional contacts hourly.",
), revision=(
    "Send-time change {ref}: the campaign previously scheduled for an "
    "earlier slot now goes out at {from_date}; the projected peak "
    "stays near {value} extra contacts an hour, tailing off by "
    "{to_date}.",
    "Updated campaign brief ({ref}): send moved to {from_date}, peak "
    "still about {value} an hour.",
))
register_templates("outage_surge", base=(
    "Incident notice {ref}: the outage from {from_date} to {to_date} "
    "is driving roughly {value} extra contacts an hour while service "
    "is degraded.",
    "Status page follow-up ({ref}): expect about {value} additional "
    "hourly contacts during the {from_date} outage window.",
))
