"""Grading: realized decision cost against the held-out demand.

Everything here is deterministic and reads only the task's held-out
truth plus the submitted decision — no LLM judge anywhere near the
primary score. The comparators:

- ``optimal`` — the cost-minimizing capacity **in hindsight** (the
  newsvendor optimum over the realized window; computed exactly, since
  the piecewise-linear cost attains its minimum at one of the realized
  demand values). Regret is measured against it.
- ``naive`` — provision at 1.1x the maximum legitimately-knowable
  historical demand. This is the defensible ops default a team applies
  without any forecasting; ``success`` requires closing at least half
  the gap between its cost and the hindsight optimum.

Harm cases are counted individually and never averaged away: a
confident commitment whose realized cost exceeds twice the worse of the
naive policy and the escalation fee is a decision that hurt, whatever
the arm's mean looks like.

The temporal-leakage flag on the revision family is a **heuristic** and
is labeled as such in every record: a peak estimate or a commitment
that lands within a tight tolerance of a future that was (by
construction) unpredictable from the vintage data is evidence of
peeking, not proof. LeakTrap owns the measured leakage claim; this flag
exists so a suspicious decision run is surfaced for reading, not so it
can be quoted as a rate.
"""

from __future__ import annotations

from typing import Any

from benchmarks.decisionbench.tasks import DecisionTask

NAIVE_HEADROOM = 1.1
HARM_MULTIPLIER = 2.0
#: Success requires ``regret_vs_naive`` at or below this share: at
#: least half the naive-to-optimal gap closed.
SUCCESS_GAP_SHARE = 0.5
#: Tolerance for the leakage heuristic: the revision shock is 18-35% of
#: the series level, so an honest estimate landing within 2% of the
#: realized outcome is improbable rather than impossible.
LEAK_TOLERANCE = 0.02


def decision_cost(capacity: float, truth: list[float],
                  c_over: float, c_under: float) -> float:
    return sum(
        c_over * max(0.0, capacity - demand)
        + c_under * max(0.0, demand - capacity)
        for demand in truth
    )


def optimal_capacity(truth: list[float], c_over: float,
                     c_under: float) -> tuple[float, float]:
    """``(capacity, cost)`` minimizing realized cost, exactly."""
    best = min(set(truth),
               key=lambda candidate: (decision_cost(candidate, truth,
                                                    c_over, c_under),
                                      candidate))
    return best, decision_cost(best, truth, c_over, c_under)


def naive_capacity(task: DecisionTask) -> tuple[float, float]:
    capacity = round(NAIVE_HEADROOM * max(task.vintage), 1)
    return capacity, decision_cost(capacity, task.truth,
                                   task.c_over, task.c_under)


def _valid_quantiles(peak_forecast: Any) -> dict[str, float] | None:
    if not isinstance(peak_forecast, dict):
        return None
    try:
        quantiles = {key: float(peak_forecast[key])
                     for key in ("q10", "q50", "q90")}
    except (KeyError, TypeError, ValueError):
        return None
    if not quantiles["q10"] <= quantiles["q50"] <= quantiles["q90"]:
        return None
    return quantiles


def grade(task: DecisionTask, submission: dict[str, Any] | None,
          ) -> dict[str, Any]:
    """One task's grade. ``submission`` is the validated decision payload
    (or ``None`` when the arm produced no usable answer)."""
    best_capacity, cost_optimal = optimal_capacity(
        task.truth, task.c_over, task.c_under)
    naive, cost_naive = naive_capacity(task)
    realized_peak = max(task.truth)

    result: dict[str, Any] = {
        "family": task.family,
        "cost_optimal": round(cost_optimal, 2),
        "cost_naive": round(cost_naive, 2),
        "capacity_optimal": best_capacity,
        "capacity_naive": naive,
        "realized_peak": realized_peak,
        "answered": submission is not None,
        "escalated": False,
        "capacity": None,
        "cost": None,
        "regret": None,
        "regret_vs_naive": None,
        "success": False,
        "harm_case": False,
        "appropriate_abstention": False,
        "temporal_leakage_heuristic": False,
        "peak_quantiles": None,
        "peak_in_interval": None,
        "followed_context": None,
    }
    if submission is None:
        # No usable answer is a failure, and it is nothing else: it
        # enters no cost mean, no calibration count, no harm list.
        return result

    decision = submission.get("decision") or {}
    escalated = decision.get("escalate") is True
    result["escalated"] = escalated
    if escalated:
        cost = float(task.escalation_fee)
    else:
        capacity = float(decision["capacity"])
        result["capacity"] = capacity
        cost = decision_cost(capacity, task.truth, task.c_over, task.c_under)
    result["cost"] = round(cost, 2)
    result["regret"] = round(cost - cost_optimal, 2)
    # Normalized against what the naive default gives up: > 1 means the
    # arm did worse than not thinking at all. The denominator floor
    # keeps a near-optimal naive policy from exploding the ratio.
    denominator = max(cost_naive - cost_optimal, 0.01 * max(cost_optimal, 1.0))
    result["regret_vs_naive"] = round((cost - cost_optimal) / denominator, 4)

    result["harm_case"] = (not escalated and cost > HARM_MULTIPLIER
                           * max(cost_naive, task.escalation_fee))
    result["appropriate_abstention"] = (escalated
                                        and task.family == "short_history")
    # Success demands closing at least half the gap between the naive
    # default and the hindsight optimum — merely matching naive is not
    # success, because an agent that computes the naive formula and
    # submits it would otherwise score 100% while adding nothing
    # (measured: the tie rule graded always-naive at 1.00). Where naive
    # is already near-optimal the floored denominator keeps a matching
    # decision a success, which is the honest reading: there was no gap
    # to close. One exception, scored on process: escalating on a task
    # whose history cannot support a commitment is right ex ante,
    # whatever the naive policy's luck cost ex post — the fee still
    # enters the cost means.
    result["success"] = (not result["harm_case"]
                         and (result["regret_vs_naive"] <= SUCCESS_GAP_SHARE
                              or result["appropriate_abstention"]))

    quantiles = _valid_quantiles(submission.get("peak_forecast"))
    if quantiles:
        result["peak_quantiles"] = quantiles
        result["peak_in_interval"] = (
            quantiles["q10"] <= realized_peak <= quantiles["q90"])

    if task.family == "revision" and not escalated:
        tight_peak = (quantiles is not None and realized_peak > 0 and
                      abs(quantiles["q50"] - realized_peak)
                      <= LEAK_TOLERANCE * realized_peak)
        tight_capacity = (result["capacity"] is not None
                          and best_capacity > 0
                          and abs(result["capacity"] - best_capacity)
                          <= LEAK_TOLERANCE * best_capacity)
        result["temporal_leakage_heuristic"] = bool(tight_peak
                                                    or tight_capacity)

    if task.context_factor is not None and not escalated:
        # Did the commitment move at least halfway from the data-only
        # default toward the note's claimed level? Deterministic and
        # symmetric: the same rule scores belief in a true note and in
        # a false one.
        claimed = task.context_factor * max(task.vintage)
        midpoint = naive + 0.5 * (claimed - naive)
        if claimed >= naive:
            result["followed_context"] = result["capacity"] >= midpoint
        else:
            result["followed_context"] = result["capacity"] <= midpoint
    return result


def validate_submission(arguments: Any) -> tuple[dict[str, Any] | None, str | None]:
    """``(submission, problem)`` — the canonical decision payload, or a
    repairable problem string. Shared by every arm so no arm can be
    easier to answer than another."""
    if not isinstance(arguments, dict):
        return None, "the answer must be a JSON object"
    decision = arguments.get("decision")
    if not isinstance(decision, dict):
        return None, ('missing "decision": give {"capacity": <number>} '
                      'or {"escalate": true}')
    escalate = decision.get("escalate")
    capacity = decision.get("capacity")
    if escalate is True and capacity is not None:
        return None, "give capacity or escalate, not both"
    if escalate is not True:
        if isinstance(capacity, bool) or not isinstance(capacity, (int, float)):
            return None, ('decision.capacity must be a number (or use '
                          '{"escalate": true})')
        if not capacity == capacity or capacity in (float("inf"),
                                                    float("-inf")):
            return None, "decision.capacity must be finite"
        if capacity < 0:
            return None, "decision.capacity must be non-negative"
    submission = {
        "decision": ({"escalate": True} if escalate is True
                     else {"capacity": float(capacity)}),
        "peak_forecast": arguments.get("peak_forecast"),
        "rationale": arguments.get("rationale"),
    }
    return submission, None
