"""Dependence-aware horizon breach risk and explicit client policy.

Per-step exceedance probabilities are marginals.  They cannot be multiplied
as though adjacent forecast leads were independent and they are not, by
themselves, a probability that *any* step in a horizon breaches.  This module
replays the aligned residual trajectory from each rolling origin around the
published median path.  The resulting empirical paths preserve the dependence
Gnomon actually observed without inventing a parametric copula.

The policy projection is deliberately separate.  It consumes client-supplied
costs, may withhold a recommendation, and never changes the forecast or risk
estimate that it reads.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
import statistics
from typing import Any, Mapping, Sequence


MIN_JOINT_PATHS = 8
STATIONARITY_SCALE_RATIO_LIMIT = 3.0
STATIONARITY_LOCATION_SHIFT_LIMIT = 1.5
WILSON_Z_90 = 1.6448536269514722
#: Synthetic trajectories drawn when complete aligned replay paths are
#: scarce. Cheap (horizon <= a few dozen leads), and enough that the
#: point estimate's Monte-Carlo error is far below the real sampling
#: uncertainty the interval reports.
BOOTSTRAP_PATHS = 200
#: Consecutive leads drawn from a single origin per block, preserving
#: short-range residual dependence; dependence across block boundaries is
#: broken and disclosed.
BOOTSTRAP_BLOCK = 6


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    centre = _median(values)
    return _median([abs(float(value) - centre) for value in values])


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _wilson(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    p = successes / trials
    z2 = WILSON_Z_90 * WILSON_Z_90
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    radius = (WILSON_Z_90 / denominator) * math.sqrt(
        p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def aligned_residual_paths(
    residuals_by_lead: Mapping[int, Sequence[float]], horizon: int,
) -> list[list[float]]:
    """Reconstruct complete fold trajectories from lead-indexed residuals.

    ``evaluation._pool_residuals`` appends every lead from one successful
    origin before moving to the next.  Therefore item *i* at every lead is
    one aligned trajectory.  We use only complete trajectories: silently
    mixing origins at missing leads would manufacture dependence.
    """
    if horizon < 1:
        return []
    leads = [list(residuals_by_lead.get(step) or [])
             for step in range(1, horizon + 1)]
    if not leads or any(not values for values in leads):
        return []
    count = min(len(values) for values in leads)
    return [
        [float(leads[step][origin]) for step in range(horizon)]
        for origin in range(count)
    ]


def _lead_lists(
    residuals_by_lead: Mapping[int, Sequence[float]], horizon: int,
) -> list[list[float]] | None:
    """Per-lead residual columns, or ``None`` when nothing is usable."""
    leads = [list(residuals_by_lead.get(step) or [])
             for step in range(1, horizon + 1)]
    if not leads or all(not values for values in leads):
        return None
    return leads


def _bootstrap_paths(
    leads: Sequence[Sequence[float]], count: int, block: int, seed: int,
) -> list[list[float]]:
    """Synthesize residual trajectories from partially aligned columns.

    Item *i* at every lead belongs to origin *i* (``_pool_residuals``
    ordering), so drawing one origin per block of consecutive leads
    preserves the within-block dependence that origin actually exhibited.
    A block no single origin covers falls back to independent per-lead
    draws (pooled residuals for an empty lead) — weaker, and reflected in
    the estimate's tier, never silently upgraded.
    """
    rng = random.Random(seed)
    horizon = len(leads)
    pooled = [value for lead in leads for value in lead]
    paths: list[list[float]] = []
    for _ in range(count):
        path: list[float] = []
        start = 0
        while start < horizon:
            end = min(horizon, start + block)
            covering = min(len(leads[step]) for step in range(start, end))
            if covering > 0:
                origin = rng.randrange(covering)
                path.extend(float(leads[step][origin])
                            for step in range(start, end))
            else:
                for step in range(start, end):
                    source = leads[step] or pooled
                    path.append(float(source[rng.randrange(len(source))]))
            start = end
        paths.append(path)
    return paths


def _bootstrap_seed(threshold: float, leads: Sequence[Sequence[float]]) -> int:
    """Deterministic seed from the estimation inputs themselves, so the
    same series and threshold always replay the same synthetic paths
    (goldens and resumed runs stay byte-stable)."""
    digest = hashlib.sha256(repr((
        round(float(threshold), 9),
        tuple(tuple(round(float(v), 9) for v in lead) for lead in leads),
    )).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stationarity(paths: Sequence[Sequence[float]]) -> dict[str, Any]:
    """A fixed, scale-free gate for gross residual-regime changes.

    This is intentionally a guard, not a stationarity-test claim.  It catches
    the large location/scale changes for which replaying old residual paths as
    the current future is indefensible.  Small samples remain ``unknown``.
    """
    count = len(paths)
    if count < MIN_JOINT_PATHS:
        return {
            "status": "unknown", "passed": False,
            "reason": f"at least {MIN_JOINT_PATHS} aligned origins are required",
        }
    midpoint = count // 2
    early = [float(value) for path in paths[:midpoint] for value in path]
    late = [float(value) for path in paths[midpoint:] for value in path]
    early_scale, late_scale = _mad(early), _mad(late)
    floor = max(1e-12, _mad(early + late) * 1e-6)
    scale_ratio = max(early_scale, late_scale, floor) / max(
        min(early_scale, late_scale), floor)
    pooled_scale = max(_mad(early + late), floor)
    location_shift = abs(_median(late) - _median(early)) / pooled_scale
    passed = (
        scale_ratio <= STATIONARITY_SCALE_RATIO_LIMIT
        and location_shift <= STATIONARITY_LOCATION_SHIFT_LIMIT
    )
    return {
        "status": "stable" if passed else "changed",
        "passed": passed,
        "scale_ratio": round(scale_ratio, 6),
        "standardised_location_shift": round(location_shift, 6),
        "limits": {
            "scale_ratio": STATIONARITY_SCALE_RATIO_LIMIT,
            "standardised_location_shift": STATIONARITY_LOCATION_SHIFT_LIMIT,
        },
        "reason": (
            "early and late residual-origin blocks are comparable"
            if passed else
            "residual location or scale changed across rolling origins"
        ),
    }


def estimate_horizon_breach(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    residuals_by_lead: Mapping[int, Sequence[float]],
    *,
    measured_interval_coverage: float | None,
    calibration_is_verifiable: bool,
    fallback_residuals_by_lead: Mapping[int, Sequence[float]] | None = None,
    step_marginals: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Estimate any-breach and first-breach distributions from joint paths.

    An estimation ladder, not a cliff. Complete aligned replay
    trajectories from the disjoint event reserve carry ``supported``
    authority. When fewer than :data:`MIN_JOINT_PATHS` exist — a short
    history yields one event-calibration origin — the estimate degrades
    to a *disclosed* best-effort estimate: the independence composition
    of the published per-step marginals when the caller supplies them
    (``step_marginals`` — they inherit the same conformal recentring and
    scaling as the published intervals, which raw few-origin residuals
    understate), otherwise a blocked residual bootstrap over whichever
    residual source is richer (the event reserve, or the selection folds
    passed as ``fallback_residuals_by_lead``). Either way the
    probability still exists, labelled with exactly why it is not
    governed, and the bootstrap's timing/maximum distributions ride
    along as diagnostics. Only when nothing at all can be estimated is
    the probability withheld. A missing estimate was measured to price
    as never-act under an asymmetric cost model — the most expensive
    policy on the board — so silence is the last resort, never the
    default.
    """
    horizon = len(rows)
    replay_paths = aligned_residual_paths(residuals_by_lead, horizon)
    centres = [float(row.get("q50", row["point"])) for row in rows]
    stability = _stationarity(replay_paths)

    basis = "aligned_fold_residual_trajectory_replay_v1"
    residual_source = "post_selection_disjoint_origins"
    synthesized = False
    effective_origins = len(replay_paths)
    residual_paths: list[list[float]] = replay_paths
    if len(replay_paths) < MIN_JOINT_PATHS:
        event_leads = _lead_lists(residuals_by_lead, horizon)
        fallback_leads = _lead_lists(fallback_residuals_by_lead or {},
                                     horizon)

        def richness(leads: list[list[float]] | None) -> int:
            return sum(len(lead) for lead in leads) if leads else 0

        chosen, source = event_leads, "post_selection_disjoint_origins"
        if richness(fallback_leads) > richness(event_leads):
            chosen, source = fallback_leads, "selection_folds_reused"
        if chosen:
            residual_paths = _bootstrap_paths(
                chosen, BOOTSTRAP_PATHS, BOOTSTRAP_BLOCK,
                _bootstrap_seed(threshold, chosen))
            basis = "blocked_residual_bootstrap_v1"
            residual_source = source
            synthesized = True
            effective_origins = max(len(lead) for lead in chosen)

    lead_centres = [
        _median([path[step] for path in residual_paths])
        for step in range(horizon)
    ] if residual_paths else []
    forecast_paths = [
        [centres[step] + path[step] - lead_centres[step]
         for step in range(horizon)]
        for path in residual_paths
    ]
    first_steps: list[int] = []
    maxima: list[float] = []
    for path in forecast_paths:
        maxima.append(max(path))
        crossing = next(
            (step for step, value in enumerate(path, 1)
             if value > float(threshold)), None)
        if crossing is not None:
            first_steps.append(crossing)
    trials = len(forecast_paths)
    successes = len(first_steps)
    probability = successes / trials if trials else None
    if synthesized:
        # Synthetic paths are resamples, not observations: the interval's
        # sample size is the real origins behind them, so uncertainty is
        # never understated by drawing more bootstrap paths.
        effective = max(1, effective_origins)
        lower, upper = (_wilson(round(probability * effective), effective)
                        if probability is not None else (None, None))
    else:
        lower, upper = _wilson(successes, trials)
    composed = None
    if step_marginals:
        survive = 1.0
        for marginal in step_marginals:
            survive *= 1.0 - min(1.0, max(0.0, float(marginal)))
        composed = 1.0 - survive
    bootstrap_diagnostic_probability = None
    if len(replay_paths) < MIN_JOINT_PATHS and composed is not None:
        # At best-effort tier the published marginals are the better
        # decision basis: they carry the same conformal recentring and
        # per-lead spread scaling as the intervals a reader is shown,
        # while raw few-origin residual paths understate the tails
        # (measured: the bootstrap-driven policy cost 2.01/case on the
        # diagnostic corpus against 1.40 for the composed marginals).
        # The bootstrap's estimate stays visible as a diagnostic; its
        # timing and maximum distributions ride along unchanged.
        bootstrap_diagnostic_probability = probability
        probability = composed
        lower = upper = None
        basis = "independence_composed_marginals_v1"
    reasons: list[dict[str, str]] = []
    if len(replay_paths) < MIN_JOINT_PATHS:
        reasons.append({
            "code": "insufficient_joint_paths",
            "message": (
                f"Only {len(replay_paths)} aligned rolling-origin residual "
                f"paths are available; at least {MIN_JOINT_PATHS} are "
                "required for a governed horizon-event probability."
            ),
        })
    if synthesized:
        reasons.append({
            "code": "bootstrap_synthesized_paths",
            "message": (
                "Trajectories were synthesized by a blocked residual "
                "bootstrap; dependence across block boundaries is not "
                "preserved, so the estimate carries best-effort authority."
            ),
        })
        if residual_source == "selection_folds_reused":
            reasons.append({
                "code": "selection_folds_reused",
                "message": (
                    "Residuals come from the folds that selected the "
                    "winning model, not the disjoint event reserve; the "
                    "estimate can inherit selection optimism."
                ),
            })
    if basis == "independence_composed_marginals_v1":
        reasons.append({
            "code": "independence_composition_used",
            "message": (
                "The probability composes published per-step marginals as "
                "though forecast leads were independent; the leads are "
                "dependent, so this is a best-effort estimate, not a "
                "governed joint-event measurement."
            ),
        })
    if not stability.get("passed") and len(replay_paths) >= MIN_JOINT_PATHS:
        reasons.append({
            "code": "residual_regime_changed",
            "message": (
                "Residual location or scale changed across rolling origins; "
                "historical residual paths are not exchangeable enough to "
                "govern a current action."
            ),
        })
    if not calibration_is_verifiable:
        reasons.append({
            "code": "interval_calibration_out_of_band",
            "message": (
                "Measured interval coverage is outside the probability-"
                "bearing calibration band."
            ),
        })
    if measured_interval_coverage is None:
        reasons.append({
            "code": "event_calibration_unmeasured",
            "message": (
                "No untouched test-fold coverage measurement is available; "
                "risk is reported but cannot govern an action."
            ),
        })
    if not reasons:
        support = "supported"
    elif probability is not None:
        support = "best_effort"
    else:
        support = "insufficient"
    first_distribution = {
        str(step): round(first_steps.count(step) / trials, 6)
        for step in sorted(set(first_steps))
    } if trials else {}
    return {
        "method": basis,
        "residual_source": residual_source,
        "probability_any_breach": (
            round(float(probability), 6) if probability is not None else None),
        "probability_any_breach_interval_90": {
            "lower": round(lower, 6) if lower is not None else None,
            "upper": round(upper, 6) if upper is not None else None,
        },
        "breach_more_likely_than_not": (
            probability >= 0.5 if probability is not None else None),
        "first_breach_step_probability": first_distribution,
        "first_breach_step_median_conditional": (
            int(statistics.median(first_steps)) if first_steps else None),
        "expected_maximum": (
            round(statistics.mean(maxima), 6) if maxima else None),
        "maximum_interval_80": {
            "lower": _quantile(maxima, 0.1),
            "upper": _quantile(maxima, 0.9),
        },
        "joint_path_count": len(replay_paths),
        "bootstrap_path_count": trials if synthesized else 0,
        "bootstrap_diagnostic_probability": (
            round(float(bootstrap_diagnostic_probability), 6)
            if bootstrap_diagnostic_probability is not None else None),
        "independence_composed_reference": (
            round(float(composed), 6) if composed is not None else None),
        "effective_origins": effective_origins,
        "calibration_partition": "post_selection_disjoint_origins",
        "dependence_preserved": (
            not synthesized
            and basis != "independence_composed_marginals_v1"),
        "residual_regime": stability,
        "measured_interval_coverage": measured_interval_coverage,
        "support": support,
        "reasons": reasons,
        "assumptions": (
            ["The probability composes conformally scaled per-step "
             "marginals under an independence assumption the leads do "
             "not satisfy; path-based timing and maximum figures are "
             "bootstrap diagnostics."]
            if basis == "independence_composed_marginals_v1" else
            ["Blocked bootstrap trajectories preserve within-block "
             "residual dependence only; the interval's sample size is "
             "the real origin count, not the synthetic path count.",
             "This is an empirical path distribution, not independent "
             "composition of per-step probabilities."]
            if synthesized else
            ["Aligned rolling-origin residual trajectories are "
             "exchangeable with the forecast horizon after the fixed "
             "regime gate.",
             "This is an empirical path distribution, not independent "
             "composition of per-step probabilities."]),
    }


@dataclass(frozen=True)
class BreachDecisionPolicy:
    """Single-shot mitigation policy supplied by the caller."""

    action_cost: float
    miss_cost: float
    mitigation_effectiveness: float = 1.0

    def validate(self) -> None:
        if not math.isfinite(self.action_cost) or self.action_cost < 0:
            raise ValueError("action_cost must be finite and >= 0")
        if not math.isfinite(self.miss_cost) or self.miss_cost <= 0:
            raise ValueError("miss_cost must be finite and > 0")
        if (not math.isfinite(self.mitigation_effectiveness)
                or not 0 < self.mitigation_effectiveness <= 1):
            raise ValueError("mitigation_effectiveness must be in (0, 1]")


def apply_breach_policy(
    event_risk: Mapping[str, Any], policy: BreachDecisionPolicy,
) -> dict[str, Any]:
    """Project immutable risk into a tiered recommendation.

    Whenever a probability exists, a cost-aware recommendation exists:
    the expected-loss comparison at the point estimate. What varies is
    its *authority*. ``supported`` requires a supported event estimate
    whose 90% interval clears the break-even entirely on one side;
    anything less publishes at ``best_effort`` with the reason stated.
    Recommending nothing is reserved for "no probability could be formed
    at all" — defaulting silence to monitor was measured to invert an
    asymmetric cost model (a missed breach costs multiples of an alarm)
    and price as the worst constant policy on the board.
    """
    policy.validate()
    probability = event_risk.get("probability_any_breach")
    interval = event_risk.get("probability_any_breach_interval_90") or {}
    lower, upper = interval.get("lower"), interval.get("upper")
    break_even = policy.action_cost / (
        policy.miss_cost * policy.mitigation_effectiveness)
    expected_act = None
    expected_monitor = None
    recommendation: str | None = None
    decision_support = "insufficient"
    reason_code: str
    if probability is None:
        reason_code = "event_probability_unavailable"
    else:
        probability = float(probability)
        expected_monitor = probability * policy.miss_cost
        expected_act = policy.action_cost + probability * policy.miss_cost * (
            1.0 - policy.mitigation_effectiveness)
        recommendation = ("act" if expected_monitor > expected_act
                          else "monitor")
        interval_resolved = (
            lower is not None and upper is not None
            and (float(lower) > break_even or float(upper) < break_even))
        if event_risk.get("support") == "supported" and interval_resolved:
            decision_support = "supported"
            reason_code = ("loss_interval_favours_action"
                           if float(lower) > break_even
                           else "loss_interval_favours_monitor")
        elif event_risk.get("support") != "supported":
            decision_support = "best_effort"
            reason_code = "event_estimate_not_governed_point_estimate_used"
        else:
            decision_support = "best_effort"
            reason_code = "policy_boundary_unresolved_point_estimate_used"
    return {
        "cost_model": "single_shot_mitigation_v1",
        "action_cost": policy.action_cost,
        "miss_cost": policy.miss_cost,
        "mitigation_effectiveness": policy.mitigation_effectiveness,
        "break_even_probability": round(break_even, 6),
        "expected_loss_if_act": (
            round(expected_act, 6) if expected_act is not None else None),
        "expected_loss_if_monitor": (
            round(expected_monitor, 6) if expected_monitor is not None else None),
        "recommended_action": recommendation,
        "decision_support": decision_support,
        "reason_code": reason_code,
        "event_support": event_risk.get("support"),
        "event_reasons": list(event_risk.get("reasons") or []),
        "breach_more_likely_than_not": event_risk.get(
            "breach_more_likely_than_not"),
        "probability_any_breach": probability,
        "policy_assumption": (
            "One irreversible decision is made now; the option value of "
            "waiting for future observations and acting later is not modelled."
        ),
        "primary_risk_unchanged": True,
    }
