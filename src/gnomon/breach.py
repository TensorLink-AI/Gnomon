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
import math
import statistics
from typing import Any, Mapping, Sequence


MIN_JOINT_PATHS = 8
STATIONARITY_SCALE_RATIO_LIMIT = 3.0
STATIONARITY_LOCATION_SHIFT_LIMIT = 1.5
WILSON_Z_90 = 1.6448536269514722


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
) -> dict[str, Any]:
    """Estimate any-breach and first-breach distributions from joint paths."""
    horizon = len(rows)
    residual_paths = aligned_residual_paths(residuals_by_lead, horizon)
    centres = [float(row.get("q50", row["point"])) for row in rows]
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
    lower, upper = _wilson(successes, trials)
    stability = _stationarity(residual_paths)
    reasons: list[dict[str, str]] = []
    if trials < MIN_JOINT_PATHS:
        reasons.append({
            "code": "insufficient_joint_paths",
            "message": (
                f"Only {trials} aligned rolling-origin residual paths are "
                f"available; at least {MIN_JOINT_PATHS} are required for a "
                "governed horizon-event probability."
            ),
        })
    if not stability.get("passed") and trials >= MIN_JOINT_PATHS:
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
    support = "supported" if not reasons else "insufficient"
    first_distribution = {
        str(step): round(first_steps.count(step) / trials, 6)
        for step in sorted(set(first_steps))
    } if trials else {}
    return {
        "method": "aligned_fold_residual_trajectory_replay_v1",
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
        "joint_path_count": trials,
        "calibration_partition": "post_selection_disjoint_origins",
        "dependence_preserved": True,
        "residual_regime": stability,
        "measured_interval_coverage": measured_interval_coverage,
        "support": support,
        "reasons": reasons,
        "assumptions": [
            "Aligned rolling-origin residual trajectories are exchangeable "
            "with the forecast horizon after the fixed regime gate.",
            "This is an empirical replay distribution, not independent "
            "composition of per-step probabilities.",
        ],
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
    """Project immutable risk into an action, or explicitly withhold it."""
    policy.validate()
    probability = event_risk.get("probability_any_breach")
    interval = event_risk.get("probability_any_breach_interval_90") or {}
    lower, upper = interval.get("lower"), interval.get("upper")
    break_even = policy.action_cost / (
        policy.miss_cost * policy.mitigation_effectiveness)
    expected_act = None
    expected_monitor = None
    if probability is not None:
        probability = float(probability)
        expected_monitor = probability * policy.miss_cost
        expected_act = policy.action_cost + probability * policy.miss_cost * (
            1.0 - policy.mitigation_effectiveness)
    recommendation: str | None = None
    reason_code: str
    if event_risk.get("support") != "supported":
        reason_code = "event_probability_insufficient"
    elif lower is None or upper is None:
        reason_code = "event_probability_interval_missing"
    elif float(lower) > break_even:
        recommendation, reason_code = "act", "loss_interval_favours_action"
    elif float(upper) < break_even:
        recommendation, reason_code = "monitor", "loss_interval_favours_monitor"
    else:
        reason_code = "policy_boundary_not_resolved"
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
        "decision_support": "supported" if recommendation else "insufficient",
        "reason_code": reason_code,
        "breach_more_likely_than_not": event_risk.get(
            "breach_more_likely_than_not"),
        "probability_any_breach": probability,
        "policy_assumption": (
            "One irreversible decision is made now; the option value of "
            "waiting for future observations and acting later is not modelled."
        ),
        "primary_risk_unchanged": True,
    }
