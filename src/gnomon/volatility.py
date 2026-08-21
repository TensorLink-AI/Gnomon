"""Fold-safe selection and execution for future residual dispersion."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any

from .temporal_distribution import (
    TemporalDecisionPolicy,
    TemporalPropertyDistribution,
)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _scale(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = _median(values)
    mad = _median([abs(item - centre) for item in values])
    return 1.4826 * mad if mad > 1e-12 else float(statistics.stdev(values))


_DIRECTIONS = ("decreased", "stable", "increased")


def _direction(ratio: float) -> str:
    return "increased" if ratio > 1.25 else "decreased" if ratio < .8 else "stable"


def _smoothed_probabilities(samples: list[float]) -> dict[str, float]:
    counts = {label: sum(_direction(value) == label for value in samples)
              for label in _DIRECTIONS}
    denominator = len(samples) + len(_DIRECTIONS)
    return {label: (counts[label] + 1) / denominator for label in _DIRECTIONS}


def _balanced_accuracy(pairs: list[tuple[str, str]]) -> float:
    recalls = []
    for label in _DIRECTIONS:
        actual = [predicted for predicted, observed in pairs if observed == label]
        if actual:
            recalls.append(sum(item == label for item in actual) / len(actual))
    return statistics.mean(recalls) if recalls else 0.0


def _brier(probabilities: dict[str, float], actual: str) -> float:
    return sum((probabilities[label] - (label == actual)) ** 2
               for label in _DIRECTIONS)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires observations")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _momentum_signal(history: list[float]) -> tuple[bool, float]:
    """Robust recent log-scale slope from overlapping windows."""
    width = max(24, min(48, len(history) // 5))
    step = max(8, width // 2)
    if len(history) < width + 3 * step:
        return False, 0.0
    ends = list(range(max(width, len(history) - 4 * step),
                      len(history) + 1, step))
    points = [(end, max(_scale(history[end - width:end]), 1e-12))
              for end in ends]
    slopes = [(math.log(right[1]) - math.log(left[1])) /
              (right[0] - left[0])
              for index, left in enumerate(points)
              for right in points[index + 1:]]
    slope = statistics.median(slopes)
    net = math.log(points[-1][1] / points[0][1])
    adjacent = [math.log(points[index + 1][1] / points[index][1])
                for index in range(len(points) - 1)]
    agreeing = sum(item * net > 0 for item in adjacent)
    persistent = abs(net) >= .15 and agreeing >= 2
    return persistent, slope


def residuals(values: list[float], season: int = 1) -> list[float]:
    """Causal one-step residuals; no fitted value sees its observation."""
    result: list[float] = []
    minimum = max(3, season)
    for index in range(minimum, len(values)):
        history = values[:index]
        if season > 1 and index >= 2 * season:
            prediction = history[-season]
        else:
            width = min(8, len(history))
            x = history[-width:]
            slope = (x[-1] - x[0]) / max(1, width - 1)
            prediction = x[-1] + slope
        result.append(values[index] - prediction)
    return result


def _predict(kind: str, history: list[float], horizon: int, season: int = 1) -> float:
    window = max(8, min(48, len(history) // 2))
    if kind == "constant":
        return _scale(history)
    if kind == "recent":
        return _scale(history[-window:])
    if kind.startswith("ewma"):
        variance = _scale(history[:window]) ** 2
        alpha = {"ewma_slow": .05, "ewma": .15, "ewma_fast": .3}[kind]
        for item in history[window:]:
            variance = alpha * item * item + (1 - alpha) * variance
        return math.sqrt(max(0.0, variance))
    if kind == "scale_trend":
        blocks = [history[i:i + window] for i in range(0, len(history), window)]
        scales = [_scale(block) for block in blocks if len(block) >= 8]
        if len(scales) < 2:
            return _scale(history)
        log_change = math.log(max(scales[-1], 1e-12) / max(scales[-2], 1e-12))
        # Bound extrapolation: volatility trends are rarely stable for long.
        return scales[-1] * math.exp(max(-0.7, min(0.7, log_change * horizon / window)))
    if kind == "robust_scale_trend":
        width = max(12, min(32, len(history) // 6))
        step = max(4, width // 2)
        blocks = [(end, _scale(history[end - width:end]))
                  for end in range(width, len(history) + 1, step)]
        blocks = [(end, scale) for end, scale in blocks if scale > 1e-12]
        if len(blocks) < 4:
            return _scale(history[-window:])
        # Median pairwise log-slope is resistant to one volatile block.
        slopes = [(math.log(right[1]) - math.log(left[1])) /
                  (right[0] - left[0])
                  for index, left in enumerate(blocks)
                  for right in blocks[index + 1:]]
        slope = statistics.median(slopes)
        return blocks[-1][1] * math.exp(
            max(-.7, min(.7, slope * max(1, horizon / 2))))
    if kind == "regime_mixture":
        width = max(12, min(48, len(history) // 4))
        if len(history) < 2 * width:
            return _scale(history[-window:])
        previous = max(_scale(history[-2 * width:-width]), 1e-12)
        recent = max(_scale(history[-width:]), 1e-12)
        baseline = max(_scale(history), 1e-12)
        evidence = abs(math.log(recent / previous))
        # A small apparent shift is mostly sampling noise; persistent large
        # shifts smoothly move weight to the recent regime.
        weight = max(0.0, min(1.0, (evidence - .1) / .6))
        return math.exp((1 - weight) * math.log(baseline)
                        + weight * math.log(recent))
    if kind == "scale_momentum":
        persistent, slope = _momentum_signal(history)
        width = max(24, min(48, len(history) // 5))
        recent = max(_scale(history[-width:]),
                     1e-12)
        if not persistent:
            return recent
        # The recent block scale is centred roughly width/2 steps before the
        # cutoff, while the requested future block is centred horizon/2 after
        # it. Extrapolate across the distance between those centres.
        distance = (width + horizon) / 2
        return recent * math.exp(max(-.7, min(.7, slope * distance)))
    if kind == "seasonal_phase":
        if season <= 1 or len(history) < 3 * season:
            return _scale(history)
        phase_scales = [_scale(history[offset::season][-8:])
                        for offset in range(season)]
        future = [phase_scales[(len(history) + step) % season]
                  for step in range(horizon)]
        return math.sqrt(sum(item * item for item in future) / len(future))
    raise ValueError(kind)


@dataclass(frozen=True)
class FittedVolatilityExecutable:
    candidate: str
    scale: float
    lower: float
    upper: float
    horizon: int
    support: str
    scores: dict[str, float]
    baseline_score: float
    improvement: float
    training_observations: int
    direction: str
    diagnostics: dict[str, Any]
    reference_scale: float
    ratio: float | None
    ratio_lower: float | None
    ratio_upper: float | None
    direction_probabilities: dict[str, float]
    direction_support: str
    direction_candidate: str
    property_distribution: TemporalPropertyDistribution
    decision_policy: TemporalDecisionPolicy

    def execute(self) -> dict[str, Any]:
        """Publish the fitted candidate's stored prediction, never reselect."""
        decision = self.property_distribution.decide(self.decision_policy)
        return {
            "direction": self.direction, "estimate": self.scale,
            "interval": {"lower": self.lower, "upper": self.upper},
            "reference_scale": self.reference_scale,
            "future_to_reference_ratio": self.ratio,
            "ratio_interval": {"lower": self.ratio_lower,
                               "upper": self.ratio_upper},
            "direction_probabilities": self.direction_probabilities,
            "direction_support": self.direction_support,
            "support": self.support,
            "automation_eligible": decision["automation_eligible"],
            "unit": "residual_scale",
            "executable": {"candidate": self.candidate,
                           "direction_candidate": self.direction_candidate,
                           "horizon": self.horizon},
            "property_distribution": self.property_distribution.to_dict(),
            "decision": decision,
        }


def fit_volatility_executable(
    values: list[float], *, horizon: int, season: int = 1,
    minimum_improvement: float = 0.02,
) -> FittedVolatilityExecutable:
    """Select by rolling origins and fit the exact executable published.

    Fold targets are realised robust residual scale over the next horizon.
    Candidate inputs end at the fold cutoff. The final forecast therefore
    cannot observe, directly or indirectly, any evaluation target.
    """
    numeric = [float(item) for item in values if math.isfinite(float(item))]
    errors = residuals(numeric, max(1, int(season)))
    candidates = (("constant", "recent", "ewma_slow", "ewma", "ewma_fast",
                   "scale_trend", "robust_scale_trend", "regime_mixture",
                   "scale_momentum",
                   "seasonal_phase") if season > 1 else
                  ("constant", "recent", "ewma_slow", "ewma", "ewma_fast",
                   "scale_trend", "robust_scale_trend", "regime_mixture",
                   "scale_momentum"))
    # Very long requested horizons must not erase all historical evidence.
    # Validate on the longest disjoint proxy horizon the observed residual
    # history can support, then execute the fitted scale model at the real
    # horizon. The proxy is recorded and never presented as full-horizon
    # calibration; it can inform a weak best estimate, never automation.
    target_span = min(max(8, horizon), max(8, len(errors) // 4))
    proxy_calibration = target_span < horizon
    minimum_history = max(
        24, (2 if proxy_calibration else 3) * target_span,
        2 * max(1, season))
    origins = list(range(minimum_history, len(errors) - target_span + 1,
                         max(1, horizon)))
    losses: dict[str, list[float]] = {name: [] for name in candidates}
    qlikes: dict[str, list[float]] = {name: [] for name in candidates}
    direction_pairs: dict[str, list[tuple[str, str]]] = {
        name: [] for name in candidates}
    ratios: dict[str, list[float]] = {name: [] for name in candidates}
    probability_losses: dict[str, list[float]] = {
        name: [] for name in candidates}
    probability_pairs: dict[str, list[tuple[str, str]]] = {
        name: [] for name in candidates}
    actual_directions: list[str] = []
    base_rate_losses: list[float] = []
    for origin in origins:
        train, future = errors[:origin], errors[origin:origin + target_span]
        actual = _scale(future)
        if actual <= 1e-12:
            continue
        reference_fold = _scale(train[-max(8, min(48, len(train) // 2)):])
        actual_direction = (_direction(actual / reference_fold)
                            if reference_fold > 1e-12 else "stable")
        if len(actual_directions) >= 3:
            base_rate_probabilities = {
                label: (actual_directions.count(label) + 1)
                / (len(actual_directions) + len(_DIRECTIONS))
                for label in _DIRECTIONS
            }
            base_rate_losses.append(_brier(
                base_rate_probabilities, actual_direction))
        for name in candidates:
            predicted = max(_predict(name, train, horizon, season), 1e-12)
            ratio = predicted / actual
            losses[name].append(abs(math.log(ratio)))
            variance_ratio = (predicted * predicted) / (actual * actual)
            qlikes[name].append(variance_ratio - math.log(variance_ratio) - 1)
            if reference_fold > 1e-12:
                actual_ratio, predicted_ratio = (actual / reference_fold,
                                                 predicted / reference_fold)
                direction_pairs[name].append((_direction(predicted_ratio),
                                              _direction(actual_ratio)))
                # Prequential calibration: the probability scored at this
                # fold may use earlier fold errors, never this fold's target.
                if len(ratios[name]) >= 3:
                    implied = [predicted_ratio / error
                               for error in ratios[name] if error > 1e-12]
                    probabilities = _smoothed_probabilities(implied)
                    predicted_state = max(
                        _DIRECTIONS,
                        key=lambda label: (probabilities[label],
                                           label == _direction(predicted_ratio)),
                    )
                    probability_losses[name].append(
                        _brier(probabilities, actual_direction))
                    probability_pairs[name].append(
                        (predicted_state, actual_direction))
            ratios[name].append(ratio)
        actual_directions.append(actual_direction)
    scores = {name: statistics.mean(items) for name, items in losses.items()
              if items}
    baseline = scores.get("constant", math.inf)
    best = min(scores, key=lambda name: (scores[name], name)) if scores else "constant"
    improvement = ((baseline - scores[best]) / baseline
                   if math.isfinite(baseline) and baseline > 0 else 0.0)
    advanced_supported = (len(origins) >= 3 and best != "constant"
                          and improvement >= minimum_improvement)
    scale_supported = len(origins) >= 3
    chosen = best if advanced_supported else "constant"
    prediction = max(_predict(chosen, errors, horizon, season), 0.0) if errors else 0.0
    reference = _scale(errors[-max(8, min(48, len(errors) // 2)):]) if errors else 0.0
    calibration = ratios.get(chosen, [])
    if len(calibration) >= 3:
        # Finite-sample conservative envelope. Regime transitions make a
        # smooth interpolated 10/90 interval anti-conservative precisely
        # where an operator needs caution most.
        ordered = sorted(calibration)
        low_ratio, high_ratio = ordered[0], ordered[-1]
        lower = prediction / max(high_ratio, 1e-12)
        upper = prediction / max(low_ratio, 1e-12)
    else:
        lower, upper = 0.0, prediction * 2
    ratio_to_reference = (prediction / reference if reference > 1e-12 else None)
    candidate_brier = {
        name: statistics.mean(items) for name, items in probability_losses.items()
        if items
    }
    candidate_balanced = {
        name: _balanced_accuracy(probability_pairs[name])
        for name in candidates if probability_pairs[name]
    }
    # Empty-sample baselines are None, never math.inf: these values reach the
    # published diagnostics, which must stay strict-JSON serializable.
    base_rate_brier = (statistics.mean(base_rate_losses)
                       if base_rate_losses else None)
    constant_brier = candidate_brier.get("constant")
    observed_baselines = [item for item in (base_rate_brier, constant_brier)
                          if item is not None]
    direction_baseline = min(observed_baselines) if observed_baselines else None
    direction_eligible = [
        name for name, loss in candidate_brier.items()
        if candidate_balanced.get(name, 0.0) > 1 / 3
        and direction_baseline is not None
        and loss <= direction_baseline * (1 - minimum_improvement)
    ]
    # Brier loss alone rewards the modal class on imbalanced histories. Among
    # candidates that genuinely beat the prequential probability baseline,
    # prefer repeatable per-class recall, then calibration. This uses only
    # past folds and is invariant to the caller's labels.
    best_direction = (min(
        direction_eligible,
        key=lambda name: (-candidate_balanced[name], candidate_brier[name], name))
        if direction_eligible else "constant")
    direction_brier_skill = (
        (direction_baseline - candidate_brier[best_direction]) / direction_baseline
        if direction_baseline is not None and direction_baseline > 0
        and best_direction in direction_eligible else 0.0
    )
    momentum_detected, momentum_change = _momentum_signal(errors)
    direction_candidate = (
        best_direction if direction_brier_skill >= minimum_improvement
        and candidate_balanced.get(best_direction, 0.0) > 1 / 3
        # Even when no classifier graduates, the exploratory best estimate
        # should be the category implied by the independently selected scale
        # executable. Its support remains weak and its Brier skill is zero;
        # this publishes useful magnitude without laundering it into an
        # automated directional claim.
        else "scale_momentum" if momentum_detected else chosen
    )
    direction_prediction = max(
        _predict(direction_candidate, errors, horizon, season), 1e-12)
    direction_ratio = (direction_prediction / reference
                       if reference > 1e-12 else None)
    calibration_candidate = direction_candidate
    direction_implied = [direction_ratio / error
                         for error in ratios.get(calibration_candidate, [])
                         if direction_ratio is not None and error > 1e-12]
    probabilities = _smoothed_probabilities(direction_implied)
    point_direction = (_direction(direction_ratio)
                       if direction_ratio is not None else "stable")
    direction_balanced = candidate_balanced.get(calibration_candidate, 0.0)
    effective_brier_skill = (direction_brier_skill
                             if direction_candidate == best_direction
                             and direction_brier_skill >= minimum_improvement
                             else 0.0)
    distribution_samples = direction_implied
    ratio_lower = (_quantile(distribution_samples, .1)
                   if distribution_samples else None)
    ratio_upper = (_quantile(distribution_samples, .9)
                   if distribution_samples else None)
    distribution = TemporalPropertyDistribution(
        quantity="future_to_reference_residual_scale_ratio",
        estimate=(ratio_to_reference or 0.0),
        lower=(ratio_lower if ratio_lower is not None else 0.0),
        upper=(ratio_upper if ratio_upper is not None else
               max(0.0, (ratio_to_reference or 0.0) * 2)),
        probabilities=probabilities,
        point_state=point_direction,
        folds=len(direction_implied),
        balanced_accuracy=direction_balanced,
        brier_skill=effective_brier_skill,
        support="supported" if scale_supported else "weak" if errors else "abstained",
    )
    policy = TemporalDecisionPolicy()
    decision = distribution.decide(policy)
    strongest = str(decision["best_state"])
    # With fewer than three prequential probability observations there is no
    # calibrated directional evidence.  Preserve the continuous scale and
    # ratio, but do not turn a noisy recent/reference ratio into a categorical
    # increase or decrease.  This is an estimator rule, independent of any
    # downstream task vocabulary.
    published_direction = (
        strongest if decision["automation_eligible"] else
        point_direction if distribution.folds >= 2 else "uncertain")
    return FittedVolatilityExecutable(
        candidate=chosen, scale=prediction, lower=max(0.0, lower),
        upper=max(lower, upper), horizon=horizon,
        support=("supported" if scale_supported else "weak"
                 if errors else "abstained"),
        scores=scores, baseline_score=baseline, improvement=improvement,
        training_observations=len(numeric),
        direction=published_direction if errors else "uncertain",
        reference_scale=reference, ratio=ratio_to_reference,
        ratio_lower=ratio_lower,
        ratio_upper=ratio_upper,
        direction_probabilities=probabilities,
        direction_support=str(decision["support"]),
        direction_candidate=direction_candidate,
        property_distribution=distribution,
        decision_policy=policy,
        diagnostics={
            "folds": len(origins), "target_span": target_span,
            "requested_horizon": horizon,
            "proxy_horizon_calibration": proxy_calibration,
            "selection_metric": "absolute_log_scale_error",
            "qlike_scores": {name: statistics.mean(items)
                             for name, items in qlikes.items() if items},
            "predicted_to_recent_scale": ratio_to_reference,
            "direction_probability_threshold": policy.minimum_probability,
            "advanced_candidate_supported": advanced_supported,
            "direction_minimum_matching_folds": policy.minimum_folds,
            "direction_matching_folds": distribution.folds,
            "fold_direction_accuracy": statistics.mean(
                predicted == actual for predicted, actual
                in direction_pairs.get(calibration_candidate, []))
                if direction_pairs.get(calibration_candidate) else 0.0,
            "fold_direction_balanced_accuracy": direction_balanced,
            "direction_candidate": direction_candidate,
            "direction_candidate_brier": candidate_brier,
            "direction_base_rate_brier": base_rate_brier,
            "direction_constant_brier": constant_brier,
            "direction_brier_skill": effective_brier_skill,
            "direction_momentum_detected": momentum_detected,
            "direction_momentum_log_change": momentum_change,
            "direction_publication_rule": (
                "calibrated_mode_when_automation_eligible_else_weak_point_state"),
            "calibration_ratios": len(calibration),
        },
    )
