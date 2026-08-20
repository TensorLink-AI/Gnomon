"""Fold-safe fitted executables for predictive temporal properties.

The module intentionally fits small, inspectable candidates.  It is not a
second point-forecast engine: it answers questions *about* a future path and
keeps those answers additive to the immutable primary forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _scale(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    centre = _median(xs)
    mad = _median([abs(x - centre) for x in xs])
    return 1.4826 * mad if mad > 1e-12 else float(statistics.stdev(xs))


def _slope(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    centre = (len(xs) - 1) / 2
    denom = sum((i - centre) ** 2 for i in range(len(xs)))
    return (sum((i - centre) * x for i, x in enumerate(xs)) / denom
            if denom else 0.0)


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 4:
        return None
    lm, rm = statistics.mean(left), statistics.mean(right)
    denom = math.sqrt(sum((x - lm) ** 2 for x in left)
                      * sum((x - rm) ** 2 for x in right))
    return (sum((x - lm) * (y - rm) for x, y in zip(left, right)) / denom
            if denom > 1e-12 else None)


def _innovation_scale(history: list[float], season: int) -> float:
    """Scale of unpredictable movement, excluding stable level/seasonality."""
    lag = season if season > 1 and len(history) >= 2 * season else 1
    changes = [history[i] - history[i - lag] for i in range(lag, len(history))]
    # Difference noise contains two independent errors.
    return max(_scale(changes) / math.sqrt(2), 1e-12)


def _path(kind: str, history: list[float], horizon: int, season: int) -> list[float]:
    if kind == "last":
        return [history[-1]] * horizon
    if kind == "median":
        return [_median(history[-min(48, len(history)):])] * horizon
    if kind in {"drift", "damped_drift"}:
        width = min(max(8, 2 * max(1, season)), len(history))
        slope = _slope(history[-width:])
        damping = .85 if kind == "damped_drift" else 1.0
        return [history[-1] + slope * sum(damping ** j for j in range(step + 1))
                for step in range(horizon)]
    if kind == "seasonal" and season > 1 and len(history) >= 2 * season:
        base = history[-season:]
        return [base[i % season] for i in range(horizon)]
    raise ValueError(kind)


def _candidates(season: int, length: int) -> tuple[str, ...]:
    result = ["last", "median", "drift", "damped_drift"]
    if season > 1 and length >= 2 * season:
        result.append("seasonal")
    return tuple(result)


def _extreme_prediction(kind: str, history: list[float], horizon: int,
                        season: int) -> float:
    lag = season if season > 1 and len(history) >= 2 * season else 1
    innovations = [history[i] - history[i - lag] for i in range(lag, len(history))]
    scale = _innovation_scale(history, season)
    standardized = [abs(value) / (math.sqrt(2) * scale) for value in innovations]
    maxima = [max(standardized[i:i + horizon])
              for i in range(0, len(standardized) - horizon + 1, horizon)]
    if not maxima:
        return max(standardized, default=0.0)
    if kind == "block_max_median":
        return _median(maxima)
    if kind == "block_max_recent":
        return maxima[-1]
    if kind == "block_max_upper":
        ordered = sorted(maxima)
        return ordered[min(len(ordered) - 1, math.ceil(.8 * len(ordered)) - 1)]
    raise ValueError(kind)


def _candidate_names(prop: str, season: int, length: int) -> tuple[str, ...]:
    if prop == "extreme":
        return ("block_max_median", "block_max_recent", "block_max_upper")
    return _candidates(season, length)


def _predict_property(prop: str, kind: str, history: list[float],
                      horizon: int, season: int) -> float:
    if prop == "extreme":
        return _extreme_prediction(kind, history, horizon, season)
    return _property_value(prop, history, _path(kind, history, horizon, season), season)


def _direction(value: float, threshold: float, labels: tuple[str, str, str]) -> str:
    return labels[2] if value > threshold else labels[0] if value < -threshold else labels[1]


def _property_value(prop: str, history: list[float], future: list[float],
                    season: int) -> float:
    recent = history[-min(max(8, season), len(history)):]
    scale = _innovation_scale(history, season)
    if prop == "level":
        return (_median(future) - _median(recent)) / scale
    if prop == "trend":
        return _slope(future) / scale
    if prop == "seasonality":
        if season <= 1 or len(history) < season:
            return 0.0
        expected = [history[-season + i % season] for i in range(len(future))]
        return _correlation(expected, future) or 0.0
    if prop == "regime":
        return abs(_median(future) - _median(recent)) / scale
    if prop == "extreme":
        expected = (_path("seasonal", history, len(future), season)
                    if season > 1 and len(history) >= 2 * season
                    else _path("damped_drift", history, len(future), season))
        return max(abs(x - centre) for x, centre in zip(future, expected)) / scale
    raise ValueError(prop)


_LABELS = {
    "level": ("lower", "similar", "higher"),
    "trend": ("downward", "constant", "upward"),
    "seasonality": ("weaker", "uncertain", "continued"),
    "regime": ("no_shift", "uncertain", "shift"),
    "extreme": ("unlikely", "uncertain", "elevated"),
}

_THRESHOLDS = {"level": .25, "trend": .02, "seasonality": .5,
               "regime": 1.0, "extreme": 3.5}


def _label(prop: str, value: float) -> str:
    if prop in {"seasonality", "regime", "extreme"}:
        threshold = _THRESHOLDS[prop]
        return _LABELS[prop][2] if value >= threshold else _LABELS[prop][0]
    return _direction(value, _THRESHOLDS[prop], _LABELS[prop])


@dataclass(frozen=True)
class FittedTemporalExecutable:
    property: str
    candidate: str
    horizon: int
    estimate: float
    lower: float
    upper: float
    direction: str
    support: str
    probabilities: dict[str, float]
    diagnostics: dict[str, Any]

    def execute(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "estimate": self.estimate,
            "interval": {"lower": self.lower, "upper": self.upper},
            "support": self.support,
            "automation_eligible": self.support == "supported",
            "direction_probabilities": self.probabilities,
            "executable": {
                "kind": "fitted_temporal_property",
                "property": self.property,
                "candidate": self.candidate,
                "horizon": self.horizon,
                "version": "0.1",
            },
            "diagnostics": self.diagnostics,
        }


def fit_temporal_executable(values: list[float], *, property: str,
                            horizon: int, season: int = 1,
                            minimum_folds: int = 8,
                            minimum_skill: float = .02) -> FittedTemporalExecutable:
    """Select and calibrate a property executable using rolling origins only."""
    if property not in _LABELS:
        raise ValueError(f"unsupported temporal property: {property}")
    numeric = [float(x) for x in values if math.isfinite(float(x))]
    horizon, season = max(1, int(horizon)), max(1, int(season))
    if not numeric:
        labels = tuple(dict.fromkeys(_LABELS[property]))
        return FittedTemporalExecutable(
            property=property, candidate="none", horizon=horizon,
            estimate=0.0, lower=0.0, upper=0.0, direction="uncertain",
            support="abstained",
            probabilities={label: 1 / len(labels) for label in labels},
            diagnostics={
                "folds": 0, "reason": "no_finite_observations",
                "primary_forecast_unchanged": True,
            },
        )
    minimum_history = max(32, 3 * horizon, 2 * season)
    origins = list(range(minimum_history, len(numeric) - horizon + 1,
                         max(1, horizon)))
    names = _candidate_names(property, season, len(numeric))
    losses: dict[str, list[float]] = {name: [] for name in names}
    errors: dict[str, list[float]] = {name: [] for name in names}
    labels: dict[str, list[tuple[str, str]]] = {name: [] for name in names}
    for origin in origins:
        train, actual_path = numeric[:origin], numeric[origin:origin + horizon]
        actual = _property_value(property, train, actual_path, season)
        for name in names:
            predicted = _predict_property(property, name, train, horizon, season)
            error = predicted - actual
            errors[name].append(error)
            losses[name].append(abs(error))
            labels[name].append((_label(property, predicted), _label(property, actual)))
    scores = {name: statistics.mean(loss) for name, loss in losses.items() if loss}
    baseline_name = "block_max_median" if property == "extreme" else "last"
    baseline = scores.get(baseline_name, math.inf)
    best = (min(scores, key=lambda name: (scores[name], name))
            if scores else baseline_name)
    skill = ((baseline - scores[best]) / baseline
             if math.isfinite(baseline) and baseline > 1e-12 else 0.0)
    chosen = (best if best != baseline_name and skill >= minimum_skill
              else baseline_name)
    estimate = (_predict_property(property, chosen, numeric, horizon, season)
                if numeric else 0.0)
    calibration = errors.get(chosen, [])
    implied = [estimate - error for error in calibration]
    if len(implied) >= minimum_folds:
        # Rolling folds are overlapping and temporal regimes are not iid.
        # A modest finite-sample expansion is safer than presenting their
        # raw extrema as a nominal interval, especially for ratio-scaled
        # regime and tail quantities.
        # Discontinuities and tail quantities are more regime-sensitive than
        # smooth level/trend properties. Their rolling errors need a larger
        # finite-sample envelope; this affects uncertainty only, never the
        # selected candidate or categorical best estimate.
        expansion = 2.0 if property in {"regime", "extreme"} else 1.5
        radius = expansion * max(abs(error) for error in calibration)
        lower, upper = estimate - radius, estimate + radius
    else:
        lower, upper = estimate, estimate
    public_labels = tuple(dict.fromkeys(_LABELS[property]))
    counts = {label: sum(_label(property, value) == label for value in implied)
              for label in public_labels}
    probabilities = {label: (counts[label] + 1) / (len(implied) + len(public_labels))
                     for label in public_labels}
    fold_accuracy = (statistics.mean(predicted == actual
                                     for predicted, actual in labels.get(chosen, []))
                     if labels.get(chosen) else 0.0)
    point_label = _label(property, estimate)
    strongest = max(
        public_labels,
        key=lambda item: (probabilities[item], item == point_label),
    )
    claim = (len(implied) >= minimum_folds and fold_accuracy >= .8
             and probabilities[strongest] >= .85)
    interval_direction_consistent = (
        _label(property, lower) == strongest == _label(property, upper))
    # A categorical directional claim is stronger than publishing a point
    # estimate. If the calibrated interval crosses a decision boundary, the
    # point label remains useful but cannot be called supported.
    if property in {"level", "trend"} and not interval_direction_consistent:
        claim = False
    # Absence of a future discontinuity is not an affirmative prediction.
    # A finite history can support a base-rate estimate, but not a categorical
    # "nothing surprising will happen" claim. Keep the estimate/distribution
    # and abstain unless the fitted executable positively predicts the event.
    if property in {"regime", "extreme"} and strongest in {"no_shift", "unlikely"}:
        claim = False
    return FittedTemporalExecutable(
        property=property, candidate=chosen, horizon=horizon,
        estimate=estimate, lower=lower, upper=upper,
        # A failed publication gate withholds automation, not the best
        # computable estimate.  The distribution and weak support make the
        # uncertainty explicit without forcing an exploratory user to guess.
        direction=strongest,
        support="supported" if claim else "weak",
        probabilities=probabilities,
        diagnostics={
            "folds": len(origins), "selection_metric": "property_absolute_error",
            "candidate_scores": scores, "baseline_candidate": baseline_name,
            "skill_vs_baseline": skill, "fold_direction_accuracy": fold_accuracy,
            "minimum_folds": minimum_folds, "primary_forecast_unchanged": True,
            "interval_direction_consistent": interval_direction_consistent,
        },
    )


@dataclass(frozen=True)
class FittedDependenceExecutable:
    candidate: str
    horizon: int
    estimate: float | None
    lower: float | None
    upper: float | None
    direction: str
    support: str
    probabilities: dict[str, float]
    diagnostics: dict[str, Any]

    def execute(self) -> dict[str, Any]:
        return {
            "direction": self.direction, "estimate": self.estimate,
            "interval": {"lower": self.lower, "upper": self.upper},
            "support": self.support,
            "automation_eligible": self.support == "supported",
            "direction_probabilities": self.probabilities,
            "executable": {"kind": "fitted_dependence", "candidate": self.candidate,
                           "horizon": self.horizon, "version": "0.1"},
            "diagnostics": self.diagnostics,
        }


def fit_dependence_executable(left: list[float], right: list[float], *,
                              horizon: int, minimum_folds: int = 5
                              ) -> FittedDependenceExecutable:
    """Forecast paired return correlation with fold-safe window selection."""
    n = min(len(left), len(right))
    x, y = [float(v) for v in left[-n:]], [float(v) for v in right[-n:]]
    dx = [x[i] - x[i - 1] for i in range(1, n)]
    dy = [y[i] - y[i - 1] for i in range(1, n)]
    horizon = max(4, int(horizon))
    windows = (24, 48, 96)
    origins = list(range(max(windows), len(dx) - horizon + 1, horizon))
    losses = {str(window): [] for window in windows}
    errors = {str(window): [] for window in windows}
    for origin in origins:
        actual = _correlation(dx[origin:origin + horizon], dy[origin:origin + horizon])
        if actual is None:
            continue
        for window in windows:
            predicted = _correlation(dx[origin-window:origin], dy[origin-window:origin])
            if predicted is not None:
                losses[str(window)].append(abs(predicted - actual))
                errors[str(window)].append(predicted - actual)
    scores = {key: statistics.mean(value) for key, value in losses.items() if value}
    chosen = min(scores, key=lambda key: (scores[key], key)) if scores else "48"
    estimate = _correlation(dx[-int(chosen):], dy[-int(chosen):])
    implied = ([estimate - error for error in errors.get(chosen, [])]
               if estimate is not None else [])
    labels = ("negative", "weak", "positive")
    classify: Callable[[float], str] = lambda value: (
        "positive" if value >= .3 else "negative" if value <= -.3 else "weak")
    counts = {label: sum(classify(value) == label for value in implied) for label in labels}
    probabilities = {label: (counts[label] + 1) / (len(implied) + 3) for label in labels}
    point_label = classify(estimate) if estimate is not None else None
    strongest = max(
        labels,
        key=lambda label: (probabilities[label], label == point_label),
    )
    claim = (estimate is not None and len(implied) >= minimum_folds
             and probabilities[strongest] >= .7)
    return FittedDependenceExecutable(
        candidate=f"rolling_correlation_{chosen}", horizon=horizon,
        estimate=estimate, lower=min(implied) if implied else None,
        upper=max(implied) if implied else None,
        direction=strongest if estimate is not None else "uncertain",
        support=("supported" if claim else "weak"
                 if estimate is not None else "abstained"),
        probabilities=probabilities,
        diagnostics={"folds": len(origins), "candidate_scores": scores,
                     "quantity": "paired_first_difference_correlation",
                     "primary_forecast_unchanged": True},
    )
