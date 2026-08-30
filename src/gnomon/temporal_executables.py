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


def _phase_fixed_slope(xs: list[float], season: int) -> float | None:
    """Slope after removing one intercept per seasonal phase.

    A raw OLS slope over one or two cycles depends on where the seasonal arc
    begins. Phase fixed effects make the requested quantity the underlying
    drift while leaving the caller's admitted period authoritative.
    """
    if len(xs) < 2:
        return None
    if season <= 1:
        return _slope(xs)
    if len(xs) < 2 * season:
        return None
    phase_means = {
        phase: statistics.mean(xs[phase::season])
        for phase in range(season)
    }
    phase_times = {
        phase: statistics.mean(range(phase, len(xs), season))
        for phase in range(season)
    }
    numerator = denominator = 0.0
    for index, value in enumerate(xs):
        phase = index % season
        centred_time = index - phase_times[phase]
        numerator += centred_time * (value - phase_means[phase])
        denominator += centred_time * centred_time
    return numerator / denominator if denominator else None


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


def _historical_trend_prediction(kind: str, history: list[float],
                                 horizon: int, season: int) -> float:
    """Predict normalized underlying drift without scoring a seasonal arc."""
    if kind in {"last", "median", "seasonal"}:
        return 0.0
    width = min(len(history), max(8, 4 * season, 2 * horizon))
    slope = _phase_fixed_slope(history[-width:], season)
    if slope is None:
        return 0.0
    if kind == "damped_drift":
        # Mean one-step damping across the requested horizon. This preserves
        # the existing damped candidate's semantics without constructing a
        # point path whose seasonal phase can masquerade as trend.
        slope *= statistics.mean(.85 ** step for step in range(horizon))
    return slope / _innovation_scale(history, season)


def _future_trend_value(history: list[float], future: list[float],
                        season: int) -> float:
    """Realized normalized drift, using only an admitted seasonal period."""
    if season <= 1:
        slope = _slope(future)
    else:
        slope = _phase_fixed_slope(future, season)
        if slope is None:
            # A short future cannot fit phase effects internally. Compare it
            # with the same phases in the visible history. A full seasonal
            # cycle averages phase-specific amplitude effects; a shorter arc
            # remains calibration evidence only and cannot become supported.
            combined = [*history, *future]
            changes = [
                (combined[len(history) + index]
                 - combined[len(history) + index - season]) / season
                for index in range(len(future))
                if len(history) + index >= season
            ]
            slope = statistics.mean(changes) if changes else 0.0
    return slope / _innovation_scale(history, season)


def _predict_property(prop: str, kind: str, history: list[float],
                      horizon: int, season: int) -> float:
    if prop == "extreme":
        return _extreme_prediction(kind, history, horizon, season)
    if prop == "trend":
        return _historical_trend_prediction(
            kind, history, horizon, season)
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
        return _future_trend_value(history, future, season)
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
    estimate: float | None
    lower: float | None
    upper: float | None
    direction: str
    support: str
    probabilities: dict[str, float]
    diagnostics: dict[str, Any]

    def execute(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "estimate": self.estimate,
            "interval": ({"lower": self.lower, "upper": self.upper}
                         if self.lower is not None and self.upper is not None
                         else None),
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
    if property == "trend" and season > 1 and len(numeric) < 2 * season:
        labels = tuple(dict.fromkeys(_LABELS[property]))
        return FittedTemporalExecutable(
            property=property, candidate="none", horizon=horizon,
            estimate=None, lower=None, upper=None, direction="uncertain",
            support="abstained",
            probabilities={label: 1 / len(labels) for label in labels},
            diagnostics={
                "folds": 0,
                "reason": "insufficient_cycles_for_seasonally_adjusted_trend",
                "visible_cycles": len(numeric) / season,
                "required_cycles": 2,
                "admitted_period": season,
                "primary_forecast_unchanged": True,
            },
        )
    unmodelled_seasonality: dict[str, Any] | None = None
    if property == "trend" and season == 1 and len(numeric) >= 8:
        # Detection is evidence, not authority to replace the caller's
        # admitted period. Only strong measured autocorrelation enters this
        # diagnostic; the executable continues to run with season=1.
        from .temporal import detect_season
        detected_period, detected_strength, detected_basis = detect_season(
            numeric, "unknown")
        if (detected_basis == "autocorrelation"
                and detected_period > 1
                and detected_strength >= .8
                and len(numeric) >= 2 * detected_period):
            adjusted_estimate = _historical_trend_prediction(
                "drift", numeric, horizon, detected_period)
            unmodelled_seasonality = {
                "detected_period": detected_period,
                "strength": detected_strength,
                "basis": detected_basis,
                "declared_period": season,
                "adjusted_direction_diagnostic": _label(
                    "trend", adjusted_estimate),
                "adjusted_estimate_diagnostic": adjusted_estimate,
                "period_was_not_silently_admitted": True,
            }
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
    if len(implied) >= minimum_folds or (property == "trend" and implied):
        # Rolling folds are overlapping and temporal regimes are not iid.
        # A modest finite-sample expansion is safer than presenting their
        # raw extrema as a nominal interval, especially for ratio-scaled
        # regime and tail quantities.
        # Discontinuities and tail quantities are more regime-sensitive than
        # smooth level/trend properties. Their rolling errors need a larger
        # finite-sample envelope; this affects uncertainty only, never the
        # selected candidate or categorical best estimate.
        expansion = (3.0 if property == "regime" else
                     2.0 if property == "extreme" else 1.5)
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
    calibration_direction_disagreement = (
        property == "trend" and strongest != point_label)
    if calibration_direction_disagreement:
        claim = False
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
    unmodelled_disagreement = bool(
        unmodelled_seasonality
        and unmodelled_seasonality["adjusted_direction_diagnostic"]
        != point_label)
    if unmodelled_seasonality:
        # Even agreement cannot make an unadmitted adjustment automatic.
        claim = False
        unmodelled_seasonality["direction_disagreement"] = (
            unmodelled_disagreement)
    public_estimate: float | None = estimate
    public_lower: float | None = lower
    public_upper: float | None = upper
    public_direction = point_label if property == "trend" else strongest
    public_support = "supported" if claim else "weak"
    if unmodelled_disagreement:
        public_estimate = public_lower = public_upper = None
        public_direction = "uncertain"
        public_support = "abstained"
    return FittedTemporalExecutable(
        property=property, candidate=chosen, horizon=horizon,
        estimate=public_estimate, lower=public_lower, upper=public_upper,
        # A failed publication gate withholds automation, not the best
        # computable estimate.  The distribution and weak support make the
        # uncertainty explicit without forcing an exploratory user to guess.
        direction=public_direction,
        support=public_support,
        probabilities=probabilities,
        diagnostics={
            "folds": len(origins), "selection_metric": "property_absolute_error",
            "candidate_scores": scores, "baseline_candidate": baseline_name,
            "skill_vs_baseline": skill, "fold_direction_accuracy": fold_accuracy,
            "minimum_folds": minimum_folds, "primary_forecast_unchanged": True,
            "interval_direction_consistent": interval_direction_consistent,
            "calibration_modal_direction": strongest,
            "point_estimate_direction": point_label,
            "calibration_direction_disagreement": (
                calibration_direction_disagreement),
            **({"unmodelled_seasonality": unmodelled_seasonality}
               if unmodelled_seasonality else {}),
            **({"reason": "unmodelled_seasonality_changes_trend_direction"}
               if unmodelled_disagreement else {}),
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


def _seasonal_phase_state(reference: list[float], candidate: list[float],
                          season: int) -> tuple[str, int | None, float | None]:
    """Compare two observed windows; never inspect a point forecast."""
    if season <= 1 or len(reference) < 2 * season or len(candidate) < season:
        return "uncertain", None, None
    template = [_median(reference[offset::season]) for offset in range(season)]
    correlations: list[tuple[int, float]] = []
    for shift in range(season):
        # Candidate follows reference in time. Preserve that phase offset when
        # window length is not an exact multiple of the seasonal period.
        expected = [template[(len(reference) + index - shift) % season]
                    for index in range(len(candidate))]
        correlation = _correlation(expected, candidate)
        if correlation is not None:
            correlations.append((shift, correlation))
    if not correlations:
        return "uncertain", None, None
    zero = next((value for shift, value in correlations if shift == 0), None)
    best_shift, best = max(correlations, key=lambda item: item[1])
    distance = min(best_shift, season - best_shift)
    tolerance = max(1, round(.1 * season))
    if zero is not None and zero >= .45 and (distance <= tolerance
                                              or best - zero < .15):
        return "fixed", 0, zero
    if best >= .55 and distance > tolerance \
            and (zero is None or best - zero >= .15):
        return "shifting", best_shift, best
    return "uncertain", best_shift, best


@dataclass(frozen=True)
class FittedSeasonalityExecutable:
    direction: str
    support: str
    probabilities: dict[str, float]
    phase_shift_steps: int | None
    diagnostics: dict[str, Any]

    def execute(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "estimate": {"phase_shift_steps": self.phase_shift_steps},
            "interval": None, "support": self.support,
            "automation_eligible": self.support == "supported",
            "direction_probabilities": self.probabilities,
            "executable": {"kind": "fitted_future_seasonality",
                           "property": "seasonality", "version": "0.1"},
            "diagnostics": self.diagnostics,
        }


def fit_future_seasonality_executable(
    values: list[float], *, horizon: int, season: int,
    minimum_folds: int = 5,
) -> FittedSeasonalityExecutable:
    """Forecast fixed-vs-shifting seasonality with ordered validation.

    The candidate is persistence of the most recently observed phase state.
    Rolling origins estimate its confusion distribution. Point-forecast
    alignment is intentionally absent from this executable.
    """
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    season, horizon = max(1, int(season)), max(1, int(horizon))
    width = max(2 * season, horizon)
    origins = list(range(2 * width, len(numeric) - horizon + 1,
                         max(1, horizon)))
    pairs: list[tuple[str, str]] = []
    for origin in origins:
        prior = numeric[origin - 2 * width:origin - width]
        recent = numeric[origin - width:origin]
        future = numeric[origin:origin + horizon]
        predicted, _, _ = _seasonal_phase_state(prior, recent, season)
        actual, _, _ = _seasonal_phase_state(recent, future, season)
        pairs.append((predicted, actual))
    current, current_shift, _ = _seasonal_phase_state(
        numeric[-2 * width:-width], numeric[-width:], season)
    labels = ("fixed", "shifting", "uncertain")
    matching_actual = [actual for predicted, actual in pairs
                       if predicted == current]
    counts = {label: matching_actual.count(label) for label in labels}
    probabilities = {label: (counts[label] + 1) /
                     (len(matching_actual) + len(labels)) for label in labels}
    selected = max(labels, key=lambda label: (
        probabilities[label], label == current))
    accuracy = (statistics.mean(predicted == actual for predicted, actual in pairs)
                if pairs else 0.0)
    supported = (len(pairs) >= minimum_folds and accuracy >= .7
                 and probabilities[selected] >= .7
                 and selected != "uncertain")
    # A process that failed to preserve its phase-state classification out of
    # sample has no defensible directional forecast. Preserve the estimated
    # distribution in the receipt, but abstain instead of publishing whichever
    # smoothed class happened to be largest.
    direction = ("uncertain" if pairs and accuracy < .5 else
                 selected if pairs else current)
    return FittedSeasonalityExecutable(
        direction=direction, support=("supported" if supported else "weak"
                                      if direction != "uncertain" else "abstained"),
        probabilities=probabilities, phase_shift_steps=current_shift,
        diagnostics={"folds": len(pairs), "fold_direction_accuracy": accuracy,
                     "quantity": "future_process_seasonal_phase_state",
                     "point_forecast_used": False,
                     "primary_forecast_unchanged": True},
    )
