from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .models import BASELINES, MODELS, predict


@dataclass(frozen=True)
class Evaluation:
    selected_model: str | None
    strongest_baseline: str | None
    selection_scores: dict[str, float | None]
    test_scores: dict[str, float | None]
    improvement: float | None
    residuals: list[float]
    coverage: float | None
    warnings: list[str]
    supported: bool
    degraded: bool = False


def error_score(actual: list[float], predicted: list[float]) -> float:
    absolute_error = sum(abs(a - p) for a, p in zip(actual, predicted))
    scale = sum(abs(a) for a in actual)
    return absolute_error / scale if scale > 1e-12 else absolute_error / len(actual)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def interval_bounds(
    point: float, residual_quantiles: dict[float, float], step: int
) -> tuple[float, float, float]:
    """Interval for forecast step ``step`` (1-based): the median residual shifts
    the centre, and the spread around it widens with sqrt(step) so uncertainty
    grows over the horizon instead of staying constant."""
    centre = point + residual_quantiles[0.5]
    scale = step ** 0.5
    low = centre + (residual_quantiles[0.1] - residual_quantiles[0.5]) * scale
    high = centre + (residual_quantiles[0.9] - residual_quantiles[0.5]) * scale
    return min(low, centre, high), centre, max(low, centre, high)


def _origins(length: int, horizon: int, minimum_train: int) -> list[int]:
    return list(range(minimum_train, length - horizon + 1, horizon))


def evaluate(
    values: list[float], horizon: int, season: int, minimum_improvement: float,
) -> Evaluation:
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = _origins(len(values), horizon, minimum_train)
    empty_scores = {name: None for name in MODELS}
    if len(origins) < 2:
        minimum_required = minimum_train + 2 * horizon
        full_required = minimum_train + 4 * horizon
        return Evaluation(
            None, None, empty_scores, empty_scores.copy(), None, [], None,
            [
                f"Need at least {minimum_required} observations (have {len(values)}) "
                f"for separated selection and calibration windows; "
                f"{full_required} observations enable fully separated selection, "
                f"calibration, and test windows."
            ],
            False,
        )

    # Full mode holds out both a calibration fold and a final test fold after
    # the selection folds. With only two or three folds we degrade gracefully
    # instead of refusing: fewer selection folds, and with two folds no
    # held-out test at all — each degradation is named in a warning.
    degraded = len(origins) < 4
    warnings: list[str] = []
    if len(origins) >= 3:
        selection_origins, calibration_origin = origins[:-2], origins[-2]
        test_origin: int | None = origins[-1]
    else:
        selection_origins, calibration_origin, test_origin = origins[:-1], origins[-1], None
    if degraded:
        warnings.append(
            f"Limited evaluation: only {len(origins)} rolling folds were available; "
            f"{minimum_train + 4 * horizon} observations enable fully separated "
            f"selection, calibration, and test windows."
        )

    fold_scores: dict[str, list[float]] = {name: [] for name in MODELS}
    for origin in selection_origins:
        actual = values[origin : origin + horizon]
        for name in MODELS:
            try:
                forecast = predict(name, values[:origin], horizon, season)
            except ValueError:
                continue
            fold_scores[name].append(error_score(actual, forecast))
    scores: dict[str, float | None] = {
        name: mean(items) if len(items) == len(selection_origins) else None
        for name, items in fold_scores.items()
    }
    baseline_scores = {name: score for name, score in scores.items() if name in BASELINES and score is not None}
    if not baseline_scores:
        return Evaluation(None, None, scores, empty_scores.copy(), None, [], None,
                          ["No baseline completed every selection fold."], False)
    strongest_baseline = min(baseline_scores, key=baseline_scores.get)  # type: ignore[arg-type]
    selected = strongest_baseline
    baseline_score = baseline_scores[strongest_baseline]
    candidates = {name: score for name, score in scores.items() if name not in BASELINES and score is not None}
    if candidates:
        candidate = min(candidates, key=candidates.get)  # type: ignore[arg-type]
        candidate_score = candidates[candidate]
        if baseline_score > 0 and candidate_score <= baseline_score * (1 - minimum_improvement):
            selected = candidate
    selected_score = scores[selected]
    improvement = 0.0 if selected in BASELINES else (baseline_score - selected_score) / baseline_score  # type: ignore[operator]

    # Pool residuals of the selected model across every selection fold plus
    # the calibration fold: one horizon of residuals is too small a sample
    # for stable quantiles.
    residuals: list[float] = []
    for origin in [*selection_origins, calibration_origin]:
        actual = values[origin : origin + horizon]
        prediction = predict(selected, values[:origin], horizon, season)
        residuals.extend(a - p for a, p in zip(actual, prediction))
    residual_quantiles = {p: quantile(residuals, p) for p in (0.1, 0.5, 0.9)}

    test_scores: dict[str, float | None] = {name: None for name in MODELS}
    coverage: float | None = None
    if test_origin is not None:
        test_actual = values[test_origin : test_origin + horizon]
        for name in {selected, strongest_baseline}:
            try:
                test_scores[name] = error_score(test_actual, predict(name, values[:test_origin], horizon, season))
            except ValueError:
                pass
        test_prediction = predict(selected, values[:test_origin], horizon, season)
        covered = []
        for step, (actual, prediction) in enumerate(zip(test_actual, test_prediction), 1):
            low, _, high = interval_bounds(prediction, residual_quantiles, step)
            covered.append(1.0 if low <= actual <= high else 0.0)
        coverage = mean(covered)
        if coverage < 0.7:
            warnings.append(f"Final-test 80% interval coverage was {coverage:.1%}, below 70%.")
    else:
        warnings.append(
            "Limited evaluation: no held-out test fold remained, so interval "
            "coverage is unmeasured."
        )
    return Evaluation(selected, strongest_baseline, scores, test_scores, improvement,
                      residuals, coverage, warnings, True, degraded)
