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


def _origins(length: int, horizon: int, minimum_train: int) -> list[int]:
    return list(range(minimum_train, length - horizon + 1, horizon))


def evaluate(
    values: list[float], horizon: int, season: int, minimum_improvement: float,
) -> Evaluation:
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = _origins(len(values), horizon, minimum_train)
    empty_scores = {name: None for name in MODELS}
    if len(origins) < 4:
        return Evaluation(
            None, None, empty_scores, empty_scores.copy(), None, [], None,
            [f"Need at least {minimum_train + 4 * horizon} observations for separated selection, calibration, and test windows."],
            False,
        )

    selection_origins, calibration_origin, test_origin = origins[:-2], origins[-2], origins[-1]
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

    calibration_actual = values[calibration_origin : calibration_origin + horizon]
    calibration_prediction = predict(selected, values[:calibration_origin], horizon, season)
    residuals = [actual - predicted for actual, predicted in zip(calibration_actual, calibration_prediction)]
    low, high = quantile(residuals, 0.1), quantile(residuals, 0.9)

    test_actual = values[test_origin : test_origin + horizon]
    test_scores: dict[str, float | None] = {name: None for name in MODELS}
    for name in {selected, strongest_baseline}:
        try:
            test_scores[name] = error_score(test_actual, predict(name, values[:test_origin], horizon, season))
        except ValueError:
            pass
    test_prediction = predict(selected, values[:test_origin], horizon, season)
    coverage = mean(
        1.0 if prediction + low <= actual <= prediction + high else 0.0
        for actual, prediction in zip(test_actual, test_prediction)
    )
    warnings: list[str] = []
    if coverage < 0.7:
        warnings.append(f"Final-test 80% interval coverage was {coverage:.1%}, below 70%.")
    return Evaluation(selected, strongest_baseline, scores, test_scores, improvement,
                      residuals, coverage, warnings, True)

