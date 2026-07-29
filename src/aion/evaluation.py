from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from .models import BASELINES, MODELS, predict
from .tsfm import TSFMError, TSFMUnavailable, tsfm_candidates

logger = logging.getLogger(__name__)


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
    tsfm_scores: dict[str, float | None] = field(default_factory=dict)


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
    values: list[float],
    horizon: int,
    season: int,
    minimum_improvement: float,
    *,
    frequency: str = "h",
    tsfm_names: list[str] | None = None,
    config: Any = None,
) -> Evaluation:
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = _origins(len(values), horizon, minimum_train)
    all_model_names = list(MODELS.keys())
    empty_scores = {name: None for name in all_model_names}
    if len(origins) < 4:
        return Evaluation(
            None, None, empty_scores, empty_scores.copy(), None, [], None,
            [f"Need at least {minimum_train + 4 * horizon} observations for separated selection, calibration, and test windows."],
            False,
        )

    # --- Resolve config ---
    ensemble_enabled = getattr(config, "ensemble", None) and config.ensemble.enabled if config else False
    ensemble_strategy = config.ensemble.strategy if config and ensemble_enabled else "weighted_mean"
    ensemble_cfg = config.ensemble if config else None
    meta_model_enabled = getattr(config, "meta_model", None) and config.meta_model.enabled if config else False
    meta_model_cfg = config.meta_model if config and meta_model_enabled else None

    # --- Load TSFM candidates (lazy, graceful) ---
    # Prefer sandboxed adapters (isolated venvs) to avoid dependency conflicts.
    # Fall back to in-process adapters if no sandboxes are set up.
    from .tsfm_sandbox import sandbox_tsfm_candidates, sandbox_available_tsfms
    sandbox_names = sandbox_available_tsfms()
    if sandbox_names:
        tsfm_adapters = sandbox_tsfm_candidates(
            requested=tsfm_names if tsfm_names else None,
            frequency=frequency,
        )
    elif tsfm_names is None:
        tsfm_adapters = tsfm_candidates(frequency=frequency)
    elif tsfm_names:
        tsfm_adapters = tsfm_candidates(requested=tsfm_names, frequency=frequency)
    tsfm_model_names = [a.name for a in tsfm_adapters]
    all_model_names = list(MODELS.keys()) + tsfm_model_names

    # --- API inference adapters ---
    if config and config.backends.api.enabled:
        from .api_inference import APIAdapter
        for name, provider_cfg in config.backends.api.providers.items():
            if name in (tsfm_names or config.models.tsfm_candidates or []):
                try:
                    api_adapter = APIAdapter(name, provider_cfg)
                    tsfm_adapters.append(api_adapter)
                    if name not in all_model_names:
                        all_model_names.append(name)
                except Exception:
                    logger.debug("API adapter %s failed to initialize", name, exc_info=True)

    selection_origins, calibration_origin, test_origin = origins[:-2], origins[-2], origins[-1]

    # --- Run built-in models on selection folds ---
    fold_scores: dict[str, list[float]] = {name: [] for name in MODELS}
    # Store per-fold forecasts for ensemble/meta-model training
    fold_forecasts: dict[str, list[list[float]]] = {name: [] for name in MODELS}
    for origin in selection_origins:
        actual = values[origin : origin + horizon]
        for name in MODELS:
            try:
                forecast = predict(name, values[:origin], horizon, season)
                fold_forecasts[name].append(forecast)
            except ValueError:
                continue
            fold_scores[name].append(error_score(actual, forecast))

    # --- Run TSFM candidates on selection folds ---
    tsfm_fold_scores: dict[str, list[float]] = {a.name: [] for a in tsfm_adapters}
    tsfm_fold_forecasts: dict[str, list[list[float]]] = {a.name: [] for a in tsfm_adapters}
    for adapter in tsfm_adapters:
        for origin in selection_origins:
            actual = values[origin : origin + horizon]
            train = values[:origin]
            try:
                forecast = adapter.predict(train, horizon, season)
                if len(forecast) != horizon:
                    tsfm_fold_scores[adapter.name].append(None)  # type: ignore[arg-type]
                    tsfm_fold_forecasts[adapter.name].append([])  # type: ignore[arg-type]
                    continue
                tsfm_fold_scores[adapter.name].append(error_score(actual, forecast))
                tsfm_fold_forecasts[adapter.name].append(forecast)
            except (TSFMError, TSFMUnavailable, Exception) as exc:
                logger.debug("TSFM %s failed on fold at origin %d: %s", adapter.name, origin, exc)
                tsfm_fold_scores[adapter.name].append(None)  # type: ignore[arg-type]
                tsfm_fold_forecasts[adapter.name].append([])  # type: ignore[arg-type]

    # --- Aggregate scores ---
    scores: dict[str, float | None] = {
        name: mean(items) if items and len(items) == len(selection_origins) else None
        for name, items in fold_scores.items()
    }
    tsfm_scores: dict[str, float | None] = {}
    for name, items in tsfm_fold_scores.items():
        valid = [x for x in items if x is not None]
        if valid and len(valid) == len(selection_origins):
            tsfm_scores[name] = mean(valid)
        else:
            tsfm_scores[name] = None

    # --- Compute ensemble forecast on selection folds ---
    ensemble_fold_scores: list[float | None] = []
    if ensemble_enabled or meta_model_enabled:
        from .ensemble import compute_ensemble_forecast, ENSEMBLE_MODEL_NAME
        all_valid_forecasts: dict[str, list[float]] = {}
        all_valid_scores: dict[str, float | None] = {**scores, **tsfm_scores}

        for fold_idx in range(len(selection_origins)):
            fold_forecast_map: dict[str, list[float]] = {}
            for name in MODELS:
                if fold_idx < len(fold_forecasts[name]) and fold_forecasts[name][fold_idx]:
                    fold_forecast_map[name] = fold_forecasts[name][fold_idx]
            for adapter in tsfm_adapters:
                if fold_idx < len(tsfm_fold_forecasts[adapter.name]) and tsfm_fold_forecasts[adapter.name][fold_idx]:
                    fold_forecast_map[adapter.name] = tsfm_fold_forecasts[adapter.name][fold_idx]

            if len(fold_forecast_map) >= (ensemble_cfg.min_models if ensemble_cfg else 2):
                try:
                    combined = compute_ensemble_forecast(
                        fold_forecast_map, all_valid_scores,
                        strategy=ensemble_strategy,
                        last_observed=values[selection_origins[fold_idx] - 1] if selection_origins else 0.0,
                        config=ensemble_cfg,
                    )
                    actual = values[selection_origins[fold_idx] : selection_origins[fold_idx] + horizon]
                    ensemble_fold_scores.append(error_score(actual, combined))
                except Exception:
                    ensemble_fold_scores.append(None)
            else:
                ensemble_fold_scores.append(None)

    ensemble_score: float | None = None
    if ensemble_enabled and ensemble_fold_scores:
        valid = [x for x in ensemble_fold_scores if x is not None]
        if valid and len(valid) == len(selection_origins):
            ensemble_score = mean(valid)
            all_model_names.append(ENSEMBLE_MODEL_NAME if 'ENSEMBLE_MODEL_NAME' in dir() else "ensemble")

    # --- Meta-model training ---
    meta_model_weights: dict[str, float] | None = None
    meta_model_score: float | None = None
    if meta_model_enabled:
        from .meta_model import train_meta_model, predict_meta_model, META_MODEL_NAME
        # Collect fold forecasts and actuals for training
        mm_fold_forecasts: dict[str, list[list[float]]] = {}
        mm_fold_actuals: list[list[float]] = []
        for fold_idx in range(len(selection_origins)):
            origin = selection_origins[fold_idx]
            actual = values[origin : origin + horizon]
            mm_fold_actuals.append(actual)

        for name in MODELS:
            if fold_forecasts[name]:
                mm_fold_forecasts[name] = fold_forecasts[name]
        for adapter in tsfm_adapters:
            valid_forecasts = [f for f in tsfm_fold_forecasts[adapter.name] if f]
            if valid_forecasts:
                mm_fold_forecasts[adapter.name] = tsfm_fold_forecasts[adapter.name]

        if len(mm_fold_forecasts) >= (meta_model_cfg.min_models if meta_model_cfg else 2):
            meta_model_weights = train_meta_model(
                mm_fold_forecasts,
                mm_fold_actuals,
                non_negative=meta_model_cfg.non_negative if meta_model_cfg else True,
            )
            if meta_model_weights:
                # Evaluate meta-model on selection folds
                from .meta_model import predict_meta_model as pmm
                mm_scores = []
                for fold_idx in range(len(selection_origins)):
                    fold_map = {}
                    for name, weights_val in meta_model_weights.items():
                        if name in mm_fold_forecasts and fold_idx < len(mm_fold_forecasts[name]):
                            f = mm_fold_forecasts[name][fold_idx]
                            if f:
                                fold_map[name] = f
                    if len(fold_map) >= 2:
                        try:
                            combined = pmm(meta_model_weights, fold_map)
                            actual = values[selection_origins[fold_idx] : selection_origins[fold_idx] + horizon]
                            mm_scores.append(error_score(actual, combined))
                        except Exception:
                            mm_scores.append(None)
                    else:
                        mm_scores.append(None)
                valid_mm = [x for x in mm_scores if x is not None]
                if valid_mm and len(valid_mm) == len(selection_origins):
                    meta_model_score = mean(valid_mm)
                    if "meta_model" not in all_model_names:
                        all_model_names.append("meta_model")

    baseline_scores = {name: score for name, score in scores.items() if name in BASELINES and score is not None}
    if not baseline_scores:
        return Evaluation(
            None, None, scores, empty_scores.copy(), None, [], None,
            ["No baseline completed every selection fold."], False,
            tsfm_scores=tsfm_scores,
        )
    strongest_baseline = min(baseline_scores, key=baseline_scores.get)  # type: ignore[arg-type]
    selected = strongest_baseline
    baseline_score = baseline_scores[strongest_baseline]

    # Consider all non-baseline candidates (built-in + TSFM + ensemble + meta-model)
    candidate_scores: dict[str, float] = {}
    for name, score in scores.items():
        if name not in BASELINES and score is not None:
            candidate_scores[name] = score
    for name, score in tsfm_scores.items():
        if score is not None:
            candidate_scores[name] = score
    if ensemble_score is not None:
        candidate_scores["ensemble"] = ensemble_score
    if meta_model_score is not None:
        candidate_scores["meta_model"] = meta_model_score

    if candidate_scores:
        candidate = min(candidate_scores, key=candidate_scores.get)  # type: ignore[arg-type]
        candidate_score = candidate_scores[candidate]
        if baseline_score > 0 and candidate_score <= baseline_score * (1 - minimum_improvement):
            selected = candidate

    # --- Calibration ---
    all_scores = {**scores, **tsfm_scores}
    if ensemble_score is not None:
        all_scores["ensemble"] = ensemble_score
    if meta_model_score is not None:
        all_scores["meta_model"] = meta_model_score
    selected_score = all_scores.get(selected, baseline_score)
    improvement = 0.0 if selected in BASELINES else (baseline_score - selected_score) / baseline_score if baseline_score > 0 else 0.0  # type: ignore[operator]

    # Get calibration prediction from the selected model
    if selected in MODELS:
        calibration_prediction = predict(selected, values[:calibration_origin], horizon, season)
    else:
        # TSFM selected — find the adapter and predict
        adapter = next((a for a in tsfm_adapters if a.name == selected), None)
        if adapter is None:
            # Fallback to strongest baseline if adapter disappeared
            selected = strongest_baseline
            calibration_prediction = predict(selected, values[:calibration_origin], horizon, season)
        else:
            try:
                calibration_prediction = adapter.predict(values[:calibration_origin], horizon, season)
            except (TSFMError, TSFMUnavailable, Exception) as exc:
                logger.warning("TSFM %s failed during calibration, falling back to %s: %s", selected, strongest_baseline, exc)
                selected = strongest_baseline
                calibration_prediction = predict(selected, values[:calibration_origin], horizon, season)

    calibration_actual = values[calibration_origin : calibration_origin + horizon]
    residuals = [actual - predicted for actual, predicted in zip(calibration_actual, calibration_prediction)]
    low, high = quantile(residuals, 0.1), quantile(residuals, 0.9)

    # --- Test ---
    test_actual = values[test_origin : test_origin + horizon]
    test_scores: dict[str, float | None] = {name: None for name in all_model_names}
    for name in {selected, strongest_baseline}:
        if name in MODELS:
            try:
                test_scores[name] = error_score(test_actual, predict(name, values[:test_origin], horizon, season))
            except ValueError:
                pass
        else:
            adapter = next((a for a in tsfm_adapters if a.name == selected), None)
            if adapter:
                try:
                    test_prediction_tsfm = adapter.predict(values[:test_origin], horizon, season)
                    test_scores[name] = error_score(test_actual, test_prediction_tsfm)
                except (TSFMError, TSFMUnavailable, Exception) as exc:
                    logger.warning("TSFM %s failed during test fold", name, exc_info=True)

    # Get test prediction for coverage assessment
    if selected in MODELS:
        test_prediction = predict(selected, values[:test_origin], horizon, season)
    else:
        adapter = next((a for a in tsfm_adapters if a.name == selected), None)
        if adapter:
            try:
                test_prediction = adapter.predict(values[:test_origin], horizon, season)
            except (TSFMError, TSFMUnavailable, Exception):
                test_prediction = predict(strongest_baseline, values[:test_origin], horizon, season)
        else:
            test_prediction = predict(strongest_baseline, values[:test_origin], horizon, season)

    coverage = mean(
        1.0 if prediction + low <= actual <= prediction + high else 0.0
        for actual, prediction in zip(test_actual, test_prediction)
    )
    warnings: list[str] = []
    if coverage < 0.7:
        warnings.append(f"Final-test 80% interval coverage was {coverage:.1%}, below 70%.")
    return Evaluation(selected, strongest_baseline, scores, test_scores, improvement,
                      residuals, coverage, warnings, True, tsfm_scores=tsfm_scores)

