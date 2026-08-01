from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable

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
    degraded: bool = False
    tsfm_scores: dict[str, float | None] = field(default_factory=dict)
    # Informational only: notes never downgrade support, unlike warnings.
    notes: list[str] = field(default_factory=list)


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


def select_model_lightweight(
    values: list[float], horizon: int, season: int,
    train_at: Callable[[int], list[float]] | None = None,
) -> Evaluation:
    """Select on one trailing holdout when separated rolling folds do not fit."""
    if train_at is None:
        train_at = lambda origin: values[:origin]  # noqa: E731
    if len(values) < horizon + 2:
        scores = {name: None for name in MODELS}
        return Evaluation(None, None, scores, scores.copy(), None, [], None,
                          [f"Need at least {horizon + 2} observations (have {len(values)}) for degraded forecasting."], False, True)
    holdout = min(horizon, max(1, len(values) // 4))
    origin = len(values) - holdout
    scores: dict[str, float | None] = {name: None for name in MODELS}
    forecasts: dict[str, list[float]] = {}
    actual = values[origin:]
    train = train_at(origin)
    for name in MODELS:
        try:
            prediction = predict(name, train, holdout, season)
            scores[name] = error_score(actual, prediction)
            forecasts[name] = prediction
        except (ValueError, ArithmeticError):
            continue
    valid = {name: score for name, score in scores.items() if score is not None}
    if not valid:
        return Evaluation(None, None, scores, scores.copy(), None, [], None,
                          ["Series is too short for lightweight model selection."], False, True)
    selected = min(valid, key=valid.get)  # type: ignore[arg-type]
    baselines = {name: score for name, score in valid.items() if name in BASELINES}
    strongest = min(baselines, key=baselines.get) if baselines else selected  # type: ignore[arg-type]
    residuals = [a - p for a, p in zip(actual, forecasts[selected])]
    return Evaluation(selected, strongest, scores, {name: None for name in MODELS}, None,
                      residuals, None, [
                          f"Degraded forecast: model selection used a single trailing {holdout}-observation holdout; rolling-origin calibration and final testing were unavailable. At least {max(2 * season, 2 * horizon, 8) + 2 * horizon} observations (have {len(values)}) are needed for separated selection and calibration."
                      ], True, True)


def evaluate(
    values: list[float],
    horizon: int,
    season: int,
    minimum_improvement: float,
    *,
    frequency: str = "h",
    tsfm_names: list[str] | None = None,
    config: Any = None,
    strict_abstention: bool = False,
    train_at: Callable[[int], list[float]] | None = None,
) -> Evaluation:
    """``train_at(origin)`` returns the training history for a fold whose
    forecast origin is index ``origin`` — by default a plain prefix slice,
    but a snapshot-backed provider returns the series *as known at* the
    fold cutoff, which is what makes backtests vintage-honest."""
    if train_at is None:
        train_at = lambda origin: values[:origin]  # noqa: E731
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = _origins(len(values), horizon, minimum_train)
    empty_scores = {name: None for name in MODELS}
    if len(origins) < 2:
        if not strict_abstention:
            return select_model_lightweight(values, horizon, season, train_at)
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

    # --- Resolve config ---
    ensemble_enabled = getattr(config, "ensemble", None) and config.ensemble.enabled if config else False
    ensemble_strategy = config.ensemble.strategy if config and ensemble_enabled else "weighted_mean"
    ensemble_cfg = config.ensemble if config else None
    meta_model_enabled = getattr(config, "meta_model", None) and config.meta_model.enabled if config else False
    meta_model_cfg = config.meta_model if config and meta_model_enabled else None

    # --- Load TSFM candidates (lazy, graceful) ---
    # Prefer sandboxed adapters (isolated venvs) to avoid dependency conflicts.
    # Fall back to in-process adapters if no sandboxes are set up.
    from .tsfm import eligible_tsfms
    from .tsfm_sandbox import sandbox_tsfm_candidates, sandbox_available_tsfms
    tsfm_adapters: list[Any] = []
    eligible_names, capability_exclusions = eligible_tsfms(
        history_length=len(values), horizon=horizon, frequency=frequency,
    )
    requested_names = tsfm_names if tsfm_names is not None else eligible_names
    requested_names = [name for name in requested_names if name in eligible_names]
    for name, reasons in capability_exclusions.items():
        if tsfm_names is None or name in tsfm_names:
            warnings.append(f"Skipped TSFM {name}: {'; '.join(reasons)}.")
    sandbox_names = sandbox_available_tsfms()
    if sandbox_names and requested_names:
        tsfm_adapters = sandbox_tsfm_candidates(
            requested=requested_names,
            frequency=frequency,
        )
    elif requested_names:
        tsfm_adapters = tsfm_candidates(requested=requested_names, frequency=frequency)
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

    # Disclose the model tier that could not compete. A fresh install has no
    # TSFM sandboxes, so without this note the operator most likely to benefit
    # from a stronger candidate never learns one was eligible.
    from .tsfm import installed_tsfms
    notes: list[str] = []
    if requested_names and not sandbox_names and not installed_tsfms():
        notes.append(
            f"No foundation-model candidate competed: "
            f"{', '.join(requested_names)} "
            f"{'is' if len(requested_names) == 1 else 'are'} eligible for this "
            f"series but no sandbox is installed. Run "
            f"`aion tsfm install {requested_names[0]}` to add one; it enters "
            f"the same folds against the same baselines."
        )

    # --- Run built-in models on selection folds ---
    fold_scores: dict[str, list[float]] = {name: [] for name in MODELS}
    # Store per-fold forecasts for ensemble/meta-model training
    fold_forecasts: dict[str, list[list[float]]] = {name: [] for name in MODELS}
    for origin in selection_origins:
        actual = values[origin : origin + horizon]
        train = train_at(origin)
        for name in MODELS:
            try:
                forecast = predict(name, train, horizon, season)
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
            train = train_at(origin)
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
                        last_observed=(train_at(selection_origins[fold_idx]) or [0.0])[-1] if selection_origins else 0.0,
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
            all_model_names.append("ensemble")

    # --- Meta-model training ---
    meta_model_weights: dict[str, float] | None = None
    meta_model_score: float | None = None
    if meta_model_enabled:
        from .meta_model import train_meta_model
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
            degraded, tsfm_scores=tsfm_scores, notes=notes,
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

    def _predict_selected(name: str, train: list[float]) -> list[float]:
        """Dispatch a prediction to a built-in model or a TSFM adapter."""
        if name in MODELS:
            return predict(name, train, horizon, season)
        adapter = next((a for a in tsfm_adapters if a.name == name), None)
        if adapter is None:
            raise ValueError(f"no adapter available for {name}")
        return adapter.predict(train, horizon, season)

    # Get calibration prediction from the selected model; fall back to the
    # strongest baseline if a TSFM/ensemble selection cannot predict here.
    try:
        calibration_prediction = _predict_selected(selected, train_at(calibration_origin))
    except Exception as exc:
        if selected not in MODELS:
            logger.warning(
                "%s failed during calibration, falling back to %s: %s",
                selected, strongest_baseline, exc,
            )
        selected = strongest_baseline
        calibration_prediction = predict(selected, train_at(calibration_origin), horizon, season)

    # Pool residuals of the selected model across every selection fold plus
    # the calibration fold: one horizon of residuals is too small a sample
    # for stable quantiles. Folds where the selected model cannot predict
    # (possible for TSFM adapters) simply contribute nothing.
    residuals: list[float] = []
    for origin in selection_origins:
        try:
            prediction = _predict_selected(selected, train_at(origin))
        except Exception:
            continue
        actual = values[origin : origin + horizon]
        residuals.extend(a - p for a, p in zip(actual, prediction))
    calibration_actual = values[calibration_origin : calibration_origin + horizon]
    residuals.extend(a - p for a, p in zip(calibration_actual, calibration_prediction))
    residual_quantiles = {p: quantile(residuals, p) for p in (0.1, 0.5, 0.9)}

    # --- Test ---
    test_scores: dict[str, float | None] = {name: None for name in all_model_names}
    coverage: float | None = None
    if test_origin is not None:
        test_actual = values[test_origin : test_origin + horizon]
        for name in {selected, strongest_baseline}:
            try:
                test_scores[name] = error_score(
                    test_actual, _predict_selected(name, train_at(test_origin))
                )
            except Exception:
                logger.debug("model %s failed during test fold", name, exc_info=True)

        # Get test prediction for coverage assessment
        try:
            test_prediction = _predict_selected(selected, train_at(test_origin))
        except Exception:
            test_prediction = predict(strongest_baseline, train_at(test_origin), horizon, season)
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
                      residuals, coverage, warnings, True, degraded,
                      tsfm_scores=tsfm_scores, notes=notes)
