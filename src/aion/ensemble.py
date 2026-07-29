"""Ensemble engine: combine multiple model forecasts into one.

An ensemble is selected as the "model" when:
  - The ensemble config is enabled
  - At least ``min_models`` candidates produced valid forecasts
  - The ensemble's backtest error beats the strongest baseline by the
    configured margin

Supported strategies:
  - ``weighted_mean``: weight each model by inverse backtest error
  - ``median``: robust median of all model point forecasts
  - ``voting``: majority vote for direction/threshold decisions
  - ``stacking``: delegate to the meta-model (see meta_model.py)

The ensemble is a candidate in the evaluation pipeline — it competes
against individual models and baselines on identical rolling folds.
If it doesn't beat the baseline, the baseline wins.

Quantile combination strategies:
  - ``union``: widest interval (max of upper, min of lower)
  - ``intersection``: narrowest interval
  - ``weighted``: weighted average of quantiles
"""

from __future__ import annotations

import logging
import math
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point forecast combination
# ---------------------------------------------------------------------------

def weighted_mean_forecast(
    forecasts: dict[str, list[float]],
    scores: dict[str, float],
    max_weight_ratio: float = 0.7,
) -> list[float]:
    """Combine forecasts using inverse-error weighted mean.

    Models with lower backtest error get higher weight. Weights are
    normalised and capped so no single model dominates.

    Args:
        forecasts: model_name → list of point forecasts
        scores: model_name → backtest error (lower is better)
        max_weight_ratio: cap on any single model's weight

    Returns:
        Combined forecast as a list of floats
    """
    if not forecasts:
        raise ValueError("No forecasts to ensemble")

    horizon = len(next(iter(forecasts.values())))

    # Compute inverse-error weights
    weights: dict[str, float] = {}
    for name, score in scores.items():
        if name in forecasts and score is not None and score > 0:
            weights[name] = 1.0 / score

    if not weights:
        # All scores zero or None — equal weights
        for name in forecasts:
            weights[name] = 1.0

    # Normalise and cap
    total = sum(weights.values())
    for name in weights:
        weights[name] /= total

    # Cap any weight that exceeds max_weight_ratio
    # Redistribute excess to other models proportionally
    for _ in range(3):  # iterate to settle caps
        excess = 0.0
        for name in weights:
            if weights[name] > max_weight_ratio:
                excess += weights[name] - max_weight_ratio
                weights[name] = max_weight_ratio
        if excess <= 1e-9:
            break
        uncapped = [n for n in weights if weights[n] < max_weight_ratio]
        if uncapped:
            uncapped_total = sum(weights[n] for n in uncapped)
            for n in uncapped:
                weights[n] += excess * (weights[n] / uncapped_total) if uncapped_total > 0 else 0

    # Renormalise
    total = sum(weights.values())
    if total > 0:
        for name in weights:
            weights[name] /= total

    # Combine
    combined = [0.0] * horizon
    for name, forecast in forecasts.items():
        w = weights.get(name, 0.0)
        for i in range(horizon):
            combined[i] += w * forecast[i]

    return combined


def median_forecast(forecasts: dict[str, list[float]]) -> list[float]:
    """Combine forecasts using the median at each step.

    Robust to outliers — no weighting needed.
    """
    if not forecasts:
        raise ValueError("No forecasts to ensemble")

    horizon = len(next(iter(forecasts.values())))
    combined = []
    for step in range(horizon):
        values = [f[step] for f in forecasts.values() if step < len(f)]
        combined.append(median(values))
    return combined


def voting_forecast(
    forecasts: dict[str, list[float]],
    last_observed: float,
    threshold: float = 0.5,
) -> list[float]:
    """Combine forecasts using majority vote on direction.

    For each step, count how many models predict up vs down relative
    to the last observed value. The median of the winning side is used.
    """
    if not forecasts:
        raise ValueError("No forecasts to ensemble")

    horizon = len(next(iter(forecasts.values())))
    combined = []
    for step in range(horizon):
        up = [f[step] for f in forecasts.values() if step < len(f) and f[step] > last_observed]
        down = [f[step] for f in forecasts.values() if step < len(f) and f[step] <= last_observed]
        if len(up) >= len(down) * threshold:
            combined.append(median(up) if up else last_observed)
        elif len(down) >= len(up) * threshold:
            combined.append(median(down) if down else last_observed)
        else:
            # No consensus — use median of all
            all_vals = [f[step] for f in forecasts.values() if step < len(f)]
            combined.append(median(all_vals))
    return combined


# ---------------------------------------------------------------------------
# Quantile combination
# ---------------------------------------------------------------------------

def combine_quantiles(
    all_quantiles: list[list[dict[str, float]]],
    strategy: str = "union",
    weights: list[float] | None = None,
) -> list[dict[str, float]]:
    """Combine quantile forecasts from multiple models.

    Args:
        all_quantiles: list of per-model quantile forecasts, each is a list
            of dicts: [{"0.1": v, "0.5": v, "0.9": v}, ...]
        strategy: "union" (widest), "intersection" (narrowest), "weighted"
        weights: per-model weights for "weighted" strategy

    Returns:
        Combined quantile forecasts: [{"0.1": v, "0.5": v, "0.9": v}, ...]
    """
    if not all_quantiles:
        return []

    horizon = len(all_quantiles[0])
    quantile_keys = list(all_quantiles[0][0].keys())
    combined = []

    for step in range(horizon):
        row = {}
        for q in quantile_keys:
            values = []
            for model_idx, model_q in enumerate(all_quantiles):
                if step < len(model_q) and q in model_q[step]:
                    values.append((model_idx, model_q[step][q]))

            if not values:
                row[q] = 0.0
                continue

            if strategy == "union":
                # Widest interval: min of lower quantiles, max of upper
                q_float = float(q)
                if q_float <= 0.5:
                    row[q] = min(v for _, v in values)
                else:
                    row[q] = max(v for _, v in values)

            elif strategy == "intersection":
                # Narrowest: max of lower, min of upper
                q_float = float(q)
                if q_float <= 0.5:
                    row[q] = max(v for _, v in values)
                else:
                    row[q] = min(v for _, v in values)

            elif strategy == "weighted" and weights:
                total_w = sum(weights[i] for i, _ in values if i < len(weights))
                if total_w > 0:
                    row[q] = sum(v * weights[i] for i, v in values if i < len(weights)) / total_w
                else:
                    row[q] = sum(v for _, v in values) / len(values)

            else:
                # Default: simple mean
                row[q] = sum(v for _, v in values) / len(values)

        combined.append(row)
    return combined


# ---------------------------------------------------------------------------
# Ensemble evaluation — used as a candidate in the evaluation pipeline
# ---------------------------------------------------------------------------

ENSEMBLE_MODEL_NAME = "ensemble"


def compute_ensemble_forecast(
    forecasts: dict[str, list[float]],
    scores: dict[str, float | None],
    strategy: str = "weighted_mean",
    last_observed: float = 0.0,
    config: Any = None,
) -> list[float]:
    """Compute the ensemble point forecast given individual model forecasts.

    Args:
        forecasts: model_name → point forecasts
        scores: model_name → backtest error (lower is better)
        strategy: "weighted_mean" | "median" | "voting"
        last_observed: last observed value (for voting strategy)
        config: ensemble config (for strategy-specific params)

    Returns:
        Combined point forecast
    """
    # Filter to models with valid forecasts and scores
    valid = {
        name: f for name, f in forecasts.items()
        if f and len(f) > 0
    }
    if not valid:
        raise ValueError("No valid forecasts for ensemble")

    if strategy == "weighted_mean":
        valid_scores = {
            name: score for name, score in scores.items()
            if name in valid and score is not None
        }
        max_ratio = getattr(config, "max_weight_ratio", 0.7) if config else 0.7
        return weighted_mean_forecast(valid, valid_scores, max_ratio)

    elif strategy == "median":
        return median_forecast(valid)

    elif strategy == "voting":
        threshold = 0.5
        if config and hasattr(config, "voting") and isinstance(config.voting, dict):
            threshold = config.voting.get("threshold", 0.5)
        return voting_forecast(valid, last_observed, threshold)

    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")
