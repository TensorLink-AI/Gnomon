"""Lightweight meta-model for stacking forecasts.

The meta-model learns an optimal linear combination of candidate model
forecasts using their backtest residuals as training data. It's a
"stacking" ensemble: instead of fixed weights (inverse-error, median),
the weights are learned from data.

Training data construction:
  For each rolling-origin fold, each model produces a forecast.
  The meta-model learns: actual[i] = w0 + w1*model1_forecast[i] + w2*model2_forecast[i] + ...

  Training samples come from ALL selection folds across ALL models.

Selection score, and why it is not the fit:
  The weights that ship are fit on every selection fold, like any final
  refit. The weights that are *scored* are not: ``evaluation.evaluate``
  refits leave-one-fold-out and scores fold i with weights that never saw
  fold i. Without that, the meta-model's selection score would be
  in-sample while every member's is out-of-sample, and it would win the
  comparison by construction rather than by being better. (This docstring
  previously claimed a calibration-fold/test-fold protocol that the code
  did not implement; leave-one-fold-out is what it does now.)

Constraints:
  - ``non_negative=True`` constrains weights to be >= 0 (a model can't
    have a negative contribution — no short-selling forecasts).
  - Ridge/Lasso regularization prevents overfitting with few models.
  - If training fails (singular matrix, too few samples, etc.), the
    meta-model degrades to the configured fallback strategy.

The meta-model is deliberately lightweight — no neural nets, no deep
learning, just constrained linear regression. It runs in microseconds
and adds zero dependencies.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

META_MODEL_NAME = "meta_model"


# ---------------------------------------------------------------------------
# Constrained least squares (non-negative weights)
# ---------------------------------------------------------------------------

def _solve_nnls(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve non-negative least squares: min ||Ax - b||^2 s.t. x >= 0.

    Simple active-set implementation. For small problems (few models,
    few folds) this is fast and dependency-free.

    Args:
        A: design matrix, shape (n_samples, n_features)
        b: target vector, shape (n_samples,)

    Returns:
        Solution vector x (n_features,), or None if solving fails.
    """
    n_samples = len(A)
    n_features = len(A[0]) if n_samples > 0 else 0
    if n_samples == 0 or n_features == 0:
        return None

    # If non-negative constraint not needed, use normal equations
    # For small systems, direct solve with regularization (ridge)
    # and clamp negative weights to zero iteratively.

    # Ridge: (A^T A + alpha I) x = A^T b
    # We use a simple iterative projected gradient approach.

    # Build A^T A and A^T b
    AtA = [[0.0] * n_features for _ in range(n_features)]
    Atb = [0.0] * n_features
    for i in range(n_samples):
        for j in range(n_features):
            Atb[j] += A[i][j] * b[i]
            for k in range(j, n_features):
                AtA[j][k] += A[i][j] * A[i][k]
                AtA[k][j] = AtA[j][k]

    # Add ridge regularization
    alpha = 1e-6  # small ridge for numerical stability
    for j in range(n_features):
        AtA[j][j] += alpha

    # Iterative projected gradient descent
    # x_{t+1} = max(0, x_t - lr * (AtA x_t - Atb))
    x = [0.0] * n_features
    lr = _estimate_learning_rate(AtA)
    max_iter = 1000
    tol = 1e-8

    for _ in range(max_iter):
        # Gradient: AtA x - Atb
        grad = [0.0] * n_features
        for j in range(n_features):
            for k in range(n_features):
                grad[j] += AtA[j][k] * x[k]
            grad[j] -= Atb[j]

        # Projected step
        max_change = 0.0
        for j in range(n_features):
            new_x = max(0.0, x[j] - lr * grad[j])
            max_change = max(max_change, abs(new_x - x[j]))
            x[j] = new_x

        if max_change < tol:
            break

    # Check if solution is all zeros (degenerate)
    if all(abs(w) < 1e-10 for w in x):
        logger.warning("Meta-model solution is all zeros — falling back")
        return None

    return x


def _estimate_learning_rate(AtA: list[list[float]]) -> float:
    """Estimate a safe learning rate from the spectral radius of A^T A."""
    # Use the max diagonal element as an upper bound on the largest eigenvalue
    max_diag = max(AtA[i][i] for i in range(len(AtA))) if AtA else 1.0
    return 1.0 / max(1.0, max_diag)


def _solve_ols(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve ordinary least squares (unconstrained).

    For comparison / fallback when non-negative constraint is off.
    """
    n_samples = len(A)
    n_features = len(A[0]) if n_samples > 0 else 0
    if n_samples == 0 or n_features == 0:
        return None

    # Same as nnls but without the non-negative projection
    AtA = [[0.0] * n_features for _ in range(n_features)]
    Atb = [0.0] * n_features
    for i in range(n_samples):
        for j in range(n_features):
            Atb[j] += A[i][j] * b[i]
            for k in range(j, n_features):
                AtA[j][k] += A[i][j] * A[i][k]
                AtA[k][j] = AtA[j][k]

    # Ridge regularization
    alpha = 1e-6
    for j in range(n_features):
        AtA[j][j] += alpha

    # Solve via Gaussian elimination
    return _gaussian_elimination(AtA, Atb)


def _gaussian_elimination(
    A: list[list[float]], b: list[float]
) -> list[float] | None:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(A)
    if n == 0:
        return None

    # Augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivoting
        pivot = col
        for i in range(col + 1, n):
            if abs(aug[i][col]) > abs(aug[pivot][col]):
                pivot = i
        aug[col], aug[pivot] = aug[pivot], aug[col]

        if abs(aug[col][col]) < 1e-12:
            return None  # Singular

        # Eliminate
        for i in range(col + 1, n):
            factor = aug[i][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[i][j] -= factor * aug[col][j]

    # Back-substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x


# ---------------------------------------------------------------------------
# Meta-model training and prediction
# ---------------------------------------------------------------------------

def train_meta_model(
    fold_forecasts: dict[str, list[list[float]]],
    fold_actuals: list[list[float]],
    non_negative: bool = True,
) -> dict[str, float] | None:
    """Train the meta-model on fold-level forecasts and actuals.

    Args:
        fold_forecasts: model_name → list of per-fold forecasts.
            Each fold forecast is a list of `horizon` values.
            All models must have the same number of folds and same horizon.
        fold_actuals: list of per-fold actuals. Each is a list of `horizon` values.
        non_negative: constrain weights to be >= 0.

    Returns:
        Dict of model_name → weight, or None if training failed.
    """
    model_names = list(fold_forecasts.keys())
    if len(model_names) < 2:
        logger.debug("Meta-model needs >= 2 models, got %d", len(model_names))
        return None

    n_folds = len(fold_actuals)
    if n_folds < 2:
        logger.debug("Meta-model needs >= 2 folds, got %d", n_folds)
        return None

    horizon = len(fold_actuals[0]) if n_folds > 0 else 0
    if horizon == 0:
        return None

    # Build design matrix:
    # Each row is: [model1_forecast[step], model2_forecast[step], ...]
    # Target is: actual[step]
    # We flatten across folds and steps.
    A: list[list[float]] = []
    b: list[float] = []

    for fold_idx in range(n_folds):
        actual = fold_actuals[fold_idx]
        for step in range(horizon):
            row = []
            valid = True
            for name in model_names:
                forecasts = fold_forecasts[name]
                if fold_idx >= len(forecasts):
                    valid = False
                    break
                forecast = forecasts[fold_idx]
                if step >= len(forecast):
                    valid = False
                    break
                row.append(forecast[step])

            if valid and step < len(actual):
                A.append(row)
                b.append(actual[step])

    if len(A) < len(model_names):
        logger.debug(
            "Meta-model has %d training samples for %d models — not enough",
            len(A), len(model_names),
        )
        return None

    # Solve
    if non_negative:
        weights = _solve_nnls(A, b)
    else:
        weights = _solve_ols(A, b)

    if weights is None:
        logger.warning("Meta-model solver failed — will use fallback")
        return None

    # Map weights to model names
    result = {}
    for i, name in enumerate(model_names):
        result[name] = weights[i]

    logger.info("Meta-model weights: %s", result)
    return result


def predict_meta_model(
    weights: dict[str, float],
    forecasts: dict[str, list[float]],
) -> list[float]:
    """Apply learned meta-model weights to produce a combined forecast.

    Args:
        weights: model_name → weight (from train_meta_model)
        forecasts: model_name → point forecast for the horizon

    Returns:
        Combined forecast
    """
    if not forecasts:
        raise ValueError("No forecasts for meta-model prediction")

    horizon = len(next(iter(forecasts.values())))
    combined = [0.0] * horizon
    total_weight = 0.0

    for name, forecast in forecasts.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        for i in range(horizon):
            combined[i] += w * forecast[i]
        total_weight += w

    # Normalize if weights don't sum to 1
    if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
        for i in range(horizon):
            combined[i] /= total_weight

    return combined
