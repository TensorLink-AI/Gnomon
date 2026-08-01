"""Deterministic scoring primitives and aggregate statistics.

Pure Python on purpose. The scorers run on every graded row of every run; a
numpy dependency here would make the harness harder to install than the thing
it benchmarks, and none of these are hot enough to need it.

Aggregation includes a paired significance test. Benchmark tables routinely
report a two-point accuracy difference on 150 items as though it meant
something; ``mcnemar`` gives the exact p-value for the paired design this
harness actually runs, so the reports can say when a difference is noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

EPSILON = 1e-9


# -- error metrics ---------------------------------------------------------

def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    _check_aligned(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    _check_aligned(actual, predicted)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error, as a fraction (0.05 == 5%).

    Points where the actual is zero are skipped rather than divided by an
    epsilon: an epsilon turns one zero into an arbitrarily large error and
    silently decides the comparison. If every point is zero the result is
    undefined and reported as such.
    """
    _check_aligned(actual, predicted)
    terms = [abs(a - p) / abs(a) for a, p in zip(actual, predicted) if abs(a) > EPSILON]
    if not terms:
        return float("nan")
    return sum(terms) / len(terms)


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric MAPE as a fraction in [0, 1]. Defined when actuals hit zero."""
    _check_aligned(actual, predicted)
    terms = []
    for a, p in zip(actual, predicted):
        denominator = (abs(a) + abs(p)) / 2
        if denominator > EPSILON:
            terms.append(abs(a - p) / denominator)
    if not terms:
        return 0.0
    # The raw mean lies in [0, 2]; halving puts it on the same 0-1 scale as
    # MAPE so a single pass threshold can serve both.
    return sum(terms) / len(terms) / 2


def mase(actual: Sequence[float], predicted: Sequence[float], history: Sequence[float], season: int = 1) -> float:
    """MAE scaled by the in-sample seasonal-naive MAE."""
    _check_aligned(actual, predicted)
    if len(history) <= season:
        return float("nan")
    scale_terms = [abs(history[i] - history[i - season]) for i in range(season, len(history))]
    scale = sum(scale_terms) / len(scale_terms)
    if scale <= EPSILON:
        return float("nan")
    return mae(actual, predicted) / scale


def _check_aligned(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if not actual or not predicted:
        raise ValueError("Both series must be non-empty.")
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: expected {len(actual)}, got {len(predicted)}.")


# -- classification --------------------------------------------------------

@dataclass(frozen=True)
class ClassificationScore:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def binary_f1(actual: Sequence[int], predicted: Sequence[int]) -> ClassificationScore:
    """F1 for a positive class, with the confusion counts kept.

    An empty prediction set scores 0 rather than raising: "flagged nothing" is
    a real answer to an anomaly-detection question and should be graded, not
    excluded from the denominator.
    """
    _check_aligned(actual, predicted)
    tp = sum(1 for a, p in zip(actual, predicted) if a and p)
    fp = sum(1 for a, p in zip(actual, predicted) if not a and p)
    fn = sum(1 for a, p in zip(actual, predicted) if a and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ClassificationScore(precision, recall, f1, tp, fp, fn)


def index_f1(actual_indices: Sequence[int], predicted_indices: Sequence[int], length: int) -> ClassificationScore:
    """F1 over sets of flagged positions rather than dense label vectors."""
    actual = [0] * length
    predicted = [0] * length
    for index in actual_indices:
        if 0 <= index < length:
            actual[index] = 1
    for index in predicted_indices:
        if 0 <= index < length:
            predicted[index] = 1
    return binary_f1(actual, predicted)


# -- aggregate statistics --------------------------------------------------

def accuracy(flags: Sequence[bool]) -> float:
    return sum(1 for flag in flags if flag) / len(flags) if flags else 0.0


def bootstrap_ci(
    flags: Sequence[bool],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260801,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a rate. Seeded, so runs replay."""
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    values = [1.0 if flag else 0.0 for flag in flags]
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    tail = (1 - confidence) / 2
    low = means[max(0, int(tail * iterations) - 1)]
    high = means[min(iterations - 1, int((1 - tail) * iterations))]
    return (round(low, 6), round(high, 6))


@dataclass(frozen=True)
class PairedTest:
    """Result of an exact McNemar test on paired correct/incorrect flags."""

    n_pairs: int
    treatment_only: int      # treatment right, control wrong
    control_only: int        # control right, treatment wrong
    both: int
    neither: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.treatment_only + self.control_only


def mcnemar(control: Sequence[bool], treatment: Sequence[bool]) -> PairedTest:
    """Exact two-sided McNemar test over paired outcomes.

    The pairing is the whole point of running both conditions on identical
    task ids: it removes item difficulty from the comparison, so the test only
    has to explain the items where the two conditions disagreed.
    """
    if len(control) != len(treatment):
        raise ValueError("Paired test needs equal-length control and treatment vectors.")
    both = sum(1 for c, t in zip(control, treatment) if c and t)
    neither = sum(1 for c, t in zip(control, treatment) if not c and not t)
    treatment_only = sum(1 for c, t in zip(control, treatment) if not c and t)
    control_only = sum(1 for c, t in zip(control, treatment) if c and not t)

    n = treatment_only + control_only
    if n == 0:
        p_value = 1.0
    else:
        smaller = min(treatment_only, control_only)
        tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)
        p_value = min(1.0, 2 * tail)
    return PairedTest(
        n_pairs=len(control), treatment_only=treatment_only, control_only=control_only,
        both=both, neither=neither, p_value=round(p_value, 8),
    )
