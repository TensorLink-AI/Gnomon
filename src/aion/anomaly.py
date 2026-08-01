"""Evaluated anomaly detection: competing detectors, graded selection.

Candidate detectors — a robust z-score, a rolling-median residual score,
and a one-step-ahead forecast-residual score — compete on a synthetic
anomaly-injection grader before any of them is allowed to label the real
series. The grader plants spikes, level shifts, and dropouts of a
standard, noise-scaled magnitude into copies of the observed series and
scores each detector's precision, recall, and F1 at recovering them; the
winner (ties broken toward the simpler detector) produces the reported
anomalies, and every candidate's grade ships in the output.

Injection placement uses a PRNG seeded from the series content, so the
whole procedure is deterministic: same data in, same anomalies out. When
labelled anomalies are supplied, selection uses the labels instead and
the synthetic grade is reported alongside.

Like every operator, this one abstains honestly: too little history is
``inconclusive``, and a best grader F1 below ``GRADER_FLOOR`` downgrades
the result to ``conditionally_supported`` — if no detector can reliably
find planted anomalies in this series' noise, real detections inherit
that doubt.
"""

from __future__ import annotations

import random
import zlib
from statistics import median
from typing import Any, Callable

from .contracts import SupportAssessment, SupportReason
from .operators import _robust_scale, inconclusive

MIN_DETECTION_HISTORY = 16
DEFAULT_THRESHOLD = 3.5
GRADER_FLOOR = 0.5
SPIKE_TRIALS = 3
SHIFT_TRIALS = 2
DROPOUT_TRIALS = 2
SPIKE_SCALE = 6.0
SHIFT_SCALE = 4.0
DROPOUT_SCALE = 5.0
DROPOUT_RUN = 3
SHIFT_HIT_WINDOW = 3

ScoreFunction = Callable[[list[float], int], list[float]]


# ---------------------------------------------------------------------------
# Candidate detectors — each returns standardised scores aligned to values
# ---------------------------------------------------------------------------

def _warmup(values: list[float], season: int) -> int:
    return min(max(8, season + 1), max(1, len(values) // 2))


def robust_zscore_scores(values: list[float], season: int) -> list[float]:
    """Median/MAD z-scores, per-phase adjusted when a season is usable."""
    if season > 1 and len(values) >= 2 * season:
        adjusted = []
        for index, value in enumerate(values):
            phase = [values[j] for j in range(index % season, len(values), season)]
            adjusted.append(value - median(phase))
    else:
        centre = median(values)
        adjusted = [value - centre for value in values]
    scale = _robust_scale(adjusted)
    return [value / scale for value in adjusted]


def rolling_median_scores(values: list[float], season: int) -> list[float]:
    """Residuals against a centred rolling median: local level tracking
    keeps trends and slow drifts from masquerading as anomalies."""
    window = season if season >= 5 else 5
    if window % 2 == 0:
        window += 1
    half = window // 2
    residuals = [
        value - median(values[max(0, index - half): index + half + 1])
        for index, value in enumerate(values)
    ]
    scale = _robust_scale(residuals)
    return [value / scale for value in residuals]


def forecast_interval_scores(values: list[float], season: int) -> list[float]:
    """One-step-ahead forecast residuals, standardised — the interval-
    exceedance route. The forecaster is the better of seasonal_naive and
    theta by one-step MAE over the scored region; warm-up points score 0."""
    from .models import predict
    warmup = _warmup(values, season)
    if warmup >= len(values):
        return [0.0] * len(values)
    best_residuals: list[float] | None = None
    best_error: float | None = None
    for name in ("seasonal_naive", "theta"):
        residuals = []
        try:
            for index in range(warmup, len(values)):
                prediction = predict(name, values[:index], 1, season)[0]
                residuals.append(values[index] - prediction)
        except (ValueError, ArithmeticError):
            continue
        error = sum(abs(value) for value in residuals) / len(residuals)
        if best_error is None or error < best_error:
            best_error, best_residuals = error, residuals
    if best_residuals is None:
        return [0.0] * len(values)
    scale = _robust_scale(best_residuals)
    return [0.0] * warmup + [value / scale for value in best_residuals]


#: Selection order doubles as the tie-break: simpler detectors first.
DETECTORS: dict[str, ScoreFunction] = {
    "robust_zscore": robust_zscore_scores,
    "rolling_median_residual": rolling_median_scores,
    "forecast_interval": forecast_interval_scores,
}


# ---------------------------------------------------------------------------
# The synthetic-injection grader
# ---------------------------------------------------------------------------

def _content_seed(values: list[float]) -> int:
    canonical = ",".join(f"{value:.9g}" for value in values)
    return zlib.crc32(canonical.encode("utf-8"))


def _flagged(scores: list[float], threshold: float) -> set[int]:
    return {index for index, score in enumerate(scores) if abs(score) >= threshold}


def _injection_trials(
    values: list[float], season: int, rng: random.Random,
) -> list[dict[str, Any]]:
    """Deterministic injection plan: (family, indices, deltas) triples."""
    n = len(values)
    scale = _robust_scale([value - median(values) for value in values])
    warmup = _warmup(values, season)
    shift_length = max(4, min(8, n // 8))
    trials: list[dict[str, Any]] = []
    for _ in range(SPIKE_TRIALS):
        index = rng.randrange(warmup, n - 1)
        sign = rng.choice((-1.0, 1.0))
        trials.append({"family": "spike", "indices": [index],
                       "deltas": [sign * SPIKE_SCALE * scale],
                       "hit_indices": {index - 1, index, index + 1}})
    for _ in range(SHIFT_TRIALS):
        if warmup >= n - shift_length - 1:
            break
        onset = rng.randrange(warmup, n - shift_length)
        sign = rng.choice((-1.0, 1.0))
        indices = list(range(onset, n))
        trials.append({"family": "level_shift", "indices": indices,
                       "deltas": [sign * SHIFT_SCALE * scale] * len(indices),
                       "hit_indices": set(range(onset, min(n, onset + SHIFT_HIT_WINDOW)))})
    for _ in range(DROPOUT_TRIALS):
        if warmup >= n - DROPOUT_RUN - 1:
            break
        onset = rng.randrange(warmup, n - DROPOUT_RUN)
        indices = list(range(onset, onset + DROPOUT_RUN))
        trials.append({"family": "dropout", "indices": indices,
                       "deltas": [-DROPOUT_SCALE * scale] * DROPOUT_RUN,
                       "hit_indices": set(range(onset - 1, onset + DROPOUT_RUN + 1))})
    return trials


def grade_detectors(
    values: list[float], season: int,
    *, threshold: float = DEFAULT_THRESHOLD,
    extra_detectors: dict[str, ScoreFunction] | None = None,
) -> dict[str, Any]:
    """Grade every candidate detector on synthetic injections.

    Flags a detector already raises on the clean series are excluded from
    both hits and false positives — the grader measures recovery of the
    *planted* anomalies only, so pre-existing oddities in the data cannot
    reward or punish a candidate."""
    detectors = {**DETECTORS, **(extra_detectors or {})}
    rng = random.Random(_content_seed(values))
    trials = _injection_trials(values, season, rng)
    grades: dict[str, dict[str, Any]] = {}
    for name, score_function in detectors.items():
        clean_flags = _flagged(score_function(values, season), threshold)
        hits, false_positives = 0, 0
        for trial in trials:
            injected = list(values)
            for index, delta in zip(trial["indices"], trial["deltas"]):
                injected[index] += delta
            novel = _flagged(score_function(injected, season), threshold) - clean_flags
            if novel & trial["hit_indices"]:
                hits += 1
            false_positives += len(novel - trial["hit_indices"] - set(trial["indices"]))
        recall = hits / len(trials) if trials else 0.0
        precision = hits / (hits + false_positives) if hits + false_positives else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        grades[name] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "trials": len(trials),
            "false_positives": false_positives,
        }
    return {
        "grades": grades,
        "injection": {
            "seed": _content_seed(values),
            "families": {
                "spike": SPIKE_TRIALS, "level_shift": SHIFT_TRIALS,
                "dropout": DROPOUT_TRIALS,
            },
            "magnitudes_in_robust_scale": {
                "spike": SPIKE_SCALE, "level_shift": SHIFT_SCALE,
                "dropout": DROPOUT_SCALE,
            },
        },
    }


def _grade_against_labels(
    values: list[float], season: int, label_indices: list[int],
    *, threshold: float,
    extra_detectors: dict[str, ScoreFunction] | None = None,
) -> dict[str, dict[str, Any]]:
    """Grade detectors against user-supplied labels (±1 index tolerance)."""
    detectors = {**DETECTORS, **(extra_detectors or {})}
    tolerant = set()
    for index in label_indices:
        tolerant.update((index - 1, index, index + 1))
    grades = {}
    for name, score_function in detectors.items():
        flags = _flagged(score_function(values, season), threshold)
        hits = sum(1 for index in label_indices
                   if flags & {index - 1, index, index + 1})
        false_positives = len(flags - tolerant)
        recall = hits / len(label_indices) if label_indices else 0.0
        precision = hits / (hits + false_positives) if hits + false_positives else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        grades[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                        "f1": round(f1, 4), "labels": len(label_indices)}
    return grades


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------

def detect_anomalies(
    timestamps: list[Any], values: list[float],
    *, season: int = 1, threshold: float = DEFAULT_THRESHOLD,
    label_indices: list[int] | None = None,
    extra_detectors: dict[str, ScoreFunction] | None = None,
) -> dict[str, Any]:
    """Graded anomaly detection over one series.

    Selection is by synthetic-injection F1 — or by label F1 when labels
    are supplied — with ties broken toward the earlier (simpler) detector.
    """
    if len(values) < MIN_DETECTION_HISTORY:
        return {"anomalies": [], "detector": None, "detector_grades": {},
                "support": inconclusive(
                    "insufficient_history",
                    f"Anomaly detection needs at least {MIN_DETECTION_HISTORY} "
                    f"observations (have {len(values)}).",
                    "Supply a longer window of history.",
                ).to_dict()}
    grading = grade_detectors(values, season, threshold=threshold,
                              extra_detectors=extra_detectors)
    grades = grading["grades"]
    label_grades: dict[str, dict[str, Any]] | None = None
    if label_indices:
        label_grades = _grade_against_labels(
            values, season, label_indices, threshold=threshold,
            extra_detectors=extra_detectors,
        )
        ranking = label_grades
        selection_basis = "label_f1"
    else:
        ranking = grades
        selection_basis = "synthetic_injection_f1"
    order = list({**DETECTORS, **(extra_detectors or {})})
    selected = max(order, key=lambda name: (ranking[name]["f1"], -order.index(name)))
    scores = {**DETECTORS, **(extra_detectors or {})}[selected](values, season)
    anomalies = [
        {"timestamp": timestamps[index], "value": values[index],
         "score": round(score, 4)}
        for index, score in enumerate(scores) if abs(score) >= threshold
    ]
    best_f1 = ranking[selected]["f1"]
    if best_f1 < GRADER_FLOOR:
        support = SupportAssessment(
            "conditionally_supported",
            [SupportReason(
                "low_grader_confidence",
                f"The best detector ({selected}) recovered planted anomalies "
                f"with F1 {best_f1:.2f} (< {GRADER_FLOOR}); real detections "
                "in this series' noise carry the same doubt.",
            )],
            sensitivity={"threshold": threshold, "selection_basis": selection_basis},
        )
    else:
        support = SupportAssessment(
            "supported",
            [] if anomalies else [SupportReason(
                "no_anomalies_detected",
                "No point exceeded the detection threshold; absence of "
                "anomalies is a conclusion, not a failure.",
            )],
            sensitivity={"threshold": threshold, "selection_basis": selection_basis,
                         "selected_f1": best_f1},
        )
    result = {
        "detector": selected,
        "selection_basis": selection_basis,
        "detector_grades": grades,
        "injection": grading["injection"],
        "anomalies": anomalies,
        "scores": [round(score, 4) for score in scores],
        "support": support.to_dict(),
    }
    if label_grades is not None:
        result["label_grades"] = label_grades
    return result
