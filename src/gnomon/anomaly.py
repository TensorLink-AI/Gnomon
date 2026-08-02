"""Evaluated anomaly detection: competing detectors, graded selection.

Candidate detectors — a robust z-score, a rolling-median residual score,
a local-slope deviation score, and a one-step-ahead forecast-residual
score — compete on a synthetic anomaly-injection grader before any of
them is allowed to label the real series. The grader plants spikes,
level shifts, dropouts, and trend shifts of a standard, noise-scaled
magnitude into copies of the observed series and scores each detector's
precision, recall, and F1 at recovering them; the winner (ties broken
toward the simpler detector) produces the reported anomalies, and every
candidate's grade ships in the output.

The grade vouches for the families the grader planted and nothing more,
so the tested families travel with the verdict (``graded_families`` in
the support sensitivity, and an assumption naming them). A detector that
recovers spikes has said nothing about anomaly kinds nobody tested it
on — and a support status that did not say so would overstate what was
measured.

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
TREND_TRIALS = 2
SPIKE_SCALE = 6.0
SHIFT_SCALE = 4.0
DROPOUT_SCALE = 5.0
TREND_SCALE = 4.0
DROPOUT_RUN = 3
SHIFT_HIT_WINDOW = 3
SLOPE_MIN_WINDOW = 9
SLOPE_MAX_WINDOW = 101

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


def local_slope_scores(values: list[float], season: int) -> list[float]:
    """Deviation of the *local slope* from the series' typical slope.

    The other three detectors all ask whether a value is where it should
    be; a trend anomaly keeps every value in range and changes only how
    fast the series moves. Scoring the rolling median of first
    differences against the median slope flags the whole stretch that
    drifts at the wrong rate, not merely its onset — and taking medians
    keeps a lone spike (one large difference, immediately undone) from
    reading as a change in slope.
    """
    if len(values) < 3:
        return [0.0] * len(values)
    differences = [values[index + 1] - values[index]
                   for index in range(len(values) - 1)]
    # A slope is only visible once enough differences are averaged: the
    # noise of a windowed slope falls as 1/sqrt(window), so the window
    # scales with the series (and covers whole seasons when there is
    # one) instead of sitting at the fixed width a point detector wants.
    window = max(SLOPE_MIN_WINDOW,
                 min(len(values) // 20, SLOPE_MAX_WINDOW))
    if season > 1:
        window = max(window, 2 * season + 1)
    window = min(window, max(3, len(differences)))
    if window % 2 == 0:
        window += 1
    half = window // 2
    slopes = [
        median(differences[max(0, index - half): index + half + 1])
        for index in range(len(differences))
    ]
    centre = median(slopes)
    deviations = [slope - centre for slope in slopes]
    scale = _robust_scale(deviations)
    scores = [deviation / scale for deviation in deviations]
    # Differences are one shorter than values: carry the last score so
    # every observation is scored.
    return scores + [scores[-1]]


#: Selection order doubles as the tie-break: simpler detectors first.
DETECTORS: dict[str, ScoreFunction] = {
    "robust_zscore": robust_zscore_scores,
    "rolling_median_residual": rolling_median_scores,
    "local_slope": local_slope_scores,
    "forecast_interval": forecast_interval_scores,
}


def _reconstruction_score_function(adapter: Any) -> ScoreFunction:
    """Standardised reconstruction-error scores from a multi-task adapter."""
    def scores(values: list[float], season: int) -> list[float]:
        window = values[-512:]
        reconstruction = adapter.reconstruct(window)
        residuals = [value - rebuilt for value, rebuilt in zip(window, reconstruction)]
        scale = _robust_scale(residuals)
        offset = len(values) - len(window)
        return [0.0] * offset + [value / scale for value in residuals]
    return scores


def tsfm_reconstruction_detectors() -> dict[str, ScoreFunction]:
    """Reconstruction detectors for every installed sandbox whose adapter
    has the verified ``detect_anomalies`` task. Empty when none are pulled —
    the candidate pool only ever contains runnable detectors."""
    try:
        from .tsfm import tsfm_capabilities
        from .tsfm_sandbox import SubprocessAdapter, list_sandboxes
    except Exception:  # pragma: no cover - defensive: optional surface
        return {}
    detectors: dict[str, ScoreFunction] = {}
    for name in list_sandboxes():
        try:
            capabilities = tsfm_capabilities(name)
        except KeyError:
            continue
        if "detect_anomalies" not in capabilities.tasks:
            continue
        detectors[f"{name}_reconstruction"] = _reconstruction_score_function(
            SubprocessAdapter(name)
        )
    return detectors


# ---------------------------------------------------------------------------
# The synthetic-injection grader
# ---------------------------------------------------------------------------

def _content_seed(values: list[float]) -> int:
    canonical = ",".join(f"{value:.9g}" for value in values)
    return zlib.crc32(canonical.encode("utf-8"))


def _flagged(scores: list[float], threshold: float) -> set[int]:
    return {index for index, score in enumerate(scores) if abs(score) >= threshold}


def _false_alarms(stray: set[int]) -> int:
    """Count stray flags as *events*, not points.

    Recall is measured per planted anomaly — one trial, one hit — so
    precision must be measured the same way. Counting every stray point
    separately punishes a detector whose response is inherently wide
    (a slope detector marks the whole stretch that drifts, not one
    index) by roughly the width of its window, which would keep such a
    detector from ever winning selection no matter how well it worked.
    Contiguous stray flags are one false alarm.
    """
    if not stray:
        return 0
    ordered = sorted(stray)
    runs = 1
    for previous, current in zip(ordered, ordered[1:]):
        if current - previous > 1:
            runs += 1
    return runs


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
    # Long enough to be a slope change rather than a step.
    ramp_length = max(5, min(n // 8, 64))
    for _ in range(TREND_TRIALS):
        if warmup >= n - ramp_length:
            break
        onset = rng.randrange(warmup, n - ramp_length)
        sign = rng.choice((-1.0, 1.0))
        total = sign * TREND_SCALE * scale
        # A slope change, not a step: the series drifts at the wrong rate
        # across the ramp and keeps the level it reached.
        indices = list(range(onset, n))
        deltas = [
            total * min(1.0, (index - onset + 1) / ramp_length)
            for index in indices
        ]
        trials.append({"family": "trend_shift", "indices": indices,
                       "deltas": deltas,
                       "hit_indices": set(range(onset, onset + ramp_length))})
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
        try:
            clean_flags = _flagged(score_function(values, season), threshold)
            hits, false_positives = 0, 0
            by_family: dict[str, list[int]] = {}
            for trial in trials:
                injected = list(values)
                for index, delta in zip(trial["indices"], trial["deltas"]):
                    injected[index] += delta
                novel = _flagged(score_function(injected, season), threshold) - clean_flags
                strays = _false_alarms(
                    novel - trial["hit_indices"] - set(trial["indices"])
                )
                caught = bool(novel & trial["hit_indices"])
                hits += int(caught)
                false_positives += strays
                counts = by_family.setdefault(trial["family"], [0, 0, 0])
                counts[0] += int(caught)
                counts[1] += 1
                counts[2] += strays
        except Exception as exc:
            # A detector that cannot run scores zero and discloses why —
            # it must not take the whole evaluation down with it.
            grades[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                            "macro_f1": 0.0, "families": {},
                            "trials": len(trials), "false_positives": 0,
                            "error": str(exc)}
            continue
        recall = hits / len(trials) if trials else 0.0
        precision = hits / (hits + false_positives) if hits + false_positives else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        families = {}
        for family, (caught, total, strays) in sorted(by_family.items()):
            family_recall = caught / total if total else 0.0
            family_precision = (caught / (caught + strays)
                                if caught + strays else 0.0)
            family_f1 = (2 * family_precision * family_recall
                         / (family_precision + family_recall)
                         if family_precision + family_recall else 0.0)
            families[family] = {"precision": round(family_precision, 4),
                                "recall": round(family_recall, 4),
                                "f1": round(family_f1, 4), "trials": total}
        # Selection uses the macro average, not the pooled score: a
        # detector blind to one family would otherwise be crowned by
        # averaging that blindness away against families it handles,
        # which is exactly how a series whose anomalies are slope
        # changes ends up labelled by a detector that cannot see them.
        macro_f1 = (sum(entry["f1"] for entry in families.values())
                    / len(families)) if families else 0.0
        grades[name] = {
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "macro_f1": round(macro_f1, 4),
            "families": families, "trials": len(trials),
            "false_positives": false_positives,
        }
    return {
        "grades": grades,
        "injection": {
            "seed": _content_seed(values),
            "families": {
                "spike": SPIKE_TRIALS, "level_shift": SHIFT_TRIALS,
                "dropout": DROPOUT_TRIALS, "trend_shift": TREND_TRIALS,
            },
            "families_planted": sorted({trial["family"] for trial in trials}),
            "magnitudes_in_robust_scale": {
                "spike": SPIKE_SCALE, "level_shift": SHIFT_SCALE,
                "dropout": DROPOUT_SCALE, "trend_shift": TREND_SCALE,
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
        try:
            flags = _flagged(score_function(values, season), threshold)
        except Exception as exc:
            grades[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                            "labels": len(label_indices), "error": str(exc)}
            continue
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
        selection_basis = "synthetic_injection_macro_f1"
    order = list({**DETECTORS, **(extra_detectors or {})})
    # Label selection has no families to average; synthetic selection
    # ranks on coverage across every planted family.
    score_key = "f1" if label_indices else "macro_f1"
    selected = max(order,
                   key=lambda name: (ranking[name][score_key], -order.index(name)))
    scores = {**DETECTORS, **(extra_detectors or {})}[selected](values, season)
    anomalies = [
        {"timestamp": timestamps[index], "value": values[index],
         "score": round(score, 4)}
        for index, score in enumerate(scores) if abs(score) >= threshold
    ]
    best_f1 = ranking[selected][score_key]
    planted = grading["injection"]["families_planted"]
    # What the grade vouches for is exactly the families the grader
    # planted. A detector that recovers spikes says nothing about
    # anomaly kinds nobody tested it on, so the scope of the evidence
    # travels with the verdict instead of being implied by it.
    scope = (
        "The grade covers planted "
        + ", ".join(planted)
        + " anomalies only; kinds outside those families were not tested "
          "in this series and are not vouched for."
    ) if not label_indices and planted else None
    if best_f1 < GRADER_FLOOR:
        support = SupportAssessment(
            "conditionally_supported",
            [SupportReason(
                "low_grader_confidence",
                f"The best detector ({selected}) recovered planted anomalies "
                f"with F1 {best_f1:.2f} (< {GRADER_FLOOR}); real detections "
                "in this series' noise carry the same doubt.",
            )],
            assumptions=[scope] if scope else [],
            sensitivity={"threshold": threshold, "selection_basis": selection_basis,
                         "graded_families": planted},
        )
    else:
        support = SupportAssessment(
            "supported",
            [] if anomalies else [SupportReason(
                "no_anomalies_detected",
                "No point exceeded the detection threshold; absence of "
                "anomalies is a conclusion, not a failure.",
            )],
            assumptions=[scope] if scope else [],
            sensitivity={"threshold": threshold, "selection_basis": selection_basis,
                         "selected_f1": best_f1,
                         "graded_families": planted},
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
