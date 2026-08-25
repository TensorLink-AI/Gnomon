"""The model-assisted lane: preserve useful model priors beside the governed answer.

The immutable governed lane publishes the safest backtested forecast, or its
disclosed naive fallback when the evaluation protocol cannot run.  The
cross-model evaluation (docs/cross-model-evaluation-2026-08.md) showed that
this floor, while honest, discards useful predictive signal: a candidate the
folds could not *rank* may still carry the only model evidence the history
admits.  This module builds the second lane that record adopted: the best
model prior the history supports, published *beside* the governed rows —
labelled, never blended into them, and never replacing them.

Admission follows the evidence, in the admission vocabulary already used by
:mod:`gnomon.admission`:

- ``conditionally_supported`` — the candidate beat the published baseline on
  a genuine out-of-sample comparison at the requested horizon, but the full
  separated protocol could not run.
- ``prior_assisted`` — the candidate beat the published baseline on weaker
  out-of-sample evidence (a single underpowered fold, or a reduced-rigor
  holdout shorter than the horizon) and passed deterministic plausibility
  checks.

A candidate with no out-of-sample win at all earns no lane: with zero
evidence in its favour, the governed floor is already the better-evidenced
answer, and publishing a bare assertion beside it would be exactly the
laundering the abstention contract forbids.
"""

from __future__ import annotations

from typing import Any

from .admission import OutputDiagnostics, output_diagnostics
from .contracts import Evidence, SupportReason
from .models import BASELINES, MODELS, predict

MODEL_ASSISTED_LANE_VERSION = "0.1"

#: Plausibility bounds, in robust one-step scales of the history.  These are
#: deliberately generous: they exist to reject pathology (a first step ten
#: scales away from the last observation, a path twenty times more volatile
#: than the history), not to second-guess a candidate the evidence admitted.
MAX_BOUNDARY_JUMP = 10.0
MAX_SCALE_RATIO = 20.0


def _plausible(values: list[float], candidate: list[float],
               baseline: list[float]) -> tuple[bool, OutputDiagnostics]:
    diagnostics = output_diagnostics(values, candidate, baseline)
    if not diagnostics.valid:
        return False, diagnostics
    if (diagnostics.boundary_jump or 0.0) > MAX_BOUNDARY_JUMP:
        return False, diagnostics
    if (diagnostics.scale_ratio or 0.0) > MAX_SCALE_RATIO:
        return False, diagnostics
    return True, diagnostics


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _diagnostics_payload(diagnostics: OutputDiagnostics) -> dict[str, Any]:
    return {
        "valid": diagnostics.valid,
        "boundary_jump": _round(diagnostics.boundary_jump),
        "scale_ratio": _round(diagnostics.scale_ratio),
        "oscillation_ratio": _round(diagnostics.oscillation_ratio),
        "candidate_baseline_disagreement":
            _round(diagnostics.candidate_baseline_disagreement),
    }


def _scored_candidate(assessment: Any) -> tuple[str, float, float] | None:
    """The best already-scored non-baseline candidate that beat the published
    baseline on the evaluation's own (underpowered) out-of-sample comparison."""
    scores = dict(assessment.selection_scores or {})
    baseline = assessment.strongest_baseline
    baseline_score = scores.get(baseline)
    if baseline_score is None:
        return None
    valid = {
        name: float(score) for name, score in scores.items()
        if score is not None and name not in BASELINES and name in MODELS
    }
    if not valid:
        return None
    name = min(valid, key=lambda key: (valid[key], key))
    if valid[name] >= float(baseline_score):
        return None
    return name, valid[name], float(baseline_score)


def _holdout_candidate(values: list[float], horizon: int,
                       season: int) -> tuple[str, float, float, int] | None:
    """Score the built-in non-baseline models on one reduced-rigor holdout.

    Used only when the evaluation scored nothing at all.  The holdout reuses
    the degraded protocol's own sizing rule and compares every candidate
    against ``last_value`` on the identical held-out points; a winner is
    evidence, never a ranking.
    """
    from .evaluation import error_score

    if len(values) < 8:
        return None
    holdout = min(horizon, max(1, len(values) // 4))
    origin = len(values) - holdout
    if origin < 4:
        return None
    train, actual = values[:origin], values[origin:]
    try:
        baseline_score = error_score(
            actual, predict("last_value", train, holdout, season))
    except (ValueError, ArithmeticError):
        return None
    if baseline_score is None:
        return None
    best: tuple[str, float] | None = None
    for name in sorted(MODELS):
        if name in BASELINES:
            continue
        try:
            score = error_score(actual, predict(name, train, holdout, season))
        except (ValueError, ArithmeticError):
            continue
        if score is None or score >= baseline_score:
            continue
        if best is None or (score, name) < (best[1], best[0]):
            best = (name, score)
    if best is None:
        return None
    return best[0], best[1], float(baseline_score), holdout


def build_model_assisted_lane(
    series: str,
    values: list[float],
    *,
    horizon: int,
    season: int,
    future_timestamps: list[Any],
    assessment: Any,
    published_support: str,
    selected_model: str | None,
) -> tuple[dict[str, Any] | None, SupportReason | None, Evidence | None]:
    """Build the labelled second lane for one series, or nothing.

    Returns ``(lane, disclosure, evidence)``.  The lane never carries
    intervals, never claims automation eligibility, and never changes the
    primary forecast; the caller attaches it additively.
    """
    if horizon <= 0 or len(future_timestamps) != horizon or len(values) < 8:
        return None, None, None

    candidate: str | None = None
    validation: dict[str, Any] | None = None
    evidence_steps = 0
    if (assessment is not None
            and getattr(assessment, "selection_guardrail_applied", False)
            and selected_model == assessment.strongest_baseline):
        scored = _scored_candidate(assessment)
        if scored is not None:
            candidate, candidate_score, baseline_score = scored
            degraded = bool(getattr(assessment, "degraded", False))
            evidence_steps = (min(horizon, max(1, len(values) // 4))
                              if degraded else horizon)
            validation = {
                "basis": ("single_trailing_holdout" if degraded
                          else "single_selection_fold"),
                "out_of_sample_steps": evidence_steps,
                "comparisons": int(getattr(
                    assessment, "selection_fold_count", 1) or 1),
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "baseline": assessment.strongest_baseline,
                "scores_are_evidence_not_a_ranking": True,
            }
    elif published_support == "best_effort":
        held = _holdout_candidate(values, horizon, season)
        if held is not None:
            candidate, candidate_score, baseline_score, evidence_steps = held
            validation = {
                "basis": "reduced_rigor_holdout",
                "out_of_sample_steps": evidence_steps,
                "comparisons": 1,
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "baseline": "last_value",
                "scores_are_evidence_not_a_ranking": True,
            }
    if candidate is None or validation is None:
        return None, None, None

    try:
        points = predict(candidate, values, horizon, season)
        baseline_points = predict("last_value", values, horizon, season)
    except (ValueError, ArithmeticError):
        return None, None, None
    plausible, diagnostics = _plausible(values, points, baseline_points)
    if not plausible:
        return None, None, None

    # A full-horizon out-of-sample win is degraded validation; anything
    # weaker is a prior with a shred of evidence. Both are underpowered, and
    # neither is suppressed — that is the lane's entire purpose.
    support = ("conditionally_supported" if evidence_steps >= horizon
               else "prior_assisted")
    lane = {
        "version": MODEL_ASSISTED_LANE_VERSION,
        "support": support,
        "selected_model": candidate,
        # Points only, on the primary rows' own timestamp grid: the lane has
        # no calibrated residuals, so publishing quantiles or per-row
        # envelopes here would manufacture probability weight from nothing.
        "points": [round(float(point), 6) for point in points],
        "timestamps_match_primary_forecast": True,
        "validation": validation,
        "plausibility": _diagnostics_payload(diagnostics),
        "automation_eligible": False,
        "primary_forecast_unchanged": True,
    }
    disclosure = SupportReason(
        "model_assisted_lane",
        f"A {support} model-assisted forecast ({candidate}) is published in "
        f"the model_assisted lane beside the governed rows: it beat "
        f"{validation['baseline']} on {validation['basis']} "
        f"({evidence_steps} out-of-sample step"
        f"{'s' if evidence_steps != 1 else ''}) and passed plausibility "
        f"checks. It carries no calibrated intervals, is not eligible for "
        f"automatic action, and never replaces the primary forecast.",
    )
    evidence = Evidence(
        f"model_assisted:{series}", "model_assisted_lane", series, {
            "support": support,
            "selected_model": candidate,
            "validation": validation,
            "plausibility": _diagnostics_payload(diagnostics),
            "primary_forecast_unchanged": True,
        },
    )
    return lane, disclosure, evidence
