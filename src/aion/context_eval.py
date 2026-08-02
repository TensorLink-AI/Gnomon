"""The context admission gate: identical-fold ablation with known-at gating.

Context events never improve a forecast by assertion. This module replays
the exact selection folds used by the history-only evaluation, offering a
context-adjusted candidate the same information a real forecaster would
have had at each fold cutoff — only events that were verifiably known by
that cutoff, per ``aion.context.backtest_admissible``. Admission then
requires a stable, material improvement:

- every selection fold completes with context;
- mean improvement over the history-only selection meets the margin;
- more than half of the folds improve;
- the gain survives removing the single best fold; and
- measured interval coverage does not degrade beyond policy (0.1).

Anything else keeps the history-only result and records why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any

from .context import ContextEvent, backtest_admissible, event_applies
from .context_model import EFFECT_SHAPES, event_adjusted
from .evaluation import (
    Evaluation,
    conformal_spreads,
    error_score,
    interval_from_spread,
)
from .models import predict

COVERAGE_DEGRADATION_LIMIT = 0.1
CONTEXT_MODEL_NAME = "event_adjusted"


def wilson_upper(successes: float, trials: int, z: float = 1.96) -> float:
    """Upper bound of a Wilson score interval for a proportion.

    Coverage is measured on one test fold — a handful of points — so the
    point estimate is noisy enough that a veto based on it fires on
    sampling error alone. Comparing the *upper* bound against the policy
    limit means admission is refused only when the degradation is larger
    than the measurement's own uncertainty.
    """
    if trials <= 0:
        return 1.0
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    margin = z * ((proportion * (1 - proportion) / trials
                   + z * z / (4 * trials * trials)) ** 0.5)
    return min(1.0, (centre + margin) / denominator)


@dataclass
class ContextAssessment:
    considered: bool
    admitted: bool
    reasons: list[str]
    events_used: list[str] = field(default_factory=list)
    events_excluded: list[dict[str, str]] = field(default_factory=list)
    fold_improvements: list[float] = field(default_factory=list)
    mean_improvement: float | None = None
    coverage: float | None = None
    residuals: list[float] = field(default_factory=list)
    points: list[float] = field(default_factory=list)
    residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Which effect shape the ablation chose, and how every shape scored on
    # the same folds. Disclosed because a shape chosen by measurement is
    # only trustworthy if the measurement is visible.
    effect_shape: str = "level"
    shape_scores: dict[str, float] = field(default_factory=dict)
    # One entry per gate condition evaluated, whether it passed or not, with
    # the number it was decided on. `reasons` says what went wrong in prose;
    # this says what was measured, so admission rates and rejection causes
    # are countable across a run rather than parsed out of sentences.
    gate_checks: list[dict[str, Any]] = field(default_factory=list)

    def record_check(self, code: str, passed: bool, *, measured: Any = None,
                     threshold: Any = None, detail: str | None = None) -> None:
        entry: dict[str, Any] = {"code": code, "passed": passed}
        if measured is not None:
            entry["measured"] = measured
        if threshold is not None:
            entry["threshold"] = threshold
        if detail:
            entry["detail"] = detail
        self.gate_checks.append(entry)
        if not passed and detail:
            self.reasons.append(detail)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "admitted": self.admitted,
            "model": CONTEXT_MODEL_NAME if self.admitted else None,
            "reasons": self.reasons,
            "events_used": self.events_used,
            "events_excluded": self.events_excluded,
            "fold_improvements": self.fold_improvements,
            "mean_improvement": self.mean_improvement,
            "effect_shape": self.effect_shape,
            "shape_scores": self.shape_scores,
            "measured_coverage": self.coverage,
            "gate_checks": self.gate_checks,
        }


def eligible_events(
    events: list[ContextEvent], series_name: str
) -> tuple[list[ContextEvent], list[dict[str, str]]]:
    eligible: list[ContextEvent] = []
    excluded: list[dict[str, str]] = []
    for event in events:
        if not event_applies(event, series_name):
            excluded.append({"event_id": event.event_id, "reason": "scope does not include this series"})
        elif not backtest_admissible(event):
            excluded.append({
                "event_id": event.event_id,
                "reason": "no verifiable source; not admissible for backtesting",
            })
        else:
            eligible.append(event)
    return eligible, excluded


def event_flags(
    events: list[ContextEvent], timestamps: list[datetime], cutoff: datetime
) -> list[bool]:
    known = [
        event for event in events
        if datetime.fromisoformat(event.known_at) <= cutoff
    ]
    flags: list[bool] = []
    for timestamp in timestamps:
        flags.append(any(
            datetime.fromisoformat(event.effective_start)
            <= timestamp
            <= datetime.fromisoformat(event.effective_end)
            for event in known
        ))
    return flags


def assess_context(
    values: list[float],
    timestamps: list[datetime],
    future_timestamps: list[datetime],
    events: list[ContextEvent],
    series_name: str,
    horizon: int,
    season: int,
    minimum_improvement: float,
    base: Evaluation,
) -> ContextAssessment:
    eligible, excluded = eligible_events(events, series_name)
    if timestamps and timestamps[0].tzinfo is None:
        return ContextAssessment(
            False, False,
            ["dataset timestamps are timezone-naive; context events require timezone-aware data"],
            events_excluded=excluded,
        )
    if not base.supported or base.selected_model is None:
        return ContextAssessment(
            False, False,
            ["history-only evaluation is unsupported; context cannot rescue an unsupported task"],
            events_excluded=excluded,
        )
    if not eligible:
        assessment = ContextAssessment(True, False, [], events_excluded=excluded)
        assessment.record_check(
            "events_eligible", False, measured=0, threshold=1,
            detail="no admissible events apply to this series",
        )
        return assessment

    # Replay the exact partitioning used by evaluate(): earlier folds select,
    # the penultimate fold calibrates, the final fold reports.
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = list(range(minimum_train, len(values) - horizon + 1, horizon))
    if len(origins) < 4:
        assessment = ContextAssessment(
            True, False, [],
            events_used=[event.event_id for event in eligible],
            events_excluded=excluded,
        )
        assessment.record_check(
            "separated_folds_available", False,
            measured=len(origins), threshold=4,
            detail="history evaluation ran in degraded mode (fewer than four "
                   "rolling folds); context ablation requires fully separated "
                   "selection, calibration, and test folds",
        )
        return assessment
    selection_origins, calibration_origin, test_origin = origins[:-2], origins[-2], origins[-1]

    def _reject(code: str, measured: Any, detail: str) -> ContextAssessment:
        assessment = ContextAssessment(
            True, False, [],
            events_used=[event.event_id for event in eligible],
            events_excluded=excluded,
        )
        assessment.record_check(code, False, measured=measured, detail=detail)
        return assessment

    # Each shape is scored on the *same* folds, so the shape is chosen by
    # measurement rather than by the caller. A caller who could name the
    # shape could fit a story to the data, which is exactly what this gate
    # exists to prevent.
    # The candidate keeps its own drift base rather than being layered onto
    # the selected model. Layering was tried and is wrong: `event_effect`
    # measures the raw active-minus-inactive level difference, so adding it
    # to a base that *already* models the event -- any seasonal model, when
    # the events recur on a period -- counts the same bump twice, and fold
    # improvements collapse. The comparison here is therefore between two
    # complete models, which is also what makes it a fair contest.
    #
    # The known limitation this leaves: on a series where a seasonal model
    # beats drift by more than the event is worth, context cannot win however
    # real its effect. Fixing that properly means estimating the effect from
    # the *base model's residuals* during active periods rather than from the
    # raw level, which is a larger change than this phase.
    base_paths = {
        origin: predict(base.selected_model, values[:origin], horizon, season)
        for origin in selection_origins
    }
    by_shape: dict[str, list[float]] = {}
    for shape in EFFECT_SHAPES:
        improvements: list[float] = []
        failed: str | None = None
        for origin in selection_origins:
            cutoff = timestamps[origin - 1]
            actual = values[origin : origin + horizon]
            try:
                context_prediction = event_adjusted(
                    values[:origin], horizon, season,
                    event_flags(eligible, timestamps[:origin], cutoff),
                    event_flags(eligible, timestamps[origin : origin + horizon], cutoff),
                    shape,
                )
            except ValueError as exc:
                failed = str(exc)
                break
            base_score = error_score(actual, base_paths[origin])
            context_score = error_score(actual, context_prediction)
            if base_score is None or context_score is None:
                # No scale in this window. Scoring one arm and not the other
                # would break the identical-folds comparison the gate rests on.
                return _reject(
                    "folds_are_scoreable", origin,
                    "a selection fold has no scale to score against, so the "
                    "context candidate cannot be compared on identical folds",
                )
            # Symmetric relative improvement, bounded to [-1, 1]: a fold where
            # both candidates are (near-)exact contributes 0 rather than
            # aborting or dividing by zero.
            denominator = max(base_score, context_score)
            improvements.append(
                0.0 if denominator <= 1e-12 else (base_score - context_score) / denominator
            )
        if failed is None:
            by_shape[shape] = improvements
        elif shape == "level":
            # The level shape is the one the gate has always used; if it
            # cannot fit, no shape can, and the reason is the honest one.
            return _reject(
                "candidate_fits_every_fold", failed,
                f"context candidate failed a selection fold: {failed}",
            )

    if not by_shape:
        return _reject(
            "candidate_fits_every_fold", "no shape fitted every fold",
            "no effect shape fitted every selection fold",
        )
    selected_shape = max(by_shape, key=lambda name: mean(by_shape[name]))
    improvements = by_shape[selected_shape]

    assessment = ContextAssessment(
        True, False, [],
        events_used=[event.event_id for event in eligible],
        events_excluded=excluded,
        fold_improvements=improvements,
        mean_improvement=mean(improvements),
        effect_shape=selected_shape,
        shape_scores={name: round(mean(scores), 6)
                      for name, scores in by_shape.items()},
    )

    assessment.record_check(
        "events_eligible", True, measured=len(eligible), threshold=1,
    )
    assessment.record_check(
        "separated_folds_available", True, measured=len(origins), threshold=4,
    )
    assessment.record_check(
        "candidate_fits_every_fold", True, measured=len(improvements),
    )
    assessment.record_check(
        "mean_improvement_meets_margin",
        assessment.mean_improvement >= minimum_improvement,
        measured=round(assessment.mean_improvement, 6),
        threshold=minimum_improvement,
        detail=(f"mean fold improvement {assessment.mean_improvement:.3f} is below "
                f"the required margin {minimum_improvement}"),
    )
    improved = sum(1 for value in improvements if value > 0)
    assessment.record_check(
        "majority_of_folds_improve", improved * 2 > len(improvements),
        measured=improved, threshold=len(improvements) // 2 + 1,
        detail=(f"only {improved} of {len(improvements)} folds improved; "
                "a majority is required"),
    )
    if len(improvements) > 1:
        without_single_best = sorted(improvements)[:-1]
        assessment.record_check(
            "gain_survives_best_fold", mean(without_single_best) > 0,
            measured=round(mean(without_single_best), 6), threshold=0.0,
            detail="the gain is confined to a single fold",
        )

    # Calibrate and measure coverage with context before deciding, so a
    # coverage regression can veto admission.
    calibration_cutoff = timestamps[calibration_origin - 1]
    calibration_prediction = event_adjusted(
        values[:calibration_origin], horizon, season,
        event_flags(eligible, timestamps[:calibration_origin], calibration_cutoff),
        event_flags(eligible, timestamps[calibration_origin : calibration_origin + horizon], calibration_cutoff),
        selected_shape,
    )
    calibration_actual = values[calibration_origin : calibration_origin + horizon]
    assessment.residuals = [
        actual - predicted for actual, predicted in zip(calibration_actual, calibration_prediction)
    ]
    assessment.residuals_by_lead = {
        step: [residual]
        for step, residual in enumerate(assessment.residuals, 1)
    }
    spreads = conformal_spreads(
        assessment.residuals_by_lead, horizon, assessment.residuals,
    )

    test_cutoff = timestamps[test_origin - 1]
    test_prediction = event_adjusted(
        values[:test_origin], horizon, season,
        event_flags(eligible, timestamps[:test_origin], test_cutoff),
        event_flags(eligible, timestamps[test_origin : test_origin + horizon], test_cutoff),
        selected_shape,
    )
    test_actual = values[test_origin : test_origin + horizon]
    covered = []
    for step, (actual, prediction) in enumerate(zip(test_actual, test_prediction), 1):
        spread = spreads.get(step)
        if spread is None:
            continue
        low, _, high = interval_from_spread(prediction, spread)
        covered.append(1.0 if low <= actual <= high else 0.0)
    assessment.coverage = mean(covered) if covered else None
    if base.coverage is not None and assessment.coverage is not None:
        # Veto on the interval, not the point estimate: with one test fold
        # of `horizon` points, a coverage drop well inside sampling noise
        # would otherwise reject context that never degraded anything.
        limit = base.coverage - COVERAGE_DEGRADATION_LIMIT
        upper = wilson_upper(sum(covered), len(covered))
        assessment.record_check(
            "coverage_not_degraded", upper >= limit,
            measured={"coverage": round(assessment.coverage, 4),
                      "upper_bound": round(upper, 4),
                      "points": len(covered)},
            threshold=round(limit, 4),
            detail=(f"interval coverage degraded from {base.coverage:.1%} to "
                    f"{assessment.coverage:.1%} (upper bound {upper:.1%}, "
                    f"below the {limit:.1%} policy limit)"),
        )

    if assessment.reasons:
        return assessment

    assessment.admitted = True
    final_cutoff = timestamps[-1]
    assessment.points = event_adjusted(
        values, horizon, season,
        event_flags(eligible, timestamps, final_cutoff),
        event_flags(eligible, future_timestamps, final_cutoff),
        selected_shape,
    )
    if assessment.coverage < 0.7:
        assessment.warnings.append(
            f"Final-test 80% interval coverage was {assessment.coverage:.1%}, below 70%."
        )
    return assessment
