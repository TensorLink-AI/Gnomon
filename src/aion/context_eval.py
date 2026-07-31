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
from .context_model import event_adjusted
from .evaluation import Evaluation, error_score, interval_bounds, quantile
from .models import predict

COVERAGE_DEGRADATION_LIMIT = 0.1
CONTEXT_MODEL_NAME = "event_adjusted"


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
    warnings: list[str] = field(default_factory=list)

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
            "measured_coverage": self.coverage,
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
        return ContextAssessment(
            True, False, ["no admissible events apply to this series"],
            events_excluded=excluded,
        )

    # Replay the exact partitioning used by evaluate(): earlier folds select,
    # the penultimate fold calibrates, the final fold reports.
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = list(range(minimum_train, len(values) - horizon + 1, horizon))
    if len(origins) < 4:
        return ContextAssessment(
            True, False,
            ["history evaluation ran in degraded mode (fewer than four rolling "
             "folds); context ablation requires fully separated selection, "
             "calibration, and test folds"],
            events_used=[event.event_id for event in eligible],
            events_excluded=excluded,
        )
    selection_origins, calibration_origin, test_origin = origins[:-2], origins[-2], origins[-1]

    improvements: list[float] = []
    for origin in selection_origins:
        cutoff = timestamps[origin - 1]
        actual = values[origin : origin + horizon]
        try:
            context_prediction = event_adjusted(
                values[:origin], horizon, season,
                event_flags(eligible, timestamps[:origin], cutoff),
                event_flags(eligible, timestamps[origin : origin + horizon], cutoff),
            )
        except ValueError as exc:
            return ContextAssessment(
                True, False,
                [f"context candidate failed a selection fold: {exc}"],
                events_used=[event.event_id for event in eligible],
                events_excluded=excluded,
            )
        base_score = error_score(actual, predict(base.selected_model, values[:origin], horizon, season))
        context_score = error_score(actual, context_prediction)
        # Symmetric relative improvement, bounded to [-1, 1]: a fold where
        # both candidates are (near-)exact contributes 0 rather than
        # aborting or dividing by zero.
        denominator = max(base_score, context_score)
        improvements.append(
            0.0 if denominator <= 1e-12 else (base_score - context_score) / denominator
        )

    assessment = ContextAssessment(
        True, False, [],
        events_used=[event.event_id for event in eligible],
        events_excluded=excluded,
        fold_improvements=improvements,
        mean_improvement=mean(improvements),
    )

    if assessment.mean_improvement < minimum_improvement:
        assessment.reasons.append(
            f"mean fold improvement {assessment.mean_improvement:.3f} is below the "
            f"required margin {minimum_improvement}"
        )
    improved = sum(1 for value in improvements if value > 0)
    if improved * 2 <= len(improvements):
        assessment.reasons.append(
            f"only {improved} of {len(improvements)} folds improved; a majority is required"
        )
    if len(improvements) > 1:
        without_single_best = sorted(improvements)[:-1]
        if mean(without_single_best) <= 0:
            assessment.reasons.append("the gain is confined to a single fold")

    # Calibrate and measure coverage with context before deciding, so a
    # coverage regression can veto admission.
    calibration_cutoff = timestamps[calibration_origin - 1]
    calibration_prediction = event_adjusted(
        values[:calibration_origin], horizon, season,
        event_flags(eligible, timestamps[:calibration_origin], calibration_cutoff),
        event_flags(eligible, timestamps[calibration_origin : calibration_origin + horizon], calibration_cutoff),
    )
    calibration_actual = values[calibration_origin : calibration_origin + horizon]
    assessment.residuals = [
        actual - predicted for actual, predicted in zip(calibration_actual, calibration_prediction)
    ]
    residual_quantiles = {p: quantile(assessment.residuals, p) for p in (0.1, 0.5, 0.9)}

    test_cutoff = timestamps[test_origin - 1]
    test_prediction = event_adjusted(
        values[:test_origin], horizon, season,
        event_flags(eligible, timestamps[:test_origin], test_cutoff),
        event_flags(eligible, timestamps[test_origin : test_origin + horizon], test_cutoff),
    )
    test_actual = values[test_origin : test_origin + horizon]
    covered = []
    for step, (actual, prediction) in enumerate(zip(test_actual, test_prediction), 1):
        low, _, high = interval_bounds(prediction, residual_quantiles, step)
        covered.append(1.0 if low <= actual <= high else 0.0)
    assessment.coverage = mean(covered)
    if base.coverage is not None and assessment.coverage < base.coverage - COVERAGE_DEGRADATION_LIMIT:
        assessment.reasons.append(
            f"interval coverage degraded from {base.coverage:.1%} to {assessment.coverage:.1%}"
        )

    if assessment.reasons:
        return assessment

    assessment.admitted = True
    final_cutoff = timestamps[-1]
    assessment.points = event_adjusted(
        values, horizon, season,
        event_flags(eligible, timestamps, final_cutoff),
        event_flags(eligible, future_timestamps, final_cutoff),
    )
    if assessment.coverage < 0.7:
        assessment.warnings.append(
            f"Final-test 80% interval coverage was {assessment.coverage:.1%}, below 70%."
        )
    return assessment
