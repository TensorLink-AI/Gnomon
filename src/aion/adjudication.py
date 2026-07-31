"""The enrichment championship ladder: one honest comparison, one winner.

Context events and covariates each earn admission in their own ablation
against the history-only forecast. That is enough to answer "does this
enrichment help?" but not "which of these enrichments should this forecast
use?" — two independent yes/no gates cannot rank a promo calendar against a
marketing-spend column, and they cannot see that two enrichments only clear
the bar together, or that they clear it twice for the same signal.

This module answers that question by running every candidate on **identical
folds**: the same rolling origins the history-only evaluation used, the same
prefix of observations at each origin, and enrichment inputs gated by
``known_at`` at that origin's cutoff. Nothing is compared across different
information sets.

Selection is a complexity ladder rather than a plain argmin. The champion
starts as the history-only forecast; each rung of added complexity — one
enrichment, then both — must beat the standing champion by
``minimum_baseline_improvement`` to unseat it. That single rule delivers
both properties the independent gates lacked:

- a joint candidate is compared against the *base* when neither enrichment
  wins alone, so enrichments that only work together can still be admitted;
- a joint candidate is compared against the *winning single* when one does
  win, so a second enrichment that merely re-encodes the first cannot buy
  admission with a rounding-error gain.

Ties never promote (the strict margin keeps the simpler champion), and
equal-scoring candidates at the same rung break by a fixed candidate order,
so the ladder is deterministic and its goldens are stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any, Callable

from .context import ContextEvent
from .context_eval import CONTEXT_MODEL_NAME, eligible_events, event_flags
from .context_model import apply_event_effect, event_adjusted, measure_effect
from .covariates import (
    COVARIATE_MODEL_NAME,
    CovariateDataset,
    covariate_fit_residuals,
    covariate_forecast,
)
from .evaluation import Evaluation, error_score, interval_bounds, quantile
from .models import MODELS, predict

BASE_CANDIDATE = "base"
CONTEXT_CANDIDATE = "context"
COVARIATE_CANDIDATE = "covariates"
JOINT_CANDIDATE = "context+covariates"

#: Fixed ordering used as the final tie-break, so equal scores never depend
#: on dict iteration or input ordering.
LADDER_ORDER = (BASE_CANDIDATE, CONTEXT_CANDIDATE, COVARIATE_CANDIDATE, JOINT_CANDIDATE)

JOINT_MODEL_NAME = f"{COVARIATE_MODEL_NAME}+{CONTEXT_MODEL_NAME}"

#: An enrichment may not publish intervals that miss the policy floor the
#: history-only forecast met. Deliberately not "no worse than the base": a
#: base that over-covers (100% of a nominal 80% interval) is wide, not
#: better, and must not veto a well-calibrated challenger.
COVERAGE_POLICY_FLOOR = 0.7
COVERAGE_WARNING_LIMIT = 0.7

# A horizon forecast for the fold whose origin is ``origin``, over the given
# target timestamps; ``None`` when the candidate cannot produce one there.
Forecaster = Callable[[int, list[datetime]], "list[float] | None"]


@dataclass
class Candidate:
    """One entrant in the ladder, with everything needed to audit it."""

    key: str
    model: str
    level: int
    forecaster: Forecaster
    events: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()
    fold_scores: list[float] = field(default_factory=list)
    mean_score: float | None = None
    points: list[float] = field(default_factory=list)
    excluded_reason: str | None = None

    @property
    def eligible(self) -> bool:
        return self.excluded_reason is None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.key,
            "model": self.model,
            "enrichment_count": self.level,
            "events": list(self.events),
            "covariates": list(self.covariates),
            "eligible": self.eligible,
            "excluded_reason": self.excluded_reason,
            "fold_scores": self.fold_scores,
            "mean_fold_score": self.mean_score,
        }


@dataclass
class EnrichmentAdjudication:
    """The record of the comparison: who ran, what they scored, who won."""

    considered: bool
    adjudicated: bool
    reasons: list[str] = field(default_factory=list)
    minimum_improvement: float = 0.0
    folds: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    winner: str | None = None
    winner_model: str | None = None
    points: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    coverage: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def winner_is_base(self) -> bool:
        return self.winner == BASE_CANDIDATE

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "adjudicated": self.adjudicated,
            "reasons": self.reasons,
            "minimum_improvement": self.minimum_improvement,
            "protocol": (
                "every candidate is scored on the same rolling origins with the "
                "same observation prefix; enrichment inputs are gated by known_at "
                "at each fold cutoff. Each rung of added complexity must beat the "
                "standing champion by minimum_improvement to unseat it."
            ),
            "folds": self.folds,
            "candidates": [candidate.to_public_dict() for candidate in self.candidates],
            "decisions": self.decisions,
            "winner": self.winner,
            "winner_model": self.winner_model,
            "measured_test_coverage": self.coverage,
        }


def _mean_fold_score(
    candidate: Candidate,
    selection_origins: list[int],
    values: list[float],
    timestamps: list[datetime],
    horizon: int,
) -> None:
    """Score one candidate on the shared folds; a candidate that cannot
    complete every fold is excluded rather than scored on a subset."""
    scores: list[float] = []
    for origin in selection_origins:
        targets = timestamps[origin : origin + horizon]
        forecast = candidate.forecaster(origin, targets)
        if forecast is None or len(forecast) != horizon:
            candidate.excluded_reason = (
                f"could not produce a forecast on the fold at origin {origin}"
            )
            return
        scores.append(error_score(values[origin : origin + horizon], forecast))
    candidate.fold_scores = scores
    candidate.mean_score = mean(scores)


def climb(
    candidates: list[Candidate], minimum_improvement: float
) -> tuple[Candidate, list[dict[str, Any]]]:
    """Walk the complexity ladder over scored candidates.

    The first candidate is the standing champion (the history-only
    forecast). Each rung — one enrichment, then two — puts forward its best
    eligible entrant, which must beat the champion's score by
    ``minimum_improvement`` to take its place. A rung with no eligible
    entrant is skipped without comment; a rung that fails is recorded with
    the score it would have needed."""
    champion = candidates[0]
    decisions: list[dict[str, Any]] = []
    for level in sorted({candidate.level for candidate in candidates} - {champion.level}):
        contenders = [
            candidate for candidate in candidates
            if candidate.level == level and candidate.eligible
        ]
        if not contenders:
            continue
        best = min(contenders, key=lambda item: (item.mean_score, LADDER_ORDER.index(item.key)))
        required = champion.mean_score * (1 - minimum_improvement)
        # Strictly better by the margin, so a tie always keeps the simpler
        # champion; a perfect champion can never be unseated.
        promoted = champion.mean_score > 0 and best.mean_score <= required
        decisions.append({
            "enrichment_count": level,
            "champion": champion.key,
            "champion_score": champion.mean_score,
            "challenger": best.key,
            "challenger_score": best.mean_score,
            "required_score": required,
            "outcome": "promoted" if promoted else "held",
            "reason": (
                f"{best.key} scored {best.mean_score:.6f} on the shared folds, at or "
                f"below the {required:.6f} needed to beat {champion.key} "
                f"({champion.mean_score:.6f}) by the {minimum_improvement} margin"
                if promoted else
                f"{best.key} scored {best.mean_score:.6f}; unseating {champion.key} "
                f"({champion.mean_score:.6f}) by the {minimum_improvement} margin "
                f"requires {required:.6f} or better"
            ),
        })
        if promoted:
            champion = best
    return champion, decisions


def _build_candidates(
    values: list[float],
    timestamps: list[datetime],
    *,
    base_model: str,
    horizon: int,
    season: int,
    events: list[ContextEvent],
    dataset: CovariateDataset | None,
    covariate_names: list[str],
    series_name: str,
) -> list[Candidate]:
    def base_forecaster(origin: int, targets: list[datetime]) -> list[float] | None:
        try:
            return predict(base_model, values[:origin], horizon, season)
        except (ValueError, ArithmeticError):
            return None

    def context_forecaster(origin: int, targets: list[datetime]) -> list[float] | None:
        cutoff = timestamps[origin - 1]
        try:
            return event_adjusted(
                values[:origin], horizon, season,
                event_flags(events, timestamps[:origin], cutoff),
                event_flags(events, targets, cutoff),
            )
        except ValueError:
            return None

    def covariate_forecaster(origin: int, targets: list[datetime]) -> list[float] | None:
        if dataset is None:
            return None
        return covariate_forecast(
            values[:origin], timestamps[:origin], targets, dataset,
            covariate_names, series_name, timestamps[origin - 1], season,
        )

    def joint_forecaster(origin: int, targets: list[datetime]) -> list[float] | None:
        # The covariate model's forecast, adjusted by the part of the event
        # that model has *not* already explained: the effect is measured on
        # its in-sample residuals, not on raw history. Two enrichments
        # encoding the same signal therefore compose to roughly the stronger
        # one rather than to double its effect.
        if dataset is None:
            return None
        base_points = covariate_forecaster(origin, targets)
        if base_points is None:
            return None
        cutoff = timestamps[origin - 1]
        fit = covariate_fit_residuals(
            values[:origin], timestamps[:origin], dataset, covariate_names,
            series_name, cutoff, season,
        )
        if fit is None:
            return None
        indices, residuals = fit
        active_history = event_flags(events, timestamps[:origin], cutoff)
        try:
            effect = measure_effect(residuals, [active_history[index] for index in indices])
        except ValueError:
            return None
        return apply_event_effect(base_points, effect, event_flags(events, targets, cutoff))

    event_ids = tuple(event.event_id for event in events)
    covariate_ids = tuple(covariate_names)
    candidates = [
        Candidate(BASE_CANDIDATE, base_model, 0, base_forecaster),
    ]
    if events:
        candidates.append(Candidate(
            CONTEXT_CANDIDATE, CONTEXT_MODEL_NAME, 1, context_forecaster,
            events=event_ids,
        ))
    if dataset is not None and covariate_names:
        candidates.append(Candidate(
            COVARIATE_CANDIDATE, COVARIATE_MODEL_NAME, 1, covariate_forecaster,
            covariates=covariate_ids,
        ))
    if events and dataset is not None and covariate_names:
        candidates.append(Candidate(
            JOINT_CANDIDATE, JOINT_MODEL_NAME, 2, joint_forecaster,
            events=event_ids, covariates=covariate_ids,
        ))
    return candidates


def adjudicate_enrichments(
    values: list[float],
    timestamps: list[datetime],
    future_timestamps: list[datetime],
    *,
    series_name: str,
    horizon: int,
    season: int,
    minimum_improvement: float,
    base: Evaluation,
    base_model: str | None,
    context_events: list[ContextEvent] | None,
    covariates: CovariateDataset | None,
    covariate_names: list[str],
) -> EnrichmentAdjudication:
    """Run the ladder and return the winner with its calibration.

    ``covariate_names`` is the covariate set the ablation retained (or the
    full declared set when it retained none — a covariate that cannot carry a
    forecast alone may still carry one alongside context, and the ladder's
    margin rule is what keeps it honest)."""
    adjudication = EnrichmentAdjudication(True, False, minimum_improvement=minimum_improvement)
    if not base.supported or base_model is None:
        adjudication.considered = False
        adjudication.reasons.append(
            "history-only evaluation is unsupported; enrichments cannot rescue an "
            "unsupported task"
        )
        return adjudication
    if base_model not in MODELS:
        adjudication.considered = False
        adjudication.reasons.append(
            f"the history-only forecast uses {base_model!r}, which the ladder cannot "
            "replay fold by fold; adjudication requires a built-in univariate "
            "comparison model"
        )
        return adjudication
    if timestamps and timestamps[0].tzinfo is None and context_events:
        adjudication.considered = False
        adjudication.reasons.append(
            "dataset timestamps are timezone-naive; context events require "
            "timezone-aware data"
        )
        return adjudication

    events, _ = eligible_events(context_events or [], series_name)
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = list(range(minimum_train, len(values) - horizon + 1, horizon))
    if len(origins) < 4:
        adjudication.reasons.append(
            "fewer than four rolling folds are available; adjudication requires "
            "fully separated selection, calibration, and test folds"
        )
        return adjudication
    selection_origins, calibration_origin, test_origin = origins[:-2], origins[-2], origins[-1]

    candidates = _build_candidates(
        values, timestamps, base_model=base_model, horizon=horizon, season=season,
        events=events, dataset=covariates, covariate_names=covariate_names,
        series_name=series_name,
    )
    adjudication.candidates = candidates
    adjudication.folds = [
        {
            "origin": origin,
            "cutoff": timestamps[origin - 1].isoformat(),
            "first_target": timestamps[origin].isoformat(),
        }
        for origin in selection_origins
    ]
    if len(candidates) < 2:
        adjudication.reasons.append(
            "no enrichment challenger was available; the history-only forecast stands"
        )
        return adjudication

    for candidate in candidates:
        _mean_fold_score(candidate, selection_origins, values, timestamps, horizon)
        if candidate.eligible:
            final = candidate.forecaster(len(values), future_timestamps)
            if final is None or len(final) != horizon:
                candidate.excluded_reason = (
                    "could not produce a forecast over the requested horizon"
                )
            else:
                candidate.points = final

    champion = candidates[0]
    if not champion.eligible:
        adjudication.reasons.append(
            f"the history-only candidate could not be replayed on the shared folds: "
            f"{champion.excluded_reason}"
        )
        return adjudication

    champion, decisions = climb(candidates, minimum_improvement)
    adjudication.decisions = decisions
    adjudication.adjudicated = True
    adjudication.winner = champion.key
    adjudication.winner_model = champion.model
    adjudication.points = champion.points
    adjudication.reasons.append(
        f"{champion.key} won the ladder with a mean fold score of "
        f"{champion.mean_score:.6f} over {len(selection_origins)} shared folds"
    )
    if champion.key != BASE_CANDIDATE:
        rejection = _calibrate(
            adjudication, champion, values, timestamps, base_coverage=base.coverage,
            calibration_origin=calibration_origin, test_origin=test_origin,
            horizon=horizon,
        )
        if rejection is not None:
            adjudication.winner = BASE_CANDIDATE
            adjudication.winner_model = base_model
            adjudication.points = []
            adjudication.residuals = []
            adjudication.coverage = None
            adjudication.warnings = []
            adjudication.reasons.append(rejection)
    return adjudication


def _calibrate(
    adjudication: EnrichmentAdjudication,
    champion: Candidate,
    values: list[float],
    timestamps: list[datetime],
    *,
    base_coverage: float | None,
    calibration_origin: int,
    test_origin: int,
    horizon: int,
) -> str | None:
    """Residuals from the held-out calibration fold, coverage from the final
    test fold — the same protocol the history-only evaluation publishes.

    Returns a rejection reason when the winner cannot be calibrated, or when
    its intervals degrade past policy: winning on point error is not a licence
    to publish worse uncertainty than the forecast it displaces."""
    calibration = champion.forecaster(
        calibration_origin, timestamps[calibration_origin : calibration_origin + horizon],
    )
    if calibration is None:
        return (
            f"{champion.key} won on the selection folds but could not be calibrated "
            "on the held-out fold; the history-only forecast stands"
        )
    adjudication.residuals = [
        actual - predicted
        for actual, predicted in zip(
            values[calibration_origin : calibration_origin + horizon], calibration,
        )
    ]
    residual_quantiles = {p: quantile(adjudication.residuals, p) for p in (0.1, 0.5, 0.9)}
    test = champion.forecaster(test_origin, timestamps[test_origin : test_origin + horizon])
    if test is None:
        return None
    covered = []
    for step, (actual, prediction) in enumerate(
        zip(values[test_origin : test_origin + horizon], test), 1,
    ):
        low, _, high = interval_bounds(prediction, residual_quantiles, step)
        covered.append(1.0 if low <= actual <= high else 0.0)
    adjudication.coverage = mean(covered)
    introduced_miscalibration = (
        adjudication.coverage < COVERAGE_POLICY_FLOOR
        and (base_coverage is None or base_coverage >= COVERAGE_POLICY_FLOOR)
    )
    if introduced_miscalibration:
        return (
            f"{champion.key} won on point error but its 80% intervals covered only "
            f"{adjudication.coverage:.1%} of the final test fold, below the "
            f"{COVERAGE_POLICY_FLOOR:.0%} policy floor the history-only forecast met; "
            "the history-only forecast stands"
        )
    if adjudication.coverage < COVERAGE_WARNING_LIMIT:
        adjudication.warnings.append(
            f"Final-test 80% interval coverage was {adjudication.coverage:.1%}, below 70%."
        )
    return None
