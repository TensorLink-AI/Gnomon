"""The enrichment championship ladder: fold-identical adjudication.

When one run supplies both context events and covariates, each enrichment
first faces its own independent ablation gate (``assess_context``,
``assess_covariates``). This module then compares the base model against
every independently-admitted challenger — base + context, base +
covariates, base + both — on the *identical* selection folds those gates
replayed: same origins, same cutoffs, so the comparison is leakage-safe
and apples-to-apples. The winner is chosen deterministically — best mean
fold score, ties broken by fewest enrichments, then by fixed candidate
order — so golden artifacts stay stable. The full comparison is recorded
as an ``enrichment_adjudication`` evidence record: the artifact proves
the choice rather than asserting it.

The combined candidate is the simplest defensible composition of the two
admitted mechanisms: the covariate linear forecast plus the additive
event effect measured from detrended history, both fitted per fold under
that fold's cutoff. A context-only rung replays the exact estimator and shape
that cleared its independent gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from typing import Any, Callable

from .context import ContextEvent
from .context_eval import CONTEXT_MODEL_NAME, ContextAssessment, eligible_events, event_flags
from .context_model import (
    episode_residual_adjusted, event_adjusted, event_effect,
    residual_event_adjusted, rolling_residuals,
)
from .covariates import CovariateAssessment, CovariateDataset, covariate_forecast
from .evaluation import Evaluation, error_score, interval_bounds, quantile
from .models import MODELS, predict

COVARIATE_MODEL_NAME = "covariate_linear"
COMBINED_MODEL_NAME = "combined_enrichment"


@dataclass
class Candidate:
    """One rung of the ladder: a named model with its per-fold record."""

    name: str
    model: str | None
    enrichments: list[str]
    fold_scores: list[float] | None = None
    mean_score: float | None = None
    excluded_reason: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "enrichments": self.enrichments,
            "fold_scores": self.fold_scores,
            "mean_score": self.mean_score,
            "excluded_reason": self.excluded_reason,
        }


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Deterministic ranking: best mean fold score, then fewest enrichments,
    then the fixed candidate order in which the ladder was assembled."""
    order = {id(candidate): index for index, candidate in enumerate(candidates)}
    scored = [candidate for candidate in candidates if candidate.mean_score is not None]
    return sorted(
        scored,
        key=lambda candidate: (
            candidate.mean_score, len(candidate.enrichments), order[id(candidate)],
        ),
    )


@dataclass
class AdjudicationResult:
    considered: bool
    winner: str
    selected_model: str | None
    reason: str
    tie_break: str | None = None
    fold_cutoffs: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    points: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    #: The winner's residuals indexed by lead time. Carried explicitly so
    #: the pipeline can replace the base model's per-lead set when the
    #: winner replaces its points; leaving the base model's set in place
    #: published an interval belonging to a model that lost.
    residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
    coverage: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "winner": self.winner,
            "selected_model": self.selected_model,
            "reason": self.reason,
            "tie_break": self.tie_break,
            "fold_cutoffs": self.fold_cutoffs,
            "candidates": [candidate.to_public_dict() for candidate in self.candidates],
            "measured_coverage": self.coverage,
        }


def adjudicate_enrichments(
    values: list[float],
    timestamps: list[datetime],
    future_timestamps: list[datetime],
    events: list[ContextEvent],
    dataset: CovariateDataset,
    context_assessment: ContextAssessment | None,
    covariate_assessment: CovariateAssessment | None,
    series: str,
    horizon: int,
    season: int,
    base: Evaluation | None,
) -> AdjudicationResult:
    context_admitted = bool(context_assessment and context_assessment.admitted)
    covariate_admitted = bool(covariate_assessment and covariate_assessment.admitted)
    base_model = base.selected_model if base else None
    base_candidate = Candidate("base", base_model, [])
    if not (context_admitted or covariate_admitted):
        return AdjudicationResult(
            True, "base", base_model,
            "no enrichment cleared its independent ablation gate; the base model is retained",
            candidates=[base_candidate],
        )
    if base is None or base_model not in MODELS:
        # Unreachable when a challenger was admitted (both gates require a
        # supported built-in base), but never let the ladder guess.
        return AdjudicationResult(
            True, "base", base_model,
            "the base model cannot be replayed on selection folds; adjudication is unavailable",
            candidates=[base_candidate],
        )

    # The exact grid both admission gates replayed. Admission requires at
    # least four rolling origins, so an admitted challenger guarantees the
    # separated selection / calibration / test partitioning below exists.
    minimum_train = max(2 * season, 2 * horizon, 8)
    origins = list(range(minimum_train, len(values) - horizon + 1, horizon))
    selection = origins[:-2]
    calibration_origin, test_origin = origins[-2], origins[-1]

    eligible, _ = eligible_events(events or [], series)
    retained = covariate_assessment.retained if covariate_admitted else []

    def base_prediction(origin: int, cutoff: datetime) -> list[float]:
        return predict(base_model, values[:origin], horizon, season)

    def context_prediction(origin: int, cutoff: datetime) -> list[float]:
        assert context_assessment is not None
        historical = event_flags(eligible, timestamps[:origin], cutoff)
        future = event_flags(
            eligible, timestamps[origin:origin + horizon], cutoff)
        shape = context_assessment.effect_shape
        if context_assessment.effect_estimator == "base_model_episode_residual":
            return episode_residual_adjusted(
                values[:origin], horizon, season, historical, future,
                predict(base_model, values[:origin], horizon, season),
                base_model, shape,
            )
        if context_assessment.effect_estimator == "base_model_residual":
            return residual_event_adjusted(
                rolling_residuals(values[:origin], base_model, season),
                horizon, historical, future,
                predict(base_model, values[:origin], horizon, season), shape,
            )
        return event_adjusted(
            values[:origin], horizon, season, historical, future, shape,
        )

    def covariate_prediction(origin: int, cutoff: datetime) -> list[float] | None:
        return covariate_forecast(
            values[:origin], timestamps[:origin], timestamps[origin:origin + horizon],
            dataset, retained, series, cutoff, season,
        )

    def combined_prediction(origin: int, cutoff: datetime) -> list[float] | None:
        points = covariate_prediction(origin, cutoff)
        if points is None:
            return None
        effect = event_effect(
            values[:origin],
            event_flags(eligible, timestamps[:origin], cutoff))
        active = event_flags(
            eligible, timestamps[origin:origin + horizon], cutoff)
        return [point + (effect if flag else 0.0)
                for point, flag in zip(points, active)]

    def combined_final_prediction() -> list[float] | None:
        final_cutoff = timestamps[-1]
        covariate_final = covariate_forecast(
            values, timestamps, future_timestamps, dataset, retained, series,
            final_cutoff, season,
        )
        if covariate_final is None:
            return None
        effect = event_effect(
            values, event_flags(eligible, timestamps, final_cutoff))
        active = event_flags(eligible, future_timestamps, final_cutoff)
        return [point + (effect if flag else 0.0)
                for point, flag in zip(covariate_final, active)]

    ladder: list[tuple[Candidate, Callable[[int, datetime], list[float] | None]]] = [
        (base_candidate, base_prediction),
    ]
    if context_admitted:
        ladder.append((Candidate("context", CONTEXT_MODEL_NAME, ["context_events"]), context_prediction))
    if covariate_admitted:
        ladder.append((Candidate("covariates", COVARIATE_MODEL_NAME, ["covariates"]), covariate_prediction))
    if context_admitted and covariate_admitted:
        ladder.append((
            Candidate("combined", COMBINED_MODEL_NAME, ["context_events", "covariates"]),
            combined_prediction,
        ))

    candidates = [candidate for candidate, _ in ladder]
    for candidate, predictor in ladder:
        scores: list[float] | None = []
        for origin in selection:
            cutoff = timestamps[origin - 1]
            try:
                prediction = predictor(origin, cutoff)
            except ValueError:
                prediction = None
            if prediction is None:
                candidate.excluded_reason = (
                    f"candidate failed the selection fold at {cutoff.isoformat()}"
                )
                scores = None
                break
            score = error_score(values[origin:origin + horizon], prediction)
            if score is None:
                # The window has no scale, so no candidate can be scored on
                # it. Dropping it for one arm only would compare the two on
                # different folds, which is the whole point of this ladder.
                candidate.excluded_reason = (
                    f"selection fold at {cutoff.isoformat()} has no scale to "
                    f"score against"
                )
                scores = None
                break
            scores.append(score)
        if scores is not None:
            candidate.fold_scores = scores
            candidate.mean_score = mean(scores)

    winner = base_candidate
    finalised: tuple[list[float], list[float], float | None, list[str]] | None = None
    for candidate in rank_candidates(candidates):
        if candidate.name == "base":
            winner, finalised = candidate, None
            break
        finalised = _finalise(
            candidate, context_assessment, covariate_assessment,
            combined_prediction, combined_final_prediction,
            values, timestamps, future_timestamps,
            dataset, retained, eligible, series, horizon, season,
            calibration_origin, test_origin,
        )
        if finalised is not None:
            winner = candidate
            break
        candidate.excluded_reason = candidate.excluded_reason or (
            "final-horizon prediction is unavailable for this candidate"
        )

    scored = [candidate for candidate in candidates if candidate.mean_score is not None]
    tie_break = None
    tied = [candidate for candidate in scored if candidate.mean_score == winner.mean_score]
    if len(tied) > 1:
        fewest = min(len(candidate.enrichments) for candidate in tied)
        contenders = [candidate for candidate in tied if len(candidate.enrichments) == fewest]
        tie_break = "fewest_enrichments" if len(contenders) == 1 else "fixed_candidate_order"
    reason = (
        f"best mean fold score {winner.mean_score:.6f} across "
        f"{len(selection)} identical selection folds"
        + (f"; tie broken by {tie_break.replace('_', ' ')}" if tie_break else "")
    )

    result = AdjudicationResult(
        True, winner.name, winner.model, reason,
        tie_break=tie_break,
        fold_cutoffs=[timestamps[origin - 1].isoformat() for origin in selection],
        candidates=candidates,
    )
    if finalised is not None:
        (result.points, result.residuals, result.residuals_by_lead,
         result.coverage, result.warnings) = finalised
    return result


def _finalise(
    candidate: Candidate,
    context_assessment: ContextAssessment | None,
    covariate_assessment: CovariateAssessment | None,
    combined_prediction: Callable[[int, datetime], list[float] | None],
    combined_final_prediction: Callable[[], list[float] | None],
    values: list[float],
    timestamps: list[datetime],
    future_timestamps: list[datetime],
    dataset: CovariateDataset,
    retained: list[str],
    eligible: list[ContextEvent],
    series: str,
    horizon: int,
    season: int,
    calibration_origin: int,
    test_origin: int,
) -> tuple[list[float], list[float], dict[int, list[float]], float | None, list[str]] | None:
    """The winner's final forecast, calibration residuals (pooled and by
    lead time), and measured test coverage. Single-enrichment winners reuse
    their assessment's already-final outputs; the combined winner is
    computed here, calibrated and measured on the same held-out folds the
    assessors use.

    The per-lead set travels with the pooled one so a caller cannot install
    the winner's points beside the loser's interval.
    """
    if candidate.name == "context" and context_assessment is not None:
        return (
            context_assessment.points, context_assessment.residuals,
            dict(context_assessment.residuals_by_lead),
            context_assessment.coverage, list(context_assessment.warnings),
        )
    if candidate.name == "covariates" and covariate_assessment is not None:
        return (
            covariate_assessment.points, covariate_assessment.residuals,
            dict(covariate_assessment.residuals_by_lead),
            covariate_assessment.coverage, list(covariate_assessment.warnings),
        )
    if candidate.name != "combined":
        return None
    try:
        calibration = combined_prediction(calibration_origin, timestamps[calibration_origin - 1])
        if calibration is None:
            return None
        residuals = [
            actual - predicted
            for actual, predicted in zip(
                values[calibration_origin:calibration_origin + horizon], calibration,
            )
        ]
        residual_quantiles = {p: quantile(residuals, p) for p in (0.1, 0.5, 0.9)}
        coverage = None
        test = combined_prediction(test_origin, timestamps[test_origin - 1])
        if test is not None:
            covered = []
            for step, (actual, prediction) in enumerate(
                zip(values[test_origin:test_origin + horizon], test), 1,
            ):
                low, _, high = interval_bounds(prediction, residual_quantiles, step)
                covered.append(1.0 if low <= actual <= high else 0.0)
            coverage = mean(covered)
        points = combined_final_prediction()
        if points is None:
            return None
    except ValueError:
        return None
    warnings: list[str] = []
    if coverage is not None and coverage < 0.7:
        warnings.append(f"Final-test 80% interval coverage was {coverage:.1%}, below 70%.")
    # One calibration fold walked in lead order: index i is lead i+1.
    residuals_by_lead = {
        step: [residual] for step, residual in enumerate(residuals, 1)
    }
    return points, residuals, residuals_by_lead, coverage, warnings
