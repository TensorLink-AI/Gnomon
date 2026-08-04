"""Named stages of the forecast pipeline.

``runtime.forecast()`` is an orchestrator over these stages; each stage has
explicit inputs and outputs so later phases can rebind them (snapshot-backed
loading, plan execution) without touching the numerics. This module is a
behaviour-preserving extraction — the golden-artifact tests pin its output
byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .adjudication import adjudicate_enrichments
from .context import ContextEvent
from .context_eval import CONTEXT_MODEL_NAME, ContextAssessment, assess_context
from .contracts import GnomonError, DataSchema, Evidence, SupportReason
from .covariates import CovariateAssessment, CovariateDataset, assess_covariates
from .data import Observation, load_observations
from .evaluation import (
    DEFAULT_TARGET_COVERAGE,
    Evaluation,
    conformal_quantile_spreads,
    conformal_spreads,
    evaluate,
    interval_from_spread,
    quantiles_from_spread,
)
from .models import MODELS, predict
from .multivariate import MULTIVARIATE_MODEL_NAME
from .repair import RepairLog, repair_observations
from .temporal import detect_season, next_timestamp, validate_and_group
from .temporal_store import InMemoryTemporalStore, Snapshot, TemporalStore

STORE_SCHEME = "store:"


@dataclass(frozen=True)
class LoadedDataset:
    """Output of the load stage: validated observations grouped by series,
    plus the snapshot every later stage must read through."""

    source_fingerprint: str
    columns: list[str]
    groups: dict[str, list[Observation]]
    frequency: str
    timezone: str | None
    schema: DataSchema
    snapshot: Snapshot
    variable: str


@dataclass
class SeriesState:
    """Working state for one series as it moves through the stages."""

    name: str
    values: list[float]
    timestamps: list[datetime]
    future_timestamps: list[datetime]
    season: int
    assessment: Evaluation | None = None
    selected_model: str | None = None
    points: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    # Residuals indexed by lead time, when the producing stage measured
    # them that way; empty means intervals fall back to the pooled spread.
    residuals_by_lead: dict[int, list[float]] = field(default_factory=dict)
    #: Which model the residual sets above describe. Points and intervals
    #: must come from the same model, and every stage that replaces
    #: `points` must replace the residuals with it — a stage that changed
    #: one and not the other published a interval belonging to a model that
    #: was not selected. `assert_residual_provenance` enforces it.
    residual_source: str | None = None
    coverage: float | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    context_public: dict[str, object] | None = None
    covariate_public: dict[str, object] | None = None
    context_assessment: ContextAssessment | None = None
    covariate_assessment: CovariateAssessment | None = None
    #: The future-context lane's public record, set only when
    #: `context.future_events` is on and a namespaced event applied to this
    #: series (gnomon.future_context).
    future_context_public: dict[str, object] | None = None
    adjudication_public: dict[str, object] | None = None
    conditional_forecasts: list[dict[str, object]] = field(default_factory=list)
    #: Typed, correct-but-surprising facts for the support assessment.
    #: Never affects support status — see SupportAssessment.disclosures.
    disclosures: list[SupportReason] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


def load_stage(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    series_column: str | None,
    frequency: str | None,
    as_of: datetime | None = None,
    store_path: str | None = None,
    repair: str = "off",
    repair_log: "RepairLog | None" = None,
) -> LoadedDataset:
    """Resolve the input to a snapshot, then materialise the observations
    that are visible at ``as_of``. ``store:<dataset>`` inputs read from the
    persistent bitemporal store; plain files are wrapped in an ephemeral
    store with ``known_time = valid_time`` so the snapshot guarantee is
    uniform across both."""
    variable = target_column
    if input_path.startswith(STORE_SCHEME):
        dataset = input_path[len(STORE_SCHEME):]
        snapshot = TemporalStore(store_path).snapshot(dataset, as_of)
        source_fingerprint = snapshot.source_ref
        columns = [time_column, target_column] + ([series_column] if series_column else [])
    else:
        log = repair_log if repair_log is not None else RepairLog()
        raw_observations, source_fingerprint, columns = load_observations(
            input_path, time_column, target_column, series_column,
            repair=repair, repair_log=log,
        )
        raw_observations = repair_observations(raw_observations, frequency, repair, log)
        _record_reordering(raw_observations, log)
        store, _ = InMemoryTemporalStore.from_plain_observations(
            raw_observations, variable, source_fingerprint,
        )
        snapshot = store.snapshot(as_of)
    observations = [
        Observation(item.valid_time, item.value, entity)
        for entity in snapshot.entities()
        for item in snapshot.series(entity, variable)
    ]
    if not observations:
        raise GnomonError(
            "EMPTY_SNAPSHOT",
            "No observations are known at or before the requested as_of instant.",
            {"as_of": as_of.isoformat() if as_of else "latest"},
        )
    groups, resolved_frequency, zone = validate_and_group(observations, frequency)
    schema = DataSchema(time_column, target_column, series_column, resolved_frequency, zone)
    return LoadedDataset(
        source_fingerprint, columns, groups, resolved_frequency, zone, schema,
        snapshot, variable,
    )


def load_stage_multi(
    input_path: str,
    *,
    time_column: str,
    target_columns: list[str],
    frequency: str | None,
    as_of: datetime | None = None,
    repair: str = "off",
) -> tuple[dict[str, "LoadedDataset | GnomonError"], dict[str, RepairLog], str, list[str]]:
    """One shared file read, one :class:`LoadedDataset` per target column.

    Each target's parse runs through the same code as a single-target
    ``load_stage``, so a batched channel and a sequential run of that
    channel see identical observations. A target whose column cannot be
    loaded maps to its ``GnomonError`` instead of a dataset — the caller
    turns that into a per-channel abstention rather than a failed run.
    Series names are the target column names, which is what keeps
    evidence and result identifiers distinct in the combined artifact.
    """
    from .data import observations_from_rows, read_input_rows

    if input_path.startswith(STORE_SCHEME):
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "Multi-target batching reads several columns of one wide file; "
            "a store dataset holds a single variable. Forecast store inputs "
            "one variable at a time.",
            {"input": input_path},
        )
    shared_log = RepairLog()
    rows, columns, source_fingerprint = read_input_rows(
        input_path, time_column, target_columns[0],
        repair=repair, repair_log=shared_log,
    )
    datasets: dict[str, LoadedDataset | GnomonError] = {}
    logs: dict[str, RepairLog] = {}
    for target in target_columns:
        log = shared_log.clone()
        logs[target] = log
        try:
            raw_observations = observations_from_rows(
                rows, columns, time_column, target, None,
                repair=repair, repair_log=log, default_series=target,
            )
            raw_observations = repair_observations(raw_observations, frequency, repair, log)
            _record_reordering(raw_observations, log)
            store, _ = InMemoryTemporalStore.from_plain_observations(
                raw_observations, target, source_fingerprint,
            )
            snapshot = store.snapshot(as_of)
            observations = [
                Observation(item.valid_time, item.value, entity)
                for entity in snapshot.entities()
                for item in snapshot.series(entity, target)
            ]
            if not observations:
                raise GnomonError(
                    "EMPTY_SNAPSHOT",
                    "No observations are known at or before the requested as_of instant.",
                    {"as_of": as_of.isoformat() if as_of else "latest"},
                )
            groups, resolved_frequency, zone = validate_and_group(observations, frequency)
            schema = DataSchema(time_column, target, None, resolved_frequency, zone)
            datasets[target] = LoadedDataset(
                source_fingerprint, columns, groups, resolved_frequency, zone,
                schema, snapshot, target,
            )
        except GnomonError as error:
            datasets[target] = error
    return datasets, logs, source_fingerprint, columns


def _record_reordering(observations: list[Observation], log: "RepairLog") -> None:
    """Note when the input arrived out of chronological order.

    Sorting is correct and happens further down, in the snapshot — which is
    also why it was invisible: by the time any check ran, the rows were
    already ordered, so a genuinely unsorted export never raised a
    data-quality signal. This runs while the file's own order is still
    observable.
    """
    from collections import defaultdict

    by_series: dict[str, list] = defaultdict(list)
    for item in observations:
        by_series[item.series].append(item.timestamp)
    for name, stamps in sorted(by_series.items()):
        ordered = sorted(stamps)
        if stamps == ordered:
            continue
        moved = sum(1 for left, right in zip(stamps, ordered) if left != right)
        log.record(
            "timestamps_reordered",
            "Rows arrived out of chronological order and were sorted. The "
            "sort is correct; it is recorded because an unsorted export is "
            "usually a symptom worth knowing about.",
            series=name, count=moved,
        )


def horizon_stage(
    name: str,
    items: list[Observation],
    *,
    horizon: int,
    frequency: str,
    seasonal_period: int | None,
) -> SeriesState:
    """Fix the future timestamp grid and the seasonal period for one series."""
    values = [item.value for item in items]
    timestamps = [item.timestamp for item in items]
    future_timestamps: list[datetime] = []
    timestamp = timestamps[-1]
    for _ in range(horizon):
        timestamp = next_timestamp(timestamp, frequency)
        future_timestamps.append(timestamp)
    detected_season, _, _ = detect_season(values, frequency)
    season = seasonal_period or detected_season
    return SeriesState(name, values, timestamps, future_timestamps, season)


def evaluate_stage(
    state: SeriesState,
    *,
    horizon: int,
    minimum_baseline_improvement: float,
    frequency: str,
    config: Any,
    strict_abstention: bool,
    snapshot: Snapshot | None = None,
    variable: str | None = None,
    extra_candidates: dict[str, Any] | None = None,
) -> None:
    """Separated rolling evaluation: selection folds, calibration fold, test
    fold. With a snapshot, every fold trains on the series *as known at*
    that fold's cutoff instant — identical to prefix slices for
    single-vintage data, honest under revisions."""
    train_at = None
    if snapshot is not None and variable is not None:
        entity, timestamps = state.name, state.timestamps

        def train_at(origin: int) -> list[float]:
            cutoff = timestamps[origin - 1]
            return [
                item.value
                for item in snapshot.series(entity, variable, cutoff=cutoff)
                if item.valid_time <= cutoff
            ]

    # `models.tsfm.candidates` restricts which foundation models may
    # compete; an empty list is the unconfigured default (all eligible).
    # This key was parsed and documented but never passed, so listing
    # candidates in gnomon.yaml silently did nothing.
    tsfm_candidates = list(getattr(getattr(config, "models", None),
                                   "tsfm_candidates", None) or []) or None
    assessment = evaluate(
        state.values, horizon, state.season, minimum_baseline_improvement,
        frequency=frequency,
        tsfm_names=tsfm_candidates,
        config=config,
        strict_abstention=strict_abstention,
        train_at=train_at,
        extra_candidates=extra_candidates,
    )
    state.assessment = assessment
    state.selected_model = assessment.selected_model
    state.coverage = assessment.coverage
    state.warnings = list(assessment.warnings)
    state.notes = list(assessment.notes)
    state.residuals = assessment.residuals
    state.residuals_by_lead = dict(assessment.residuals_by_lead)
    state.residual_source = assessment.selected_model


def predict_stage(
    state: SeriesState,
    *,
    horizon: int,
    frequency: str,
    selection_strategy: str,
    extra_candidates: dict[str, Any] | None = None,
) -> None:
    """Produce the final point forecast from the selected model, with the
    ensemble path and the TSFM sandbox/in-process fallback chain."""
    assessment = state.assessment
    if not (assessment and assessment.supported and assessment.selected_model):
        return
    values, season = state.values, state.season
    # Intervals for the ensemble come from the ensemble's own residuals over
    # the selection and calibration folds, computed in `evaluate`. The previous
    # version pooled residuals from a trailing window of the series, which
    # overlaps the calibration and report-only test folds — so the published
    # interval width for every ensemble result was calibrated on the partition
    # the design says nothing may be chosen on (docs/concepts.md, "Temporal
    # evaluation": the final fold reports, and never changes a choice).
    wants_ensemble = assessment.selected_model == "ensemble" or selection_strategy == "ensemble"
    ensemble_residuals, ensemble_by_lead = (
        (assessment.residuals, assessment.residuals_by_lead)
        if assessment.selected_model == "ensemble"
        else (assessment.ensemble_residuals, assessment.ensemble_residuals_by_lead)
    )
    if wants_ensemble and not ensemble_residuals:
        # Nothing fold-separated to calibrate the ensemble from, and the
        # selected model's residuals describe a different forecast. Publishing
        # the ensemble point with someone else's interval is the failure this
        # fix exists to remove, so the ensemble is declined instead.
        wants_ensemble = False
        state.warnings.append(
            "Ensemble was requested but produced no fold-separated residuals "
            f"of its own; reporting {assessment.selected_model} instead, whose "
            "intervals are calibrated."
        )
    if wants_ensemble:
        from .ensemble import compute_ensemble_forecast
        forecasts = {}
        for name in MODELS:
            try:
                forecasts[name] = predict(name, values, horizon, season)
            except ValueError:
                pass
        state.points = compute_ensemble_forecast(forecasts, assessment.selection_scores,
                                                 strategy="weighted_mean", last_observed=values[-1])
        state.selected_model = "ensemble"
        state.residuals = list(ensemble_residuals)
        state.residuals_by_lead = {step: list(items)
                                   for step, items in ensemble_by_lead.items()}
        state.residual_source = "ensemble"
    elif assessment.selected_model in MODELS:
        state.points = predict(assessment.selected_model, values, horizon, season)
    elif extra_candidates and assessment.selected_model in extra_candidates:
        # A cross-series candidate that won the folds; the final forecast is
        # refit at the end of the observed history, the same way every fold
        # forecast was refit at its own origin.
        state.points = extra_candidates[assessment.selected_model](len(values), horizon)
    else:
        # TSFM selected — use the adapter for the final forecast.
        # Try sandbox first, then in-process.
        from .tsfm import get_tsfm, TSFMError, TSFMUnavailable
        from .tsfm_sandbox import sandbox_tsfm_candidates, sandbox_available_tsfms
        try:
            if assessment.selected_model in sandbox_available_tsfms():
                adapters = sandbox_tsfm_candidates(
                    requested=[assessment.selected_model],
                    frequency=frequency,
                )
            else:
                adapters = []
            if adapters:
                adapter = adapters[0]
            else:
                adapter = get_tsfm(assessment.selected_model)
                if hasattr(adapter, '_frequency'):
                    adapter._frequency = frequency
            state.points = adapter.predict(values, horizon, season)
        except (TSFMError, TSFMUnavailable, Exception) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "TSFM %s failed during final forecast, falling back to %s: %s",
                assessment.selected_model, assessment.strongest_baseline, exc,
            )
            # The same failure the ensemble path above was already fixed to
            # prevent: swapping the point path while keeping the failed
            # model's residuals publishes an interval calibrated on a
            # forecast nobody is being shown. Recalibrate from the fallback
            # model's own fold residuals, or decline the fallback.
            failed = assessment.selected_model
            fallback = assessment.strongest_baseline
            if fallback and assessment.fallback_residuals:
                state.selected_model = fallback
                state.points = predict(fallback, values, horizon, season)
                state.residuals = list(assessment.fallback_residuals)
                state.residuals_by_lead = {
                    step: list(items)
                    for step, items in assessment.fallback_residuals_by_lead.items()
                }
                state.residual_source = fallback
                state.warnings.append(
                    f"{failed} was selected by the backtest but failed at final "
                    f"prediction; reporting {fallback} instead, recalibrated on "
                    "its own fold residuals."
                )
            else:
                # Nothing fold-separated to calibrate the fallback from.
                state.points = []
                state.selected_model = None
                state.warnings.append(
                    f"{failed} was selected by the backtest but failed at final "
                    "prediction, and no fold-separated residuals exist for a "
                    "fallback; no forecast is published."
                )


def conditional_stage(
    state: SeriesState,
    context_events: list[ContextEvent],
    *,
    horizon: int,
) -> None:
    """Answer "what if this event happens" for events the gate cannot admit.

    Events without a verifiable source are excluded from backtesting because
    their `known_at` cannot be shown not to leak — which is right for the
    forecast, and turns a real question into an abstention. The answer goes
    in its own list with its own support status; nothing above it changes.
    """
    from .conditional import assess_conditional

    assessment = state.assessment
    if not (assessment and assessment.supported and state.points):
        return
    # Same centring as the published rows, so a what-if interval is never
    # shifted against the forecast interval it conditions on.
    spreads = conformal_spreads(
        state.residuals_by_lead, len(state.points), state.residuals,
        recentre=not assessment.degraded,
    )
    forecasts, excluded = assess_conditional(
        state.values, state.timestamps, state.future_timestamps,
        context_events, state.name, horizon, state.season, spreads,
        state.points,
    )
    state.conditional_forecasts = [item.to_public_dict() for item in forecasts]
    if forecasts or excluded:
        state.evidence.append(Evidence(
            f"conditional_forecasts:{state.name}", "conditional_forecasts",
            state.name,
            {
                "produced": len(forecasts),
                "declined": excluded,
                "basis": "effect measured from event-active periods in the "
                         "observed history; never from the event description",
            },
        ))


def multivariate_stage(
    state: SeriesState,
    *,
    eligible: bool,
    minimum_baseline_improvement: float,
    ineligibility_reason: str | None = None,
    strongest_correlation: float | None = None,
    series_count: int = 0,
) -> None:
    """Record what the cross-series candidate was and how it was decided.

    The VAR does not override anything here. It is entered in `evaluate` as a
    candidate on the same rolling folds as every other model and admitted only
    by the same margin rule, so by this point selection has already spoken.
    Previously this stage overwrote the forecast outright — no fold
    comparison, no baseline, no evidence record — which made it the one path
    that could publish a number the admission philosophy never examined.
    """
    assessment = state.assessment
    if assessment is None:
        return
    score = assessment.selection_scores.get(MULTIVARIATE_MODEL_NAME)
    baseline = assessment.strongest_baseline
    baseline_score = assessment.selection_scores.get(baseline) if baseline else None
    checks: list[dict[str, Any]] = [{
        "code": "series_eligible_for_var",
        "passed": eligible,
        "measured": series_count,
        **({"detail": ineligibility_reason} if ineligibility_reason else {}),
    }]
    if eligible:
        checks.append({
            "code": "var_completed_every_fold",
            "passed": score is not None,
            "measured": score,
        })
        if score is not None and baseline_score:
            margin = baseline_score * (1 - minimum_baseline_improvement)
            checks.append({
                "code": "var_beats_strongest_baseline_by_margin",
                "passed": score <= margin,
                "measured": round(score, 6),
                "threshold": round(margin, 6),
            })
            # Clearing the baseline is not the same as winning: another
            # candidate can clear it by more. Both are recorded so a
            # rejection histogram distinguishes "no cross-series signal"
            # from "cross-series signal, but a univariate model did better".
            checks.append({
                "code": "var_is_the_best_candidate",
                "passed": state.selected_model == MULTIVARIATE_MODEL_NAME,
                "measured": state.selected_model,
            })
    state.evidence.append(Evidence(
        f"multivariate_gate:{state.name}", "multivariate_gate", state.name,
        {
            "series_in_frame": series_count,
            "strongest_correlation": strongest_correlation,
            "admitted": state.selected_model == MULTIVARIATE_MODEL_NAME,
            "checks": checks,
            "decided_by": next(
                (check["code"] for check in checks if not check["passed"]), None,
            ),
        },
    ))


def _record_enrichment_counterfactual(state: SeriesState) -> None:
    """Persist the history-only point path an enrichment is about to replace.

    The tracking ledger's realised-lift score needs error(base) −
    error(published) when actuals arrive, and until now the base path was
    computed during adjudication and discarded. Emitted only when an
    enrichment is actually admitted, so an enrichment-free artifact is
    byte-identical; same precedent as `future_context_applied`'s
    history-only counterfactual.
    """
    if not state.points:
        return
    state.evidence.append(Evidence(
        f"enrichment_counterfactual:{state.name}", "enrichment_counterfactual",
        state.name,
        {
            "base_model": state.selected_model,
            "points": [float(value) for value in state.points],
            "timestamps": [moment.isoformat()
                           for moment in state.future_timestamps],
            "basis": "the history-only forecast as it stood before the "
                     "admitted enrichment replaced it; tracking scores "
                     "realised lift against it when actuals arrive",
        },
    ))


def context_stage(
    state: SeriesState,
    context_events: list[ContextEvent],
    *,
    horizon: int,
    minimum_baseline_improvement: float,
    apply: bool = True,
) -> None:
    """Leakage-safe context-event ablation; admitted events replace the
    forecast, unless a later adjudication stage owns that choice."""
    context = assess_context(
        state.values, state.timestamps, state.future_timestamps, context_events,
        state.name, horizon, state.season, minimum_baseline_improvement,
        state.assessment,
    )
    state.context_assessment = context
    state.context_public = context.to_public_dict()
    state.evidence.append(Evidence(
        f"context_ablation:{state.name}", "context_ablation", state.name,
        state.context_public,
    ))
    # The gate's own decisions, countable: how many events were supplied,
    # how many survived eligibility, and which condition decided the
    # outcome. Admission rate cannot be measured across a corpus from
    # prose reasons alone.
    state.evidence.append(Evidence(
        f"context_gate:{state.name}", "context_gate", state.name,
        {
            "events_supplied": len(context_events),
            "events_eligible": len(context.events_used),
            "events_excluded": len(context.events_excluded),
            "admitted": context.admitted,
            "checks": context.gate_checks,
            "decided_by": next(
                (check["code"] for check in context.gate_checks
                 if not check["passed"]), None,
            ),
        },
    ))
    if context.admitted and apply:
        _record_enrichment_counterfactual(state)
        state.selected_model = CONTEXT_MODEL_NAME
        state.points = context.points
        state.residuals = context.residuals
        state.residuals_by_lead = dict(context.residuals_by_lead)
        state.residual_source = CONTEXT_MODEL_NAME
        state.coverage = context.coverage
        state.warnings = list(context.warnings)


def covariate_stage(
    state: SeriesState,
    covariates: CovariateDataset,
    *,
    horizon: int,
    minimum_baseline_improvement: float,
    apply: bool = True,
) -> None:
    """Leakage-safe covariate ablation; admitted covariates replace the
    forecast, unless a later adjudication stage owns that choice."""
    covariate_assessment = assess_covariates(
        state.values, state.timestamps, state.future_timestamps, covariates,
        state.name, horizon, state.season, minimum_baseline_improvement,
        state.assessment,
    )
    state.covariate_assessment = covariate_assessment
    state.covariate_public = covariate_assessment.to_public_dict()
    state.evidence.append(Evidence(
        f"covariate_ablation:{state.name}", "covariate_ablation", state.name,
        {**state.covariate_public, "source_path": covariates.path,
         "source_fingerprint": covariates.fingerprint},
    ))
    if covariate_assessment.admitted and apply:
        state.selected_model = "covariate_linear"
        state.points = covariate_assessment.points
        state.residuals = covariate_assessment.residuals
        # Previously left as the *base* model's per-lead residuals, which
        # every lead satisfied, so the covariate model's own residuals were
        # never consulted and the published interval belonged to a model
        # that was not selected.
        state.residuals_by_lead = dict(covariate_assessment.residuals_by_lead)
        state.residual_source = "covariate_linear"
        state.coverage = covariate_assessment.coverage
        state.warnings = list(covariate_assessment.warnings)


def adjudicate_enrichments_stage(
    state: SeriesState,
    context_events: list[ContextEvent],
    covariates: CovariateDataset,
    *,
    horizon: int,
) -> None:
    """Championship ladder over independently-admitted enrichments: the base
    model, context, covariates, and their combination compete on identical
    selection folds; the deterministic winner replaces the forecast and the
    whole comparison is recorded as evidence."""
    result = adjudicate_enrichments(
        state.values, state.timestamps, state.future_timestamps,
        context_events, covariates,
        state.context_assessment, state.covariate_assessment,
        state.name, horizon, state.season, state.assessment,
    )
    state.adjudication_public = result.to_public_dict()
    state.evidence.append(Evidence(
        f"enrichment_adjudication:{state.name}", "enrichment_adjudication",
        state.name,
        {**state.adjudication_public,
         "covariates_fingerprint": covariates.fingerprint},
    ))
    if result.winner != "base":
        _record_enrichment_counterfactual(state)
        state.selected_model = result.selected_model
        state.points = result.points
        state.residuals = result.residuals
        # The winner's per-lead residuals, not the base model's. Leaving the
        # base set in place meant every lead had enough of the *wrong*
        # residuals, so the winner's own were never consulted.
        state.residuals_by_lead = dict(result.residuals_by_lead)
        state.residual_source = result.selected_model
        state.coverage = result.coverage
        state.warnings = list(result.warnings)


def threshold_analysis_stage(
    threshold: float,
    rows: list[dict[str, object]],
    points: list[float],
    residuals: list[float],
    spreads: dict[int, tuple[float, float, float]],
) -> dict[str, object]:
    """Empirical threshold-crossing analysis from the backtest residuals,
    recentred and scaled exactly like the published intervals.

    The published interval at a lead time is now the measured spread at
    that lead time, so the crossing probability is computed against the
    same scaling: pooled residuals rescaled to each lead's half-width
    rather than stretched by sqrt(step).
    """
    probabilities: list[float] = []
    pooled_half = (max(spreads[1][0], spreads[1][2])
                   if spreads and 1 in spreads else 0.0)
    for step, point in enumerate(points, 1):
        low_offset, centre_shift, high_offset = spreads[step]
        lead_half = max(low_offset, high_offset)
        # Scale the empirical residual cloud to this lead's measured
        # spread; with one lead's worth of residuals the ratio is 1.
        scale = (lead_half / pooled_half) if pooled_half > 1e-12 else 1.0
        above = sum(
            1 for residual in residuals
            if point + centre_shift + (residual - centre_shift) * scale > threshold
        )
        probabilities.append(round(above / len(residuals), 4))

    def first_timestamp(condition) -> str | None:
        for row in rows:
            if condition(row):
                return str(row["timestamp"])
        return None

    return {
        "value": threshold,
        "probability_above": probabilities,
        "first_timestamp_point_above": first_timestamp(lambda row: row["point"] > threshold),
        "first_timestamp_interval_above": first_timestamp(lambda row: row["q90"] > threshold),
        "first_timestamp_point_below": first_timestamp(lambda row: row["point"] < threshold),
        "first_timestamp_interval_below": first_timestamp(lambda row: row["q10"] < threshold),
        "basis": "empirical probabilities from backtest residuals scaled to the per-lead-time conformal spread",
    }


def _constraint_stage(
    state: SeriesState,
    context_events: list[ContextEvent],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list["Claim"]]:
    """Project the forecast onto bounds the caller says the domain has.

    Rejections are recorded as loudly as applications: a bound the training
    window already breaches is describing a different quantity, and silently
    dropping it would leave the caller believing it was enforced.

    The applied claims are returned so a later stage that rewrites rows
    (an admitted override window) can be projected onto the same feasible
    region: a caller's domain bound outranks every later influence.
    """
    from .constraints import apply_claims, collect_claims

    claims, rejected = collect_claims(
        context_events, state.name, state.values, state.timestamps,
    )
    if not claims and not rejected:
        return rows, claims
    projected, applications = apply_claims(rows, claims)
    if rows and claims:
        sample = datetime.fromisoformat(str(rows[0]["timestamp"]))
        if any(claim.needs_alignment(sample) for claim in claims):
            state.disclosures.append(SupportReason(
                "context_timezone_aligned",
                "Context events carry an explicit timezone offset and this "
                "dataset's timestamps do not (or the reverse). The event "
                "windows were matched on wall-clock time, which is the only "
                "reading available without knowing the dataset's zone. Add an "
                "offset to the dataset's timestamps to remove the assumption.",
            ))
    state.evidence.append(Evidence(
        f"constraint_applied:{state.name}", "constraint_applied", state.name,
        {
            "claims_supplied": len(claims) + len(rejected),
            "claims_applied": len(claims),
            "rejected": rejected,
            "clamps": applications,
            "basis": "bounds are projected onto the emitted quantiles after "
                     "interval construction; the point path is never adjusted "
                     "to satisfy a claim",
        },
    ))
    return projected, claims


def _reassert_claims_stage(
    state: SeriesState,
    claims: list["Claim"],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-project rewritten rows onto the caller's feasibility bounds.

    An admitted override window replaces forecast values with a value
    stated in quoted text, and that value may lie outside a bound the
    caller declared as a domain fact. The caller's bound has the stronger
    provenance — a domain feasibility statement outranks trusted text —
    so it is re-asserted last, and the correction is disclosed rather
    than left as a silent contradiction between two admitted inputs.
    An emitted row never breaches a bound the artifact says was applied.
    """
    from .constraints import apply_claims

    projected, applications = apply_claims(rows, claims)
    if applications:
        state.evidence.append(Evidence(
            f"constraint_reasserted:{state.name}", "constraint_reasserted",
            state.name,
            {
                "clamps": applications,
                "basis": "a later stage moved emitted values outside a "
                         "caller-declared feasibility bound; the bound is "
                         "re-asserted because a domain fact outranks every "
                         "text-trusted influence",
            },
        ))
    return projected


def _future_context_stage(
    state: SeriesState,
    context_events: list[ContextEvent],
    rows: list[dict[str, Any]],
    *,
    allow_future: bool = True,
    allow_structural: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """The textual-verifiability lane for future-dated events.

    Runs only behind `context.future_events: on` (constraint/override
    classes) and `context.structural_events: on` (LLM-classified
    structural effects). Admission and effects are
    `gnomon.future_context`'s; this stage owns the disclosure: the
    gate record, the history-only counterfactual rows preserved beside
    every application, and the public dict the result carries.
    """
    from .future_context import apply_future_events, assess_future_events

    assessment = assess_future_events(
        context_events, state.name, state.values, state.timestamps,
        state.future_timestamps, state.season,
        allow_future=allow_future, allow_structural=allow_structural,
    )
    if not assessment.considered:
        return rows, False
    applications: list[dict[str, Any]] = []
    counterfactual = [dict(row) for row in rows] if assessment.admitted else None
    if assessment.admitted:
        rows, applications = apply_future_events(rows, assessment.admitted)
    state.future_context_public = assessment.to_public_dict()
    state.evidence.append(Evidence(
        f"future_context_gate:{state.name}", "future_context_gate", state.name,
        {**state.future_context_public,
         "events_admitted": len(assessment.admitted),
         "events_rejected": len(assessment.rejected)},
    ))
    if assessment.admitted:
        state.evidence.append(Evidence(
            f"future_context_applied:{state.name}", "future_context_applied",
            state.name,
            {
                "applications": applications,
                "history_only_counterfactual": counterfactual,
                "support": "context_trusted",
                "basis": (
                    "admitted on textual verifiability, not fold ablation: "
                    "constraint bounds project the emitted quantiles onto "
                    "the stated feasible region; override windows take the "
                    "stated value with boundary-widened intervals; "
                    "structural effects derive every quantity from the "
                    "emitted path itself (trend_ceases removes the path's "
                    "own fitted drift). The counterfactual rows are the "
                    "forecast as it stood before this lane touched it."
                ),
            },
        ))
    return rows, bool(assessment.admitted)


def _forecast_disclosures(
    state: SeriesState,
    rows: list[dict[str, object]],
    spreads: dict[int, tuple[float, float, float]],
) -> list[SupportReason]:
    """Correct-but-surprising facts about how these rows were produced.

    Each of these is honest output that a reader will misinterpret unless
    it is stated in band, at the moment it happens. They are disclosures
    rather than warnings: none of them argues against the forecast, so none
    of them may move the support status.
    """
    from .evaluation import MIN_RESIDUALS_PER_LEAD, pooled_fallback_leads

    disclosures: list[SupportReason] = []

    # H2 / S1. `point` is the raw model output; the quantiles are recentred
    # on the median residual, so `point` is not the middle of its interval.
    corrections = [abs(float(row["point_bias_correction"])) for row in rows]
    spans = [float(row["q90"]) - float(row["q10"]) for row in rows]
    widest = max(spans) if spans else 0.0
    if corrections and max(corrections) > 1e-9:
        share = (max(corrections) / widest) if widest > 1e-12 else 0.0
        disclosures.append(SupportReason(
            "point_is_not_the_median",
            f"`point` is the selected model's raw output; every quantile is "
            f"recentred on the median backtest residual, so `q50` differs "
            f"from `point` by up to {max(corrections):.4g} "
            f"({share:.0%} of the 80% interval width). Each row carries the "
            f"gap as `point_bias_correction`.",
        ))

    # H3 / S2. When every lead borrows the pooled spread, the interval does
    # not widen with distance — the 14-step is exactly the 1-step. Not a
    # defect to repair by scaling: whole-horizon pooled residuals already
    # contain multi-step dispersion, so a lead-growth multiplier
    # double-counts it (measured: √lead widening pushed 80%-nominal
    # coverage to 94.9% — results/short-history-guardrail/HYPOTHESIS.md).
    horizon = len(rows)
    borrowed = pooled_fallback_leads(state.residuals_by_lead, horizon)
    assessment = state.assessment
    if assessment is not None and assessment.degraded:
        disclosures.append(SupportReason(
            "point_recentring_suppressed",
            "Quantiles are centred on the model's point path: the usual "
            "recentring on the median backtest residual is suppressed "
            "because on a fold-starved run that median is a location "
            "estimate from too few folds (measured at ~1σ of the series' "
            "step-to-step moves, in an effectively random direction). "
            "`point_bias_correction` is 0 on every row.",
        ))
    if horizon and len(borrowed) == horizon:
        disclosures.append(SupportReason(
            "constant_interval_width",
            f"Interval width is constant across the horizon: no lead time "
            f"has the {MIN_RESIDUALS_PER_LEAD} residuals needed to resolve "
            f"its own spread, so every lead borrows the pooled set. "
            f"Uncertainty is not modelled as growing with distance here. "
            f"More folds — a longer history relative to the horizon — would "
            f"let each lead measure its own.",
        ))
    elif borrowed:
        disclosures.append(SupportReason(
            "partial_pooled_interval_width",
            f"{len(borrowed)} of {horizon} lead times borrow the pooled "
            f"residual spread rather than measuring their own.",
        ))

    # H4. Pooling the selection folds is not split conformal, and the
    # direction of the bias is known: the winner was chosen to minimise
    # error on exactly those folds, so its residuals there are optimistic
    # and the published interval is correspondingly narrow.
    assessment = state.assessment
    if assessment is not None and assessment.residuals_pooled_across_selection:
        disclosures.append(SupportReason(
            "conformal_residuals_pooled_across_selection",
            f"Interval calibration pools residuals from "
            f"{assessment.residual_fold_count} folds, including the "
            f"selection folds the model was chosen on. Those residuals are "
            f"optimistically small, so the interval is narrower than strict "
            f"split conformal would give. Set `evaluation.pool_residuals: "
            f"false` to calibrate on the held-out fold alone.",
        ))

    # M2 / S3. Coverage measured on one fold of `horizon` points carries
    # almost no information; report the count beside the rate.
    if state.coverage is not None:
        disclosures.append(SupportReason(
            "coverage_sample_size",
            f"Measured interval coverage is {state.coverage:.1%} over "
            f"{horizon} point{'s' if horizon != 1 else ''} from a single "
            f"test fold. At this sample size the figure is indicative, not "
            f"a calibration guarantee.",
        ))

    # L6 / S-. Adjacent levels sharing an order statistic is the sample's
    # resolution, not a defect. Promoted from free-text `notes` to a code.
    collapsed = sum(
        1 for row in rows
        if len({row[key] for key in row if _is_quantile(key)})
        < len([key for key in row if _is_quantile(key)])
    )
    if collapsed:
        disclosures.append(SupportReason(
            "quantile_levels_collapsed",
            f"Quantile levels are limited by the residual sample: at "
            f"{collapsed} of {len(rows)} lead times, adjacent levels share an "
            f"order statistic and report identical values. q10/q50/q90 are "
            f"unaffected in meaning.",
        ))
    return disclosures


def _is_quantile(key: object) -> bool:
    text = str(key)
    return text.startswith("q") and text[1:].isdigit()


def assert_residual_provenance(state: SeriesState) -> None:
    """Points and intervals must come from the same model.

    Five paths used to publish a point path from one model beside an
    interval calibrated on another's residuals. Each was a separate bug;
    all of them are this one invariant, checked once, immediately before
    the intervals are built. A violation is a programming error in a
    stage — not a data condition — so it raises rather than warns.
    """
    if not state.points or state.selected_model is None:
        return
    if state.residual_source != state.selected_model:
        raise GnomonError(
            "RESIDUAL_PROVENANCE_MISMATCH",
            f"Intervals for series {state.name!r} would be built from "
            f"{state.residual_source!r} residuals while the published model "
            f"is {state.selected_model!r}.",
            {"series": state.name, "selected_model": state.selected_model,
             "residual_source": state.residual_source},
        )


#: The disclosure attached to every best-effort fallback — verbatim in the
#: warnings, so no rendering of the result can drop it.
NO_RELIABLE_FORECAST = (
    "NO RELIABLE FORECAST: the evaluation protocol could not run on this "
    "history, so no model was selected and no accuracy was measured. These "
    "rows are a last-value fallback with dispersion-scaled intervals, "
    "published only because best-effort mode was explicitly requested. "
    "Treat them as a disclosed placeholder, not a prediction."
)


def best_effort_stage(state: SeriesState) -> tuple[list[dict[str, object]], str]:
    """A disclosed naive fallback for a data-insufficiency abstention.

    Runs only when the caller explicitly asked for it and the evaluation
    abstained. The point path is the most defensible naive answer — the
    last observed value, flat — and the intervals are the dispersion of the
    observed one-step differences scaled by the square root of the lead
    time: a textbook random-walk band, not a calibrated quantile. Nothing
    here was selected by a backtest or measured on a fold, and the result
    says so three ways: support ``best_effort``, the NO RELIABLE FORECAST
    warning verbatim, and a descriptive (never predictive) lineage claim.
    """
    from statistics import NormalDist, stdev

    from .evaluation import QUANTILE_LEVELS, quantile_key

    values = state.values
    horizon = len(state.future_timestamps)
    points = predict("last_value", values, horizon, state.season)
    diffs = [right - left for left, right in zip(values, values[1:])]
    if len(diffs) >= 2:
        sigma = stdev(diffs)
    elif diffs:
        sigma = abs(diffs[0])
    else:
        sigma = 0.0
    normal = NormalDist()
    rows: list[dict[str, object]] = []
    for step, (timestamp, point) in enumerate(
        zip(state.future_timestamps, points), 1,
    ):
        scale = sigma * step ** 0.5
        row: dict[str, object] = {
            "timestamp": timestamp.isoformat(), "point": point,
            "q10": point + normal.inv_cdf(0.1) * scale,
            "q50": point,
            "q90": point + normal.inv_cdf(0.9) * scale,
            "point_bias_correction": 0.0,
        }
        for level in QUANTILE_LEVELS:
            key = quantile_key(level)
            if key not in row:
                row[key] = point + normal.inv_cdf(level) * scale
        rows.append(row)
    state.warnings.append(NO_RELIABLE_FORECAST)
    if sigma == 0.0:
        state.warnings.append(
            "The history shows no dispersion to scale intervals from, so "
            "every quantile equals the point value."
        )
    state.disclosures.append(SupportReason(
        "best_effort_fallback",
        "The forecast rows are a last-value fallback with random-walk "
        "intervals, published under --best-effort after the evaluation "
        "abstained; the abstention's reasons are preserved in the warnings.",
    ))
    return rows, "best_effort"


def interval_stage(
    state: SeriesState,
    *,
    threshold: float | None,
    context_events: list[ContextEvent] | None = None,
    target_coverage: float = DEFAULT_TARGET_COVERAGE,
    future_events: bool = False,
    structural_events: bool = False,
) -> tuple[list[dict[str, object]], str, dict[str, object] | None]:
    """Residual-quantile intervals, the support status, and threshold analysis.

    ``target_coverage`` is the nominal central coverage the interval
    carries, from ``evaluation.uncertainty.target_coverage``.
    ``future_events`` enables the textual-verifiability lane for
    future-dated context events (`context.future_events`);
    ``structural_events`` additionally admits the LLM-classified
    structural class (`context.structural_events`). A forecast either
    influences reports ``context_trusted`` support instead of any
    fold-backed state.
    """
    assert_residual_provenance(state)
    assessment = state.assessment
    rows: list[dict[str, object]] = []
    threshold_analysis: dict[str, object] | None = None
    support = "unsupported"
    if assessment and assessment.supported and state.points:
        # Fold-starved (degraded) runs centre the quantiles on the point
        # path instead of the median residual — a ≤ 2-fold location
        # estimate measured at ≈ 1σ in a coin-flip direction. That
        # suppression alone restores near-nominal coverage (79% pooled
        # against 80% nominal on the 50-series benchmark); the flat
        # borrowed band is left unscaled because whole-horizon pooled
        # residuals already contain multi-step dispersion. With enough
        # folds the recentring keeps its evidence and nothing changes.
        spreads = conformal_spreads(
            state.residuals_by_lead, len(state.points), state.residuals,
            target_coverage, recentre=not assessment.degraded,
        )
        # The full quantile set comes from the same residuals and the same
        # fit; q10/q50/q90 are the identical order statistics they always
        # were, so the additional levels are strictly additional.
        level_spreads = conformal_quantile_spreads(
            state.residuals_by_lead, len(state.points), state.residuals,
            recentre=not assessment.degraded,
        )
        for step, (timestamp, point) in enumerate(zip(state.future_timestamps, state.points), 1):
            q10, q50, q90 = interval_from_spread(point, spreads[step])
            row = {
                "timestamp": timestamp.isoformat(), "point": point,
                "q10": q10, "q50": q50, "q90": q90,
                # `point` is the model's raw output; every quantile is
                # recentred on the median residual. Publishing the gap makes
                # the relationship checkable instead of leaving `point` to
                # be read as the middle of its own interval.
                "point_bias_correction": q50 - point,
            }
            extra = quantiles_from_spread(point, level_spreads[step])
            row.update({key: value for key, value in extra.items()
                        if key not in row})
            rows.append(row)
        state.disclosures.extend(_forecast_disclosures(state, rows, spreads))
        # Feasibility bounds are projected onto the emitted quantiles here,
        # after the model has said what it believes and before anything reads
        # the rows, so the threshold analysis below sees the same numbers the
        # caller will.
        claims = []
        if context_events:
            rows, claims = _constraint_stage(state, context_events, rows)
        future_influenced = False
        if (future_events or structural_events) and context_events:
            rows, future_influenced = _future_context_stage(
                state, context_events, rows,
                allow_future=future_events,
                allow_structural=structural_events,
            )
            if future_influenced and claims:
                rows = _reassert_claims_stage(state, claims, rows)
        support = "degraded" if assessment.degraded else ("supported_ensemble" if state.selected_model == "ensemble" else ("weakly_supported" if state.warnings else "supported"))
        if future_influenced:
            # Whatever the fold-backed status would have said, this forecast
            # now rests partly on trusted text; the weaker warrant is the
            # honest one, and the fold facts survive in the reasons.
            support = "context_trusted"
        if threshold is not None:
            # The published point path, read back from the rows: a clamp or
            # an admitted override has already moved them, and a crossing
            # probability centred on a point nobody is being shown would
            # contradict the rows beside it. Identical to `state.points`
            # when nothing projected.
            threshold_analysis = threshold_analysis_stage(
                threshold, rows, [float(row["point"]) for row in rows],
                state.residuals, spreads,
            )
    return rows, support, threshold_analysis
