from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import write_artifact
from .context import ContextEvent, event_to_dict
from .contracts import DataSchema, Evidence, ForecastArtifact, ForecastTask, SeriesResult
from .covariates import CovariateDataset
from .versioning import RUNTIME_VERSION
from .ids import SYSTEM_CLOCK, Clock, content_id
from .models import BASELINES, MODELS
from .pipeline import (
    LoadedDataset,
    adjudicate_enrichments_stage,
    conditional_stage,
    context_stage,
    covariate_stage,
    evaluate_stage,
    horizon_stage,
    interval_stage,
    load_stage,
    multivariate_stage,
    predict_stage,
)
from .temporal import FREQUENCY_DESCRIPTIONS, SEASONS, detect_season


def _seasonal_period_label(period_steps: int, frequency: str) -> str:
    """Human label paired with the canonical step count, never replacing it."""
    labels = {
        ("s", 60): "minute", ("min", 60): "hourly",
        ("5min", 288): "daily", ("10min", 144): "daily",
        ("15min", 96): "daily", ("30min", 48): "daily",
        ("h", 24): "daily", ("D", 7): "weekly",
        ("W", 52): "annual", ("MS", 12): "annual",
    }
    return labels.get((frequency, period_steps), f"period-{period_steps}")


def inspect_dataset(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    series_column: str | None = None,
    frequency: str | None = None,
    seasonal_period: int | None = None,
    as_of: datetime | None = None,
    store_path: str | None = None,
    clock: Clock | None = None,
    regrid: str | None = None,
) -> dict[str, object]:
    # Diagnose, don't just reject: try the strict path, then each repair
    # level, and report what the file needs to become forecastable.
    from .contracts import GnomonError
    from .repair import REPAIR_LEVELS, RepairLog
    loaded = None
    repair_level_used = None
    log = RepairLog()
    errors: dict[str, GnomonError] = {}
    for level in REPAIR_LEVELS:
        log = RepairLog()
        try:
            loaded = load_stage(
                input_path, time_column=time_column, target_column=target_column,
                series_column=series_column, frequency=frequency,
                as_of=as_of, store_path=store_path,
                repair=level, repair_log=log, regrid=regrid,
            )
            repair_level_used = level
            break
        except GnomonError as error:
            errors[level] = error
    if loaded is None:
        # The safe-level error is the diagnosis; an aggressive-level failure
        # (e.g. EXCESSIVE_REPAIR) is a consequence of forcing assumptions.
        raise errors.get("safe") or errors["aggressive"]
    data_quality: dict[str, object] = {
        "status": {
            "off": "clean",
            "safe": "repaired_safe",
            "aggressive": "repaired_aggressive",
        }[repair_level_used],
        "repairs": log.summary()["actions"],
        "note": {
            "off": "The file passes strict validation untouched.",
            "safe": "The default repair level (safe) reads this file; every "
                    "normalisation is listed under repairs.",
            "aggressive": "This file needs --repair aggressive; the structural "
                          "fixes it would apply are listed under repairs and "
                          "will be disclosed as warnings on the forecast.",
        }[repair_level_used],
    }
    repair_flag = " --repair aggressive" if repair_level_used == "aggressive" else ""
    # A harness built on knowledge time should remark when every observation
    # postdates now. It is not an error — synthetic and planning data are
    # legitimate — but `status: valid` with no note read as endorsement.
    now = (clock or SYSTEM_CLOCK).now()
    latest = max(
        (item.timestamp for items in loaded.groups.values() for item in items),
        default=None,
    )
    if latest is not None:
        # Most datasets are naive and the clock is aware, so compare wall
        # clocks — the same alignment the context path makes, for the same
        # reason: without it the check never fires on real input.
        from .constraints import _align

        latest, now = _align(latest, now)
        if latest > now:
            earliest = min(
                item.timestamp for items in loaded.groups.values() for item in items
            )
            earliest, _ = _align(earliest, now)
            data_quality["temporal_position"] = (
                "entirely_in_the_future" if earliest > now else "extends_into_the_future"
            )
            data_quality["note"] += (
                f" Every observation is dated after the current instant "
                f"({now.isoformat()}); the series runs to {latest.isoformat()}. "
                f"That is legitimate for synthetic or planning data and "
                f"unusual otherwise."
                if earliest > now else
                f" The series extends past the current instant "
                f"({now.isoformat()}) to {latest.isoformat()}."
            )
    from .multivariate import correlation_report
    return {
        "schema_version": "0.1",
        "status": "valid",
        "input_path": (
            input_path if input_path.startswith("store:")
            else str(Path(input_path).expanduser().resolve())
        ),
        "source_fingerprint": loaded.source_fingerprint,
        "columns": loaded.columns,
        "schema": {
            "time_column": time_column,
            "target_column": target_column,
            "series_column": series_column,
            "frequency": loaded.frequency,
            "timezone": loaded.timezone,
            "missing_policy": "reject",
            "duplicate_policy": "reject",
        },
        "series": [
            {
                "name": name,
                "observations": len(items),
                "start": items[0].timestamp.isoformat(),
                "end": items[-1].timestamp.isoformat(),
                "change": {
                    "final_step": (items[-1].value - items[-2].value
                                   if len(items) > 1 else 0.0),
                    "absolute_final_step": (abs(items[-1].value - items[-2].value)
                                             if len(items) > 1 else 0.0),
                },
                "seasonality": dict(zip(("period", "strength", "source"),
                    (seasonal_period, 1.0, "override") if seasonal_period else detect_season([item.value for item in items], loaded.frequency))),
            }
            for name, items in sorted(loaded.groups.items())
        ],
        "cross_series_correlations": correlation_report(loaded.groups),
        "data_quality": data_quality,
        "suggested_next": (
            f"gnomon forecast {Path(input_path).expanduser().resolve()} "
            f"--time {time_column} --target {target_column}"
            + (f" --series {series_column}" if series_column else "")
            + f" --frequency {loaded.frequency} --horizon <periods>"
            + repair_flag
        ),
    }


def _split_prefix(
    series_name: str, items: list, loaded: Any, reachable: int, *,
    minimum_baseline_improvement: float, config: Any,
    selection_strategy: str, seasonal_period: int | None,
    target_coverage: float,
):
    """Evaluate the supportable horizon prefix for an automatic split.

    The same stages, the same floors, the same guardrails as any run at
    horizon ``reachable`` — nothing is re-graded, only re-requested at
    the horizon the data already supports. Enrichment stages are not
    applied here (disclosed by the ``horizon_split`` reason); the prefix
    is the core evaluated forecast. Returns ``None`` when even the
    prefix abstains, in which case the caller falls back to the plain
    best-effort lane."""
    sub = horizon_stage(
        series_name, items, horizon=reachable, frequency=loaded.frequency,
        seasonal_period=seasonal_period,
    )
    evaluate_stage(
        sub, horizon=reachable,
        minimum_baseline_improvement=minimum_baseline_improvement,
        frequency=loaded.frequency, config=config, strict_abstention=False,
        snapshot=loaded.snapshot, variable=loaded.variable,
    )
    predict_stage(
        sub, horizon=reachable, frequency=loaded.frequency,
        selection_strategy=selection_strategy,
    )
    prefix_rows, prefix_support, _ = interval_stage(
        sub, threshold=None, target_coverage=target_coverage,
    )
    if not prefix_rows or prefix_support == "unsupported":
        return None
    return sub, prefix_rows, prefix_support


def _config_fingerprint(config: Any) -> dict[str, object] | None:
    """The behaviour-relevant subset of the config, for content addressing.

    A missing config and the built-in defaults fingerprint identically, so
    the same task yields the same artifact ID regardless of which interface
    invoked it."""
    from .evaluation import DEFAULT_TARGET_COVERAGE

    if config is None:
        return None
    ensemble = getattr(config, "ensemble", None)
    meta_model = getattr(config, "meta_model", None)
    backends = getattr(config, "backends", None)
    api = getattr(backends, "api", None) if backends else None
    models = getattr(config, "models", None)
    evaluation = getattr(config, "evaluation", None)
    context = getattr(config, "context", None)
    payload: dict[str, object] = {
        "ensemble": asdict(ensemble) if ensemble is not None and is_dataclass(ensemble) and ensemble.enabled else None,
        "meta_model": asdict(meta_model) if meta_model is not None and is_dataclass(meta_model) and meta_model.enabled else None,
        "api_providers": sorted(api.providers) if api is not None and api.enabled else None,
        "tsfm_candidates": sorted(getattr(models, "tsfm_candidates", None) or []) or None,
        # A restricted statistical pool changes which models compete, so it
        # must change the forecast id: without it, a `candidates` run and an
        # open-contest run over the same file collided on one id, and
        # first-write-wins served whichever artifact landed first.
        "statistical_candidates": sorted(
            getattr(models, "statistical_candidates", None) or []) or None,
        # Interval-shaping options change the published numbers without
        # changing any name in the run, so leaving them out let two runs
        # with materially different bands collide on one content-addressed
        # id — the same defect `statistical_candidates` above was added to
        # fix. Only non-default values enter, so existing ids are
        # byte-identical.
        "target_coverage": (
            value if (value := getattr(evaluation, "target_coverage", None))
            is not None and value != DEFAULT_TARGET_COVERAGE else None
        ) if evaluation is not None else None,
        "pool_residuals": (
            False if getattr(evaluation, "pool_residuals", True) is False
            else None
        ) if evaluation is not None else None,
        "min_observations": (
            getattr(evaluation, "min_observations", None)
        ) if evaluation is not None else None,
        # The experimental context lanes rewrite the forecast when they
        # admit an event; off is the default and fingerprints as absent.
        "context": sorted(
            name for name in ("future_events", "structural_events")
            if getattr(context, name, False)
        ) or None if context is not None else None,
    }
    if all(value is None for value in payload.values()):
        return None
    return payload


def _restricted_pool(config: Any) -> list[str] | None:
    """The caller-narrowed candidate pool, or ``None`` when the contest is
    open. Reads the config rather than a threaded flag so a restriction set
    in `gnomon.yaml` is disclosed exactly like one passed as `candidates`."""
    models = getattr(config, "models", None) if config is not None else None
    if models is None:
        return None
    statistical = getattr(models, "statistical_candidates", None)
    if not statistical:
        # `tsfm_candidates` alone is not a restriction: listing TSFMs is how
        # they become available at all, and an empty list is the default.
        return None
    return sorted(set(statistical) | set(getattr(models, "tsfm_candidates", None) or []))


def _restrict_candidates(config: Any, candidates: list[str]):
    """A copy of ``config`` whose candidate pool is the named models.

    Baselines are added back unconditionally: a candidate is selected by
    beating them, so a pool without them has nothing to select against.
    """
    import copy as copy_module

    from .config import load_config
    from .contracts import GnomonError
    from .models import BASELINES, MODELS
    from .tsfm import available_tsfms

    known_tsfms = set(available_tsfms())
    unknown = [
        name for name in candidates
        if name not in MODELS and name not in known_tsfms
    ]
    if unknown:
        raise GnomonError(
            "UNKNOWN_MODEL",
            f"candidates names models that do not exist: "
            f"{', '.join(sorted(unknown))}.",
            {"unknown": sorted(unknown),
             "available": sorted(set(MODELS) | known_tsfms)},
        )
    resolved = copy_module.deepcopy(config) if config is not None else load_config()
    statistical = [
        name for name in candidates if name in MODELS and name not in BASELINES
    ]
    resolved.models.statistical_candidates = statistical or None
    resolved.models.tsfm_candidates = [
        name for name in candidates if name in known_tsfms
    ]
    return resolved


def _series_result(
    series_name: str,
    items: list,
    *,
    loaded: LoadedDataset,
    horizon: int,
    seasonal_period: int | None,
    minimum_baseline_improvement: float,
    config: Any,
    strict_abstention: bool,
    selection_strategy: str,
    multivariate: bool,
    var_frame: Any,
    var_ineligible: str | None,
    context_events: list[ContextEvent] | None,
    covariates: CovariateDataset | None,
    adjudicating: bool,
    threshold: float | None,
    target_coverage: float,
    repair_log: Any,
    future_events: bool = False,
    structural_events: bool = False,
    best_effort: bool = False,
    minimum_support: str = "best_effort",
) -> tuple[SeriesResult, list[Evidence]]:
    """Run one series through the full stage pipeline.

    This is the loop body of :func:`forecast`, extracted behaviour-for-
    behaviour so a multi-target run can execute it once per channel —
    concurrently, since channels are independent — without touching the
    numerics. Returns the series result and its evidence in the exact
    order the single-target loop emitted them.
    """
    state = horizon_stage(
        series_name, items, horizon=horizon, frequency=loaded.frequency,
        seasonal_period=seasonal_period,
    )
    extra_candidates: dict[str, Any] = {}
    if var_frame is not None and series_name in var_frame.names:
        from .multivariate import MULTIVARIATE_MODEL_NAME
        extra_candidates[MULTIVARIATE_MODEL_NAME] = var_frame.predictor(series_name)
    evaluate_stage(
        state, horizon=horizon,
        minimum_baseline_improvement=minimum_baseline_improvement,
        frequency=loaded.frequency, config=config,
        strict_abstention=strict_abstention,
        snapshot=loaded.snapshot, variable=loaded.variable,
        extra_candidates=extra_candidates,
    )
    predict_stage(
        state, horizon=horizon, frequency=loaded.frequency,
        selection_strategy=selection_strategy,
        extra_candidates=extra_candidates,
    )
    if multivariate:
        multivariate_stage(
            state,
            eligible=var_frame is not None,
            minimum_baseline_improvement=minimum_baseline_improvement,
            ineligibility_reason=var_ineligible,
            strongest_correlation=(
                round(var_frame.strongest_correlation, 4) if var_frame else None
            ),
            series_count=len(var_frame.names) if var_frame else len(loaded.groups),
        )
    if context_events:
        context_stage(
            state, context_events, horizon=horizon,
            minimum_baseline_improvement=minimum_baseline_improvement,
            apply=not adjudicating,
        )
    if covariates:
        covariate_stage(
            state, covariates, horizon=horizon,
            minimum_baseline_improvement=minimum_baseline_improvement,
            apply=not adjudicating,
        )
    if adjudicating:
        adjudicate_enrichments_stage(
            state, context_events, covariates, horizon=horizon,
        )
    if context_events:
        # After every stage that can change the point forecast and its
        # calibration: a conditional answer is conditioned on the
        # forecast that was actually selected.
        conditional_stage(state, context_events, horizon=horizon)
    repair_warnings = repair_log.warnings_for(series_name)
    if repair_warnings:
        state.warnings.extend(repair_warnings)
    rows, support, threshold_analysis = interval_stage(
        state, threshold=threshold, context_events=context_events,
        target_coverage=target_coverage,
        future_events=future_events,
        structural_events=structural_events,
    )
    split = None
    split_prefix_tier: str | None = None
    if (support == "unsupported" and not rows and state.values
            and minimum_support == "best_effort"
            and not strict_abstention):
        # Graduated support: the default floor publishes the most defensible
        # answer that exists — here the disclosed naive fallback — instead
        # of abstaining. Nothing about how tiers are earned changed; a
        # higher minimum_support restores the refusal, `strict_abstention`
        # keeps its refusal semantics untouched, and a series with no
        # usable history at all still abstains (the guard above). The
        # legacy `best_effort` flag maps to this same floor.
        from .pipeline import best_effort_stage
        reachable = (state.assessment.max_supportable_horizon
                     if state.assessment is not None else None)
        if reachable and 0 < reachable < horizon:
            # Automatic horizon split: the supportable prefix is evaluated
            # at whatever tier it earns — same stages, same guardrails —
            # and only the unsupportable remainder falls back.
            split = _split_prefix(
                series_name, items, loaded, reachable,
                minimum_baseline_improvement=minimum_baseline_improvement,
                config=config, selection_strategy=selection_strategy,
                seasonal_period=seasonal_period,
                target_coverage=target_coverage,
            )
        else:
            split = None
        if split is not None:
            split_state, split_rows, split_support = split
            fallback_rows, _ = best_effort_stage(state)
            rows = split_rows + fallback_rows[len(split_rows):]
            support = "best_effort"
        else:
            rows, support = best_effort_stage(state)
    assessment = state.assessment
    from .support import assess_forecast_support, disclose_epistemic_deviation
    if split is not None:
        # The split assessment carries both ranges explicitly: the prefix's
        # own reasons and tier, the remainder's fallback status, and the
        # typed `horizon_split` naming which rows sit where and why.
        from .support import achieved_tier as _tier
        prefix_assessment = assess_forecast_support(
            split_support, split_state.warnings, split_state.assessment,
            known_time_assumed=loaded.snapshot.assumed_known_time,
            disclosures=split_state.disclosures,
            measured_coverage=split_state.coverage,
        )
        prefix_tier = _tier(prefix_assessment.status, True)
        split_prefix_tier = prefix_tier
        prefix_end = split_rows[-1]["timestamp"]
        full_end = rows[-1]["timestamp"]
        from .contracts import SupportAssessment as _SupportAssessment
        from .contracts import SupportReason as _SupportReason
        enrichment_note = (
            " Context events and covariates were not applied on the "
            "split path; request the supportable horizon directly to "
            "use them." if (context_events or covariates) else ""
        )
        split_reason = _SupportReason(
            "horizon_split",
            f"Rows 1-{reachable} (through {prefix_end}) are an evaluated "
            f"forecast at tier '{prefix_tier}'; rows {reachable + 1}-"
            f"{horizon} (through {full_end}) are a naive best-effort "
            f"extrapolation — the evaluation cannot support the full "
            f"requested horizon.{enrichment_note}",
        )
        remainder_reason = _SupportReason(
            "no_reliable_forecast",
            f"Rows {reachable + 1}-{horizon} carry no measured accuracy "
            f"and no probability weight: a last-value fallback with "
            f"dispersion-scaled intervals.",
        )
        support_assessment = _SupportAssessment(
            ("conditionally_supported"
             if prefix_assessment.status == "supported"
             else prefix_assessment.status),
            [split_reason] + prefix_assessment.reasons + [remainder_reason],
            prefix_assessment.assumptions,
            {**prefix_assessment.sensitivity,
             "supported_horizon": reachable,
             "requested_horizon": horizon},
            [_SupportReason(
                "reduce_horizon",
                f"Request horizon {reachable} or less for a fully "
                f"evaluated result.",
            )],
            "best_effort",
            prefix_assessment.disclosures + [
                item for item in state.disclosures
                if item not in prefix_assessment.disclosures
            ],
        )
        # The published selection facts are the prefix's — the requested-
        # horizon evaluation selected nothing. Its coverage is deliberately
        # not published as the result's: it measures rows 1..N only, and it
        # stays visible in the sensitivity and the prefix evidence.
        state.selected_model = split_state.selected_model
        state.warnings = split_state.warnings + state.warnings
        state.coverage = None
        assessment = split_state.assessment
    else:
        support_assessment = assess_forecast_support(
            support, state.warnings, assessment,
            known_time_assumed=loaded.snapshot.assumed_known_time,
            disclosures=state.disclosures,
            measured_coverage=state.coverage,
        )
        if support == "best_effort":
            # The headline for a whole-object fallback names the history
            # it extrapolates from; the count belongs in the sensitivity
            # beside the rest of the measured facts.
            support_assessment.sensitivity["observations"] = len(state.values)
    from .contracts import DEFAULT_MINIMUM_BASELINE_IMPROVEMENT, SupportReason
    if minimum_baseline_improvement != DEFAULT_MINIMUM_BASELINE_IMPROVEMENT:
        # A caller-chosen evidence rule is not the documented one. Below the
        # default the mandated-baseline gate is weaker, so the verdict is
        # capped; above it the gate is stricter and the deviation only needs
        # to be visible. (Negative values never reach here — `evaluate`
        # refuses them as INVALID_MINIMUM_IMPROVEMENT.)
        lowered = minimum_baseline_improvement < DEFAULT_MINIMUM_BASELINE_IMPROVEMENT
        support_assessment = disclose_epistemic_deviation(
            support_assessment,
            SupportReason(
                "nonstandard_evaluation",
                f"minimum_baseline_improvement was "
                f"{'lowered' if lowered else 'raised'} to "
                f"{minimum_baseline_improvement} from the default "
                f"{DEFAULT_MINIMUM_BASELINE_IMPROVEMENT}: the mandated-"
                f"baseline gate is {'weaker' if lowered else 'stricter'} "
                f"than documented.",
            ),
            cap=lowered,
        )
    restricted_pool = _restricted_pool(config)
    if restricted_pool is not None:
        support_assessment = disclose_epistemic_deviation(
            support_assessment,
            SupportReason(
                "candidate_pool_restricted",
                f"The candidate pool was restricted by the caller to "
                f"{', '.join(restricted_pool) or 'baselines only'}; the "
                f"mandatory baselines competed regardless, so the selection "
                f"is honest within a narrowed contest.",
            ),
            cap=False,
        )
    from .support import TIER_ORDER, achieved_tier
    tier = achieved_tier(support_assessment.status, bool(rows))
    if tier is not None and TIER_ORDER[tier] < TIER_ORDER[minimum_support]:
        # The publication floor, not the evaluation, refuses this result:
        # the evidence is what it is, the caller asked not to be shown
        # anything below the floor. Typed, with the one-step recovery.
        support_assessment = assess_forecast_support(
            "unsupported", state.warnings, assessment,
            known_time_assumed=loaded.snapshot.assumed_known_time,
            disclosures=state.disclosures,
        )
        support_assessment.reasons.insert(0, SupportReason(
            "below_minimum_support",
            f"The evidence achieved tier {tier!r}, below the requested "
            f"minimum_support {minimum_support!r}; nothing was published. "
            f"The evaluation itself is unchanged — only the publication "
            f"floor refused the result.",
        ))
        support_assessment.recovery_actions.insert(0, SupportReason(
            "lower_minimum_support",
            f"Retry with minimum_support {tier!r} (or lower) to publish "
            f"the result the evidence already supports.",
        ))
        rows, support, threshold_analysis = [], "unsupported", None
    if threshold is not None and rows and support == "best_effort":
        # A requested analysis that cannot run must say so, not vanish:
        # threshold-crossing probabilities need calibrated residuals, which
        # best_effort and horizon-split rows do not have.
        state.notes.append(
            f"threshold {threshold} was requested but no crossing analysis "
            f"is reported: exceedance probabilities require calibrated "
            f"residuals, which best_effort rows (and the fallback range of "
            f"a horizon split) do not have."
        )
    # The unstrippable label: every published row names its tier, uniform
    # on a single-tier forecast, changing at the split point on a split
    # one — one shape, no special cases for consumers.
    if rows:
        if split is not None and split_prefix_tier is not None:
            # Prefix rows carry the tier their own evaluation earned; the
            # remainder is the fallback, whatever the merged status says.
            prefix_length = len(split[1])
            for index, row in enumerate(rows):
                row["tier"] = (split_prefix_tier if index < prefix_length
                               else "best_effort")
        else:
            uniform_tier = achieved_tier(support_assessment.status, True)
            for row in rows:
                row["tier"] = uniform_tier
    from .soft_context import context_outcome as project_context_outcome
    result = SeriesResult(
        series_name, support, state.selected_model, assessment.strongest_baseline,
        assessment.selection_scores, assessment.test_scores, assessment.improvement,
        state.coverage, state.warnings, rows, state.context_public,
        state.covariate_public, threshold_analysis,
        support_assessment.to_dict(),
        notes=state.notes,
        conditional_forecasts=state.conditional_forecasts,
        sensitivity_scenarios=state.sensitivity_scenarios,
        future_context=state.future_context_public,
        temporal_facts={
            "seasonal_period_steps": state.season,
            "seasonal_period_label": _seasonal_period_label(
                state.season, loaded.frequency),
            "frequency": loaded.frequency,
            "source": "computed_from_observations",
        },
        context_outcome=(
            project_context_outcome(
                context_events, series_name,
                context_assessment=state.context_assessment,
                future_context=state.future_context_public,
                conditional_forecasts=state.conditional_forecasts,
                sensitivity_scenarios=state.sensitivity_scenarios,
            ) if context_events else None
        ),
    )
    evidence = list(state.evidence)
    evidence.extend([
        Evidence(f"evaluation:{series_name}", "rolling_evaluation", series_name, {
            "partitioning": "selection folds, then calibration fold, then final test fold",
            "selection_scores": assessment.selection_scores,
            "test_scores": assessment.test_scores,
            # The verifier gates probability-bearing claims on these,
            # so they have to be *in* the calibration record rather
            # than only in the result beside it.
            "measured_interval_coverage": state.coverage,
            "baseline_improvement": assessment.improvement,
            "strongest_baseline": assessment.strongest_baseline,
            "selected_model": state.selected_model,
            "residuals_pooled_across_selection":
                assessment.residuals_pooled_across_selection,
            "residual_fold_count": assessment.residual_fold_count,
        }),
        Evidence(f"support:{series_name}", "support_assessment", series_name, {
            "support": support, "warnings": state.warnings,
            # The per-row tier mix, so the verifier can see from the
            # evidence alone whether sub-supported rows were published.
            "row_tiers": _tier_counts(rows),
            "minimum_support": minimum_support,
        }),
    ])
    if split is not None:
        # The prefix's own evaluation record, distinct from the requested-
        # horizon record above (which documents the abstention): the split
        # is auditable from evidence, not only from the reasons.
        split_state = split[0]
        split_assessment = split_state.assessment
        evidence.append(Evidence(
            f"evaluation:{series_name}:prefix", "rolling_evaluation",
            series_name, {
                "partitioning": (
                    "horizon-split prefix at the supportable horizon; "
                    "selection folds, then calibration fold, then final "
                    "test fold"),
                "horizon": len(split[1]),
                "selection_scores": split_assessment.selection_scores,
                "test_scores": split_assessment.test_scores,
                "measured_interval_coverage": split_state.coverage,
                "baseline_improvement": split_assessment.improvement,
                "strongest_baseline": split_assessment.strongest_baseline,
                "selected_model": split_state.selected_model,
            },
        ))
    return result, evidence


def _tier_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        tier = str(row.get("tier", "unlabelled"))
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def forecast(
    input_path: str,
    *,
    time_column: str,
    target_column: str,
    horizon: int,
    series_column: str | None = None,
    frequency: str | None = None,
    output: str = "gnomon-output",
    minimum_baseline_improvement: float = 0.02,
    context_events: list[ContextEvent] | None = None,
    covariates: CovariateDataset | None = None,
    threshold: float | None = None,
    config: Any = None,
    strict_abstention: bool = False,
    best_effort: bool = False,
    minimum_support: str = "best_effort",
    seasonal_period: int | None = None,
    selection_strategy: str = "best",
    multivariate: bool = False,
    clock: Clock | None = None,
    as_of: datetime | None = None,
    store_path: str | None = None,
    repair: str = "safe",
    regrid: str | None = None,
    candidates: list[str] | None = None,
    input_provenance: str | None = None,
) -> tuple[ForecastArtifact, Path]:
    clock = clock or SYSTEM_CLOCK
    if candidates:
        # `gnomon route`'s output, made actionable. The router answered "which
        # method for this task?" and nothing consumed the answer: forecast
        # had no model parameter at all, so even a confident recommendation
        # could not be acted on. Restricting the pool is advisory in the
        # right way — the named candidates still backtest against the
        # mandatory baselines, which are never removable.
        config = _restrict_candidates(config, candidates)
    from .repair import REPAIR_LEVELS, REPAIR_SAFE, RepairLog
    if repair not in REPAIR_LEVELS:
        from .contracts import GnomonError
        raise GnomonError(
            "INVALID_REPAIR_LEVEL",
            f"repair must be one of {', '.join(REPAIR_LEVELS)}.",
            {"requested": repair, "supported": list(REPAIR_LEVELS)},
        )
    if horizon < 1:
        from .contracts import GnomonError
        raise GnomonError("INVALID_HORIZON", "Horizon must be at least one period.")
    from .support import PUBLICATION_TIERS
    if minimum_support not in PUBLICATION_TIERS:
        from .contracts import GnomonError
        raise GnomonError(
            "INVALID_ARGUMENTS",
            f"minimum_support must be one of {', '.join(PUBLICATION_TIERS)}.",
            {"requested": minimum_support,
             "supported": list(PUBLICATION_TIERS)},
        )
    if selection_strategy == "ensemble":
        # Asking for the ensemble has to enter it in the evaluation, not just
        # swap the final forecast. Otherwise it is never scored on the folds
        # and has no fold-separated residuals to build an interval from.
        import copy as _copy
        from .config import load_config as _load_config
        config = _copy.deepcopy(config) if config is not None else _load_config()
        config.ensemble.enabled = True
    # When both enrichment kinds are supplied, neither ablation stage applies
    # its own winner; the adjudication ladder owns the choice.
    adjudicating = bool(context_events) and covariates is not None
    # `evaluation.uncertainty.target_coverage`, previously parsed and never
    # read: every run published an 80% interval whatever the config said.
    from .evaluation import DEFAULT_TARGET_COVERAGE
    target_coverage = DEFAULT_TARGET_COVERAGE
    if config is not None and getattr(config, "evaluation", None) is not None:
        target_coverage = float(
            getattr(config.evaluation, "target_coverage", DEFAULT_TARGET_COVERAGE)
        )
    if covariates is not None:
        # Bind the run's boundary to the covariate snapshot before anything
        # reads it, so leakage control is a property of the object rather
        # than of every call site remembering to pass a cutoff.
        covariates.bind_as_of(as_of)
    repair_log = RepairLog()
    loaded: LoadedDataset = load_stage(
        input_path, time_column=time_column, target_column=target_column,
        series_column=series_column, frequency=frequency,
        as_of=as_of, store_path=store_path,
        repair=repair, repair_log=repair_log, regrid=regrid,
    )
    task = ForecastTask(
        input_path if input_path.startswith("store:")
        else str(Path(input_path).expanduser().resolve()),
        loaded.schema, horizon,
        minimum_baseline_improvement=minimum_baseline_improvement,
        as_of=as_of.isoformat() if as_of else None,
    )
    # The textual-verifiability lane for future-dated context events.
    # Resolved once, here, so the flag has exactly one meaning per run and
    # the ID payload below can state it.
    future_events_enabled = bool(
        getattr(getattr(config, "context", None), "future_events", False)
    )
    structural_events_enabled = bool(
        getattr(getattr(config, "context", None), "structural_events", False)
    )
    future_context_admitted: dict[str, list[dict[str, object]]] = {}
    results: list[SeriesResult] = []
    evidence: list[Evidence] = []
    var_frame = None
    var_ineligible: str | None = None
    if multivariate:
        from .multivariate import VarFrame
        var_frame, var_ineligible = VarFrame.build(loaded.groups)
    for series_name, items in sorted(loaded.groups.items()):
        result, series_evidence = _series_result(
            series_name, items, loaded=loaded, horizon=horizon,
            seasonal_period=seasonal_period,
            minimum_baseline_improvement=minimum_baseline_improvement,
            config=config, strict_abstention=strict_abstention,
            selection_strategy=selection_strategy,
            multivariate=multivariate, var_frame=var_frame,
            var_ineligible=var_ineligible,
            context_events=context_events, covariates=covariates,
            adjudicating=adjudicating, threshold=threshold,
            target_coverage=target_coverage, repair_log=repair_log,
            future_events=future_events_enabled,
            structural_events=structural_events_enabled,
            best_effort=best_effort,
            minimum_support=minimum_support,
        )
        if result.future_context and result.future_context.get("admitted"):
            future_context_admitted[series_name] = list(
                result.future_context["admitted"]  # type: ignore[arg-type]
            )
        results.append(result)
        evidence.extend(series_evidence)
    if repair_log.has_actions():
        evidence.append(Evidence(
            "data_repair", "data_repair", "__all__",
            {"level": repair, **repair_log.summary()},
        ))
    snapshot_access: dict[str, object] = dict(loaded.snapshot.access_summary())
    if covariates is not None:
        # Merge the covariate reads in, so the `max_known_time` the
        # verifier's leakage check reads covers every source the run
        # consulted rather than the target series alone.
        covariate_access = covariates.access_summary()
        snapshot_access["accesses"] = list(snapshot_access.get("accesses", [])) + [
            {**entry, "source": "covariates"}
            for entry in covariate_access.get("accesses", [])
        ]
        snapshot_access["covariate_as_of"] = covariate_access.get("as_of")
    evidence.append(Evidence(
        "snapshot", "snapshot_access", "__all__", snapshot_access,
    ))
    # A selected TSFM's weights are part of what produced the numbers, so
    # they belong in the id and in the evidence. Without this the id covers
    # the model *name* only, and two runs at different Hub revisions could
    # publish different forecasts under one id.
    from .tsfm import resolved_weights
    selected_weights = {
        model: weights
        for model in sorted({
            item.selected_model for item in results if item.selected_model
        })
        if (weights := resolved_weights(model))
    }
    if selected_weights:
        evidence.append(Evidence(
            "model_weights", "model_weights", "__all__",
            {"pinned_revisions": selected_weights},
        ))
    id_payload: dict[str, object] = {
        "source": loaded.source_fingerprint,
        "as_of": as_of.isoformat() if as_of else None,
        "schema": {
            "time": time_column, "target": target_column, "series": series_column,
            "frequency": loaded.frequency, "timezone": loaded.timezone,
        },
        "horizon": horizon,
        "minimum_baseline_improvement": minimum_baseline_improvement,
        "threshold": threshold,
        "seasonal_period": seasonal_period,
        "selection_strategy": selection_strategy,
        "multivariate": multivariate,
        "strict_abstention": strict_abstention,
        "context_events": [event.__dict__ for event in context_events] if context_events else None,
        "covariates": {"source": covariates.fingerprint, "specs": [str(spec) for spec in covariates.specs]} if covariates else None,
        "config": _config_fingerprint(config),
        "runtime_version": RUNTIME_VERSION,
    }
    if repair != REPAIR_SAFE:
        # The default level is absent from the payload so IDs predating the
        # repair layer are unchanged.
        id_payload["repair"] = repair
    if best_effort:
        # Flag-on only, like future_context below: every flag-off ID —
        # including all pre-existing ones — is byte-identical.
        id_payload["best_effort"] = True
    if minimum_support != "best_effort":
        # The floor changes what is published, so it changes the id; the
        # default floor is omitted so fully supported runs keep their ids.
        id_payload["minimum_support"] = minimum_support
    if selected_weights:
        # Absent when no TSFM was selected, so ids for baseline and
        # statistical selections are unchanged by this addition.
        id_payload["model_weights"] = selected_weights
    if future_events_enabled or structural_events_enabled:
        # Same pattern as model_weights: the key exists only when a flag
        # is on, so every flag-off ID — including all pre-existing ones —
        # is byte-identical. When on, the ID covers both the flag and the
        # events that actually influenced the numbers. With only
        # future_events on the payload is unchanged from before the
        # structural class existed, so those IDs are stable too.
        id_payload["future_context"] = {
            "enabled": True,
            "admitted": future_context_admitted,
        }
        if structural_events_enabled:
            id_payload["future_context"]["structural_enabled"] = True
    forecast_id = content_id("forecast", id_payload)
    artifact = ForecastArtifact(
        "0.1", forecast_id, clock.now().isoformat(),
        "complete", task, loaded.source_fingerprint, results, evidence,
        input_provenance=(
            input_provenance
            or ("store" if input_path.startswith("store:") else None)
        ),
        runtime_version=RUNTIME_VERSION,
    )
    from .contracts import forecast_task
    from .lineage import build_forecast_lineage
    from .verifier import verify_or_raise
    temporal_task = forecast_task(
        task.input_path, time_column=time_column, target_column=target_column,
        horizon=horizon, series_column=series_column, frequency=loaded.frequency,
        threshold=threshold, minimum_baseline_improvement=minimum_baseline_improvement,
        as_of=task.as_of,
    )
    lineage = build_forecast_lineage(artifact, temporal_task)
    # No response leaves the process unverified — including our own.
    verify_or_raise(lineage, as_of=task.as_of)
    return artifact, write_artifact(
        artifact, output, lineage=lineage.to_dict(),
        output_config=getattr(config, "output", None),
    )


def _abstained_target_result(
    target: str, error: Any,
) -> tuple[SeriesResult, list[Evidence]]:
    """An honest per-channel abstention for a target whose column could
    not be loaded or evaluated. The error is carried verbatim — code and
    message in the warnings and the typed reasons, its repair options as
    recovery actions — so one bad channel is disclosed, not fatal, and
    never silently dropped."""
    from .contracts import REPAIR_OPTIONS, SupportAssessment, SupportReason

    message = f"{error.code}: {error.message}"
    repairs = (
        error.repair_options if error.repair_options is not None
        else REPAIR_OPTIONS.get(error.code, [])
    )
    assessment = SupportAssessment(
        "unsupported",
        reasons=[SupportReason(error.code, error.message)],
        recovery_actions=[
            SupportReason(str(option.get("action", "repair")),
                          str(option.get("description", "")))
            for option in repairs
        ],
        legacy_support="unsupported",
    )
    result = SeriesResult(
        target, "unsupported", None, None, {}, {}, None, None,
        [message], [], support_assessment=assessment.to_dict(),
    )
    evidence = [Evidence(f"support:{target}", "support_assessment", target, {
        "support": "unsupported", "warnings": [message],
    })]
    return result, evidence


def _default_worker_count(channels: int) -> int:
    """How many threads per-target evaluation should use by default.

    Measured, not assumed: the statistical evaluation path is pure
    Python, so under a GIL interpreter extra threads only add contention
    (a 6-channel batch ran ~13% *slower* at 6 workers than at 1 on
    CPython 3.12). Concurrency pays where the GIL is released — on a
    free-threaded build, or when sandboxed TSFM candidates run their
    inference in subprocesses — so those cases get min(channels, cpus)
    and everything else gets 1. `max_workers` overrides either way.
    """
    import sys

    cap = max(1, min(channels, os.cpu_count() or 1))
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        return cap
    try:
        from .tsfm_sandbox import list_sandboxes
        if list_sandboxes():
            return cap
    except Exception:
        pass
    return 1


def forecast_multi(
    input_path: str,
    *,
    time_column: str,
    target_columns: list[str],
    horizon: int,
    frequency: str | None = None,
    output: str = "gnomon-output",
    minimum_baseline_improvement: float = 0.02,
    context_events: list[ContextEvent] | None = None,
    covariates: CovariateDataset | None = None,
    threshold: float | None = None,
    config: Any = None,
    strict_abstention: bool = False,
    best_effort: bool = False,
    minimum_support: str = "best_effort",
    seasonal_period: int | None = None,
    selection_strategy: str = "best",
    clock: Clock | None = None,
    as_of: datetime | None = None,
    repair: str = "safe",
    regrid: str | None = None,
    candidates: list[str] | None = None,
    max_workers: int | None = None,
    input_provenance: str | None = None,
) -> tuple[ForecastArtifact, Path]:
    """Forecast several columns of one wide file in a single run.

    One shared load pass, then per-target evaluation on a thread pool —
    the channels are independent, so concurrency cannot change any
    number, and the artifact is identical at any worker count. One
    combined artifact carries one result per target column, each with
    its own support state; a channel that abstains is disclosed in its
    result and never blocks the others.

    Epistemics are untouched: each channel runs the exact single-target
    stage pipeline. Single-target invocations keep using
    :func:`forecast` and produce byte-identical artifacts to before.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .contracts import GnomonError
    from .pipeline import load_stage_multi
    from .repair import REPAIR_LEVELS

    clock = clock or SYSTEM_CLOCK
    if len(target_columns) < 2:
        raise GnomonError(
            "INVALID_ARGUMENTS",
            "forecast_multi needs at least two target columns; use forecast() "
            "for a single target.",
            {"target_columns": list(target_columns)},
        )
    if len(set(target_columns)) != len(target_columns):
        # Result series and evidence records are keyed by target name, so a
        # repeated name would collide identifiers inside one artifact.
        duplicates = sorted({
            name for name in target_columns if target_columns.count(name) > 1
        })
        raise GnomonError(
            "INVALID_ARGUMENTS",
            f"target_columns names the same column more than once: "
            f"{', '.join(duplicates)}.",
            {"duplicates": duplicates, "target_columns": list(target_columns)},
        )
    if candidates:
        config = _restrict_candidates(config, candidates)
    if repair not in REPAIR_LEVELS:
        raise GnomonError(
            "INVALID_REPAIR_LEVEL",
            f"repair must be one of {', '.join(REPAIR_LEVELS)}.",
            {"requested": repair, "supported": list(REPAIR_LEVELS)},
        )
    if horizon < 1:
        raise GnomonError("INVALID_HORIZON", "Horizon must be at least one period.")
    from .support import PUBLICATION_TIERS
    if minimum_support not in PUBLICATION_TIERS:
        from .contracts import GnomonError
        raise GnomonError(
            "INVALID_ARGUMENTS",
            f"minimum_support must be one of {', '.join(PUBLICATION_TIERS)}.",
            {"requested": minimum_support,
             "supported": list(PUBLICATION_TIERS)},
        )
    if selection_strategy == "ensemble":
        import copy as _copy
        from .config import load_config as _load_config
        config = _copy.deepcopy(config) if config is not None else _load_config()
        config.ensemble.enabled = True
    from .evaluation import DEFAULT_TARGET_COVERAGE
    target_coverage = DEFAULT_TARGET_COVERAGE
    if config is not None and getattr(config, "evaluation", None) is not None:
        target_coverage = float(
            getattr(config.evaluation, "target_coverage", DEFAULT_TARGET_COVERAGE)
        )

    datasets, repair_logs, source_fingerprint, _columns = load_stage_multi(
        input_path, time_column=time_column, target_columns=list(target_columns),
        frequency=frequency, as_of=as_of, repair=repair, regrid=regrid,
    )
    loaded_any = [item for item in datasets.values() if isinstance(item, LoadedDataset)]
    if not loaded_any:
        # Every channel failed to load: the file itself is the problem, and
        # the first error is the diagnosis.
        first_error = next(iter(datasets.values()))
        raise first_error  # type: ignore[misc]
    resolved_frequency = loaded_any[0].frequency
    timezone_name = loaded_any[0].timezone
    if covariates is not None:
        # Bind once before worker threads can read the dataset.  The snapshot
        # is immutable at this boundary, so every channel sees the same
        # point-in-time covariate view and cannot widen it independently.
        covariates.bind_as_of(as_of)

    future_events = bool(
        config is not None and getattr(getattr(config, "context", None),
                                       "future_events", False)
    )
    structural_events = bool(
        config is not None and getattr(getattr(config, "context", None),
                                       "structural_events", False)
    )

    def run_target(target: str) -> tuple[SeriesResult, list[Evidence]]:
        loaded_or_error = datasets[target]
        if isinstance(loaded_or_error, GnomonError):
            return _abstained_target_result(target, loaded_or_error)
        try:
            return _series_result(
                target, loaded_or_error.groups[target],
                loaded=loaded_or_error, horizon=horizon,
                seasonal_period=seasonal_period,
                minimum_baseline_improvement=minimum_baseline_improvement,
                config=config, strict_abstention=strict_abstention,
                selection_strategy=selection_strategy,
                multivariate=False, var_frame=None, var_ineligible=None,
                context_events=context_events, covariates=covariates,
                adjudicating=bool(context_events) and covariates is not None,
                threshold=threshold, target_coverage=target_coverage,
                repair_log=repair_logs[target],
                best_effort=best_effort,
                minimum_support=minimum_support,
                future_events=future_events,
                structural_events=structural_events,
            )
        except GnomonError as error:
            return _abstained_target_result(target, error)

    workers = max_workers or _default_worker_count(len(target_columns))
    workers = max(1, min(int(workers), len(target_columns)))
    if workers == 1:
        # No pool for a single worker: on a GIL interpreter even an idle
        # thread handoff measurably taxes this pure-CPU workload (~5% on
        # the 6-channel benchmark), and a one-worker pool cannot buy
        # anything back.
        outcomes = [run_target(target) for target in target_columns]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Submitted and collected in target order, so the artifact is a
            # pure function of the task regardless of scheduling.
            outcomes = list(pool.map(run_target, target_columns))

    results: list[SeriesResult] = []
    evidence: list[Evidence] = []
    for result, series_evidence in outcomes:
        results.append(result)
        evidence.extend(series_evidence)
    for target in target_columns:
        log = repair_logs.get(target)
        if log is not None and log.has_actions():
            evidence.append(Evidence(
                f"data_repair:{target}", "data_repair", target,
                {"level": repair, **log.summary()},
            ))
    accesses: list[dict[str, object]] = []
    for target in target_columns:
        item = datasets[target]
        if isinstance(item, LoadedDataset):
            summary = item.snapshot.access_summary()
            accesses.extend(summary.get("accesses", []))  # type: ignore[arg-type]
    evidence.append(Evidence("snapshot", "snapshot_access", "__all__", {
        "as_of": as_of.isoformat() if as_of else "latest",
        "known_time_assumed": loaded_any[0].snapshot.assumed_known_time,
        "known_time_provenance": loaded_any[0].snapshot.known_time_provenance,
        "source_ref": source_fingerprint,
        "accesses": accesses,
    }))
    if covariates is not None:
        covariate_access = covariates.access_summary()
        evidence.append(Evidence(
            "covariate_snapshot", "snapshot_access", "__all__",
            covariate_access,
        ))
    from .tsfm import resolved_weights
    selected_weights = {
        model: weights
        for model in sorted({
            item.selected_model for item in results if item.selected_model
        })
        if (weights := resolved_weights(model))
    }
    if selected_weights:
        evidence.append(Evidence(
            "model_weights", "model_weights", "__all__",
            {"pinned_revisions": selected_weights},
        ))
    id_payload: dict[str, object] = {
        "source": source_fingerprint,
        "as_of": as_of.isoformat() if as_of else None,
        # `target` is the ordered list of columns — a JSON array, which no
        # single-target payload (a string) can collide with, so existing
        # forecast IDs are untouched.
        "schema": {
            "time": time_column, "target": list(target_columns), "series": None,
            "frequency": resolved_frequency, "timezone": timezone_name,
        },
        "horizon": horizon,
        "minimum_baseline_improvement": minimum_baseline_improvement,
        "threshold": threshold,
        "seasonal_period": seasonal_period,
        "selection_strategy": selection_strategy,
        "multivariate": False,
        "strict_abstention": strict_abstention,
        "context_events": ([event_to_dict(event) for event in context_events]
                           if context_events else None),
        "covariates": ({
            "source": covariates.fingerprint,
            "specs": [str(spec) for spec in covariates.specs],
        } if covariates else None),
        "config": _config_fingerprint(config),
        "runtime_version": RUNTIME_VERSION,
    }
    from .repair import REPAIR_SAFE
    if repair != REPAIR_SAFE:
        id_payload["repair"] = repair
    if best_effort:
        id_payload["best_effort"] = True
    if minimum_support != "best_effort":
        # The floor changes what is published, so it changes the id; the
        # default floor is omitted so fully supported runs keep their ids.
        id_payload["minimum_support"] = minimum_support
    if selected_weights:
        id_payload["model_weights"] = selected_weights
    forecast_id = content_id("forecast", id_payload)
    task = ForecastTask(
        str(Path(input_path).expanduser().resolve()),
        DataSchema(time_column, ",".join(target_columns), None,
                   resolved_frequency, timezone_name),
        horizon,
        minimum_baseline_improvement=minimum_baseline_improvement,
        as_of=as_of.isoformat() if as_of else None,
    )
    artifact = ForecastArtifact(
        "0.1", forecast_id, clock.now().isoformat(),
        "complete", task, source_fingerprint, results, evidence,
        input_provenance=(
            input_provenance
            or ("store" if input_path.startswith("store:") else None)
        ),
        runtime_version=RUNTIME_VERSION,
    )
    from .contracts import forecast_task
    from .lineage import build_forecast_lineage
    from .verifier import verify_or_raise
    temporal_task = forecast_task(
        task.input_path, time_column=time_column,
        target_column=",".join(target_columns),
        horizon=horizon, series_column=None, frequency=resolved_frequency,
        threshold=threshold,
        minimum_baseline_improvement=minimum_baseline_improvement,
        as_of=task.as_of,
    )
    lineage = build_forecast_lineage(artifact, temporal_task)
    # No response leaves the process unverified — including our own.
    verify_or_raise(lineage, as_of=task.as_of)
    return artifact, write_artifact(
        artifact, output, lineage=lineage.to_dict(),
        output_config=getattr(config, "output", None),
    )


def _has_module(name: str) -> bool:
    from importlib.util import find_spec
    return find_spec(name) is not None


def _response_budget_bytes() -> int:
    # Local import: toolspec imports runtime, so the reverse edge is lazy.
    from .toolspec import RESPONSE_BUDGET_BYTES
    return RESPONSE_BUDGET_BYTES


def _default_minimum_support() -> str:
    from .support import DEFAULT_MINIMUM_SUPPORT
    return DEFAULT_MINIMUM_SUPPORT


def _mcp_profile() -> dict[str, object]:
    # Local import: toolspec imports runtime, so the reverse edge is lazy.
    from .toolspec import PROFILES, active_profile, visible_tools
    return {
        "active": active_profile(),
        "available": sorted(PROFILES) + ["full"],
        "visible_tools": [tool["name"] for tool in visible_tools()],
    }


def capabilities() -> dict[str, object]:
    try:
        import pyarrow  # type: ignore[import-not-found]  # noqa: F401
        parquet = True
    except ImportError:
        parquet = False
    from .registry import registry_capabilities
    from .tsfm import available_tsfms, capability_matrix, installed_tsfms
    from .tsfm_sandbox import list_sandboxes
    try:
        from .config import load_config
        _capabilities_config = load_config()
        future_events_on = bool(_capabilities_config.context.future_events)
        structural_events_on = bool(
            getattr(_capabilities_config.context, "structural_events", False)
        )
    except Exception:
        # A malformed config file must not make capabilities unreportable;
        # the flags read as their defaults.
        future_events_on = False
        structural_events_on = False
    return {
        "schema_version": "0.1",
        "runtime_version": RUNTIME_VERSION,
        "interfaces": {"cli": True, "python": True, "mcp": True, "http": False},
        "inputs": {
            "csv": True, "tsv": True, "json": True, "jsonl": True,
            "gzip": True, "parquet": parquet, "excel": _has_module("openpyxl"),
        },
        "frequencies": sorted(SEASONS),
        "frequency_descriptions": dict(FREQUENCY_DESCRIPTIONS),
        # Named codes are not the whole grid: any strictly regular
        # whole-second step shorter than one day is accepted, written
        # <N>s / <N>min / <N>h and inferred automatically when the series
        # has exactly one unique spacing.
        "general_frequencies": {
            "pattern": r"^[1-9]\d*(s|min|h)$",
            "description": "any whole-second step shorter than one day",
        },
        "context_events": {
            "fold_validated": {
                "model": "event_adjusted",
                "effect_shapes": ["level", "decay", "ramp"],
                "admission": "identical-fold ablation with known-at gating",
            },
            "future_events": {
                "flag": "context.future_events",
                "default": "off",
                # The loaded config file's setting. Config is honoured where
                # config is consumed: `gnomon forecast` (with or without
                # --config) and the Python API's `config=` parameter. MCP
                # tool calls and the decide/monitor macros do not read
                # gnomon.yaml — for any setting, not only this one — so the
                # lane is off there regardless of this value.
                "enabled_in_config": future_events_on,
                "event_classes": ["constraint", "deterministic_override"],
                "admission": (
                    "textual verifiability: a quoted source span and "
                    "deterministic re-parsing of its numbers (absolute, "
                    "or multiples/percentages of the recent-window "
                    "median) — only for windows with no overlap with "
                    "the observed history; history's relation to a "
                    "bound is disclosed, never used to reject"
                ),
                "disclosure": (
                    "influenced forecasts report support "
                    "'context_trusted' and carry the history-only "
                    "counterfactual in evidence"
                ),
            },
            "structural_events": {
                "flag": "context.structural_events",
                "default": "off",
                "enabled_in_config": structural_events_on,
                "event_classes": ["structural"],
                "effects": ["trend_ceases", "level_matches_seasonal_high",
                            "level_matches_seasonal_low"],
                "admission": (
                    "LLM-classified from a closed effect menu with a "
                    "quoted source span; no numeric parse — the class "
                    "carries no number, and every applied quantity is "
                    "derived from Gnomon's own data (the emitted path "
                    "for trend_ceases; per-phase envelope quantiles of "
                    "the observed history for the regime effects)"
                ),
                "disclosure": (
                    "same as future_events: support 'context_trusted', "
                    "history-only counterfactual, admitted events in "
                    "the artifact ID payload"
                ),
                "experimental": "results/structural-effects/HYPOTHESIS.md",
            },
        },
        "models": {
            "baselines": sorted(BASELINES),
            "statistical": sorted(name for name in MODELS if name not in BASELINES),
            "context": ["event_adjusted"],
            # Adapters that can actually run right now: importable
            # in-process or with a ready sandbox. `installed_tsfms()` alone
            # reported [] after a successful `gnomon tsfm install`, because
            # it requires torch importable in the *main* process — which
            # the sandbox model exists to avoid.
            "tsfm": sorted(set(installed_tsfms()) | set(list_sandboxes())),
            "tsfm_available": available_tsfms(),
            "tsfm_sandboxes": list_sandboxes(),
            "tsfm_capabilities": capability_matrix(),
            "tsfm_install_command": "gnomon tsfm install <name>",
            "tsfm_install_tool": "gnomon_install_tsfm",
            "tsfm_install_tool_profile": "full",
            "tsfm_install_note": (
                "Sandboxed TSFMs are pulled per model into isolated venvs "
                "(requires uv; weights download on first inference). "
                "Installed models join forecast selection automatically; "
                "moment_small also adds a reconstruction candidate to "
                "detect_anomalies. Install from the shell with the "
                "command above, or from the tool surface with "
                "gnomon_install_tsfm, which starts a detached install "
                "and reports state on each call."
            ),
        },
        **registry_capabilities(),
        "short_history": {
            # Behavior at fold-starved history, always on (not a flag): the
            # trust contract is that under-powered evidence is not acted on
            # as if it ranked anything.
            "selection_guardrail": (
                "below 2 disjoint selection folds the selection margin "
                "rises to 75%: a candidate is selectable only by cutting "
                "the strongest baseline's single-fold error by more than "
                "three-quarters (deterministic structure, not fold luck); "
                "otherwise the baseline is published with a "
                "'selection_underpowered' reason and selection_fold_count "
                "in sensitivity"
            ),
            "point_recentring": (
                "on degraded runs, quantiles are centred on the model's "
                "point path (point_bias_correction = 0) instead of the "
                "median backtest residual, whose few-fold location "
                "estimate measured as noise; disclosed as "
                "'point_recentring_suppressed'"
            ),
        },
        # Where this process writes. Agents used to guess output_dir (and
        # burn a round-trip on a host's path-jail refusal) because nothing
        # disclosed the allowed default; a jailed host should start the
        # server with its working directory inside the jail, and this block
        # is how the agent learns where that is.
        "workspace": {
            "cwd": os.getcwd(),
            "default_output_dir": str(Path("gnomon-output").resolve()),
            "note": (
                "Omit output_dir to write artifacts under "
                "default_output_dir (content-addressed, immutable "
                "directories). Relative paths resolve against cwd."
            ),
        },
        "mcp_profile": _mcp_profile(),
        "features": {
            "inspection": True, "forecasting": True, "separated_evaluation": True,
            "investigate_change": True, "decide": True, "monitor": True,
            "anomaly_detection": True, "graded_detector_selection": True,
            "series_fingerprints": True, "task_conditioned_leaderboard": True,
            "task_routing": True,
            "bitemporal_store": True, "as_of_replay": True, "typed_lineage": True,
            "claim_verifier": True,
            "residual_intervals": True, "horizon_widened_intervals": True,
            "threshold_analysis": True, "degraded_evaluation": True,
            "short_history_selection_guardrail": True,
            "short_history_point_centred_intervals": True,
            "project_mode": True, "actual_scoring": True,
            "decision_outcomes": True, "agent_treatment_control_eval": True,
            "context_events": True, "future_context_events": True,
            "llm_workflow_prompts": True, "sharing": False,
            "future_known_covariates": True, "point_in_time_covariates": True,
            "covariate_ablation": True, "enrichment_adjudication": True,
            "season_detection": True, "ensemble_forecasting": True,
            "multivariate_var": True, "strict_abstention": True,
            "best_effort_fallback": True,
            "graduated_support": True, "horizon_split": True,
            "row_tier_labels": True, "forecast_headline": True,
            "multi_target_batching": True, "brief_output": True,
            "inline_data_channels": True,
            "structural_regrid": True, "long_series_fit_window": True,
            "tsfm_install": True,
        },
        "forecast_surface": {
            # Machine-readable notes on the two agent-facing additions, so a
            # host can discover them without reading prose.
            "multi_target_batching": {
                "cli": "--target hr,spo2,resp or --target auto",
                "mcp": "target_column accepts a comma list or 'auto'",
                "semantics": (
                    "One shared load pass; per-target evaluation runs "
                    "concurrently; one combined artifact with a result per "
                    "column. Per-channel numbers are identical to "
                    "single-target runs, and an abstaining channel never "
                    "blocks the others."
                ),
            },
            "brief_output": {
                "cli": "--brief",
                "mcp": "format: 'brief' (the MCP default; pass 'full' to opt out)",
                "semantics": (
                    "q50 with one q10-q90 interval per step, plus the "
                    "support state, warnings, abstention reasons, recovery "
                    "actions, and disclosures verbatim. The on-disk "
                    "artifact is unchanged."
                ),
            },
            "graduated_support": {
                "default_minimum_support": _default_minimum_support(),
                "tiers": ["best_effort", "conditionally_supported",
                          "supported"],
                "cli": "--minimum-support",
                "mcp": "minimum_support",
                "semantics": (
                    "Publication floor only: the evaluation and every "
                    "tier's earning conditions are unchanged. The default "
                    "floor publishes the most defensible answer that "
                    "exists, tier-labelled; a higher floor restores the "
                    "typed refusal, and a series with no usable history "
                    "still abstains."
                ),
            },
            "response_budget": {
                "mcp_bytes": _response_budget_bytes(),
                "semantics": (
                    "Tool responses over the budget trim long arrays to "
                    "their first/last entries with truncated: true and a "
                    "pointer at the artifact. Support assessments, "
                    "warnings, assumptions, and error/repair payloads are "
                    "never trimmed."
                ),
            },
            "inline_data_channels": {
                "mcp": (
                    "observations (rows, replaces input), context_events "
                    "(concatenates with context_events_file), covariates "
                    "(rows, excludes covariates_file), actuals (rows, "
                    "excludes actuals_file)"
                ),
                "semantics": (
                    "Every file parameter on the tool surface has an "
                    "inline array equivalent, so a caller without a "
                    "filesystem can reach the whole engine. Both channels "
                    "of each pair run the identical validation; inline "
                    "content is fingerprinted for evidence just as files "
                    "are."
                ),
            },
        },
    }
