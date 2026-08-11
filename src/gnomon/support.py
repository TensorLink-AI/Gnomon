"""Reference implementation of :class:`SupportAssessment` for the forecast
verb — the one honest mapping from the frozen v0.2 support enum onto the
harness-wide vocabulary.

The v0.2 enum conflates three axes: evidence strength (``weakly_supported``),
evaluation quality (``degraded``), and model identity (``supported_ensemble``).
Here they are separated: model identity stays in ``selected_model``,
evaluation quality becomes typed reasons, and the status states only what
the evidence supports.
"""

from __future__ import annotations

from .contracts import (
    MAX_VERIFIABLE_COVERAGE,
    MIN_VERIFIABLE_COVERAGE,
    SupportAssessment,
    SupportReason,
    interval_calibration_is_verifiable,
)
from .evaluation import Evaluation
from .temporal_store import KNOWN_TIME_ASSUMED_WARNING

#: The publication tiers, weakest first. These grade what is *published*,
#: not how any tier is earned — the fold minimums, separation
#: requirements, and guardrails that earn each rung are untouched by the
#: graduated-support default. `minimum_support` is a floor over this
#: order: the evaluation runs exactly as always, then the result is
#: published at the highest tier the evidence achieved that clears the
#: floor, or abstains.
PUBLICATION_TIERS = ("best_effort", "conditionally_supported", "supported")
TIER_ORDER = {name: rank for rank, name in enumerate(PUBLICATION_TIERS)}
DEFAULT_MINIMUM_SUPPORT = "best_effort"


def achieved_tier(status: str, has_rows: bool) -> str | None:
    """The publication tier a result's assessment status earned.

    ``None`` when nothing was published. Rows published from an
    ``inconclusive`` assessment are the disclosed fallback — best_effort
    by construction."""
    if not has_rows:
        return None
    if status in ("supported", "conditionally_supported"):
        return status
    return "best_effort"


def assess_forecast_support(
    support: str,
    warnings: list[str],
    assessment: Evaluation | None,
    *,
    known_time_assumed: bool = False,
    disclosures: list[SupportReason] | None = None,
    measured_coverage: float | None = None,
) -> SupportAssessment:
    """``measured_coverage`` is the coverage of the *published* model.

    It is not always ``assessment.coverage``: a covariate or adjudication
    winner replaces the base model's forecast and its measured coverage
    with its own, and it is the published one a reader is being asked to
    trust. Omitted, it falls back to the assessment's.
    """
    disclosures = list(disclosures or [])
    reasons = [SupportReason("warning", message) for message in warnings]
    assumptions: list[str] = []
    if known_time_assumed:
        assumptions.append(KNOWN_TIME_ASSUMED_WARNING)
    sensitivity: dict[str, object] = {}
    if assessment is not None:
        if assessment.improvement is not None:
            sensitivity["baseline_improvement"] = assessment.improvement
        if assessment.coverage is not None:
            sensitivity["final_test_interval_coverage"] = assessment.coverage
        # A selection statistic without its sample size reads as evidence
        # whatever its n; on a fold-starved run the n is the story. Only on
        # degraded runs, so fully-evidenced artifacts are byte-unchanged.
        if assessment.degraded and assessment.selection_fold_count:
            sensitivity["selection_fold_count"] = assessment.selection_fold_count

    # A coverage failure is its own typed reason, prepended so that no
    # branch below can drop it. `degraded` used to replace the reason list
    # wholesale, which meant a badly calibrated *and* degraded run reported
    # only that it was degraded — the more serious fact vanished.
    coverage = measured_coverage
    if coverage is None and assessment is not None:
        coverage = assessment.coverage
    miscalibrated = not interval_calibration_is_verifiable(coverage)
    if miscalibrated:
        reasons.insert(0, SupportReason(
            "interval_coverage_out_of_band",
            f"Measured interval coverage is {float(coverage):.1%}, outside "
            f"the band [{MIN_VERIFIABLE_COVERAGE:.0%}, "
            f"{MAX_VERIFIABLE_COVERAGE:.0%}] required to treat the intervals "
            f"as probabilities. The point path stands; the intervals do not "
            f"carry probability weight.",
        ))

    if support == "context_trusted":
        # The forecast was influenced by future-dated context admitted on
        # textual verifiability (`context.future_events`), which no fold
        # ever tested. Weaker than any fold-backed status by design; the
        # evaluation facts stay visible in the reasons rather than being
        # displaced by the downgrade.
        trust_reasons = [SupportReason(
            "unbacktested_context_admitted",
            "Future-dated context events were admitted by textual "
            "verification — a quoted source span whose numbers a "
            "deterministic parser re-extracted — not by fold ablation: "
            "their windows have no historical precedent to ablate. The "
            "forecast as it stood before these events is recorded as the "
            "history-only counterfactual in the future_context_applied "
            "evidence.",
        )]
        if assessment is not None and assessment.degraded:
            trust_reasons.append(SupportReason(
                "degraded_evaluation",
                "Model selection ran without separated calibration and "
                "test folds.",
            ))
        recovery = [SupportReason(
            "compare_counterfactual",
            "Compare against the history-only counterfactual in the "
            "evidence; set context.future_events: off to remove the "
            "influence entirely.",
        )]
        if miscalibrated:
            recovery.append(SupportReason(
                "treat_intervals_as_indicative",
                "Use the point path and the model comparison; do not read "
                "the intervals as calibrated probabilities.",
            ))
        return SupportAssessment(
            "conditionally_supported", trust_reasons + reasons, assumptions,
            sensitivity, recovery, support, disclosures,
        )
    if support in ("supported", "supported_ensemble"):
        extra: list[SupportReason] = []
        status = "supported"
        if support == "supported_ensemble":
            # Derived from the measurement, never asserted. `--ensemble` can
            # force the ensemble onto a series a baseline won, and the old
            # unconditional sentence then stated the opposite of the evidence
            # sitting beside it — exactly what the verifier exists to catch.
            improvement = assessment.improvement if assessment is not None else None
            if improvement is None:
                # No measurement to appeal to, so assert nothing about one.
                extra.append(SupportReason(
                    "ensemble_selection",
                    "An ensemble of eligible models produced this forecast.",
                ))
            elif improvement > 0:
                extra.append(SupportReason(
                    "ensemble_selection",
                    f"An ensemble of eligible models beat the strongest "
                    f"baseline by {improvement:.2%} on the selection folds.",
                ))
            else:
                extra.append(SupportReason(
                    "ensemble_forced",
                    f"The ensemble was selected by an explicit request, not by "
                    f"the backtest: its measured improvement over the strongest "
                    f"baseline is {improvement:.2%}.",
                ))
                status = "conditionally_supported"
        if miscalibrated:
            status = "conditionally_supported"
        return SupportAssessment(
            status, extra + reasons, assumptions, sensitivity,
            [SupportReason(
                "treat_intervals_as_indicative",
                "Use the point path and the model comparison; do not read "
                "the intervals as calibrated probabilities.",
            )] if miscalibrated else [],
            support, disclosures,
        )
    if support == "weakly_supported":
        return SupportAssessment(
            "conditionally_supported", reasons, assumptions, sensitivity,
            [SupportReason("review_warnings",
                           "Inspect the attached warnings; each names the condition under which this forecast holds.")],
            support, disclosures,
        )
    if support == "degraded":
        # `reasons` already carries the coverage failure at its head when
        # there is one, so the degraded reason is added beside it, never
        # in place of it.
        degraded_reasons = [SupportReason(
            "degraded_evaluation",
            "Model selection ran without separated calibration and test folds.")]
        if assessment is not None and assessment.selection_guardrail_applied:
            degraded_reasons.append(SupportReason(
                "selection_underpowered",
                f"Candidates were scored on {assessment.selection_fold_count} "
                f"selection fold{'s' if assessment.selection_fold_count != 1 else ''} "
                f"— too few to rank them, so the strongest baseline "
                f"({assessment.strongest_baseline}) was published and the "
                f"candidate scores are evidence, not a ranking.",
            ))
        return SupportAssessment(
            "conditionally_supported",
            degraded_reasons + reasons,
            assumptions, sensitivity,
            [SupportReason("provide_more_history",
                           "Supply enough observations for separated selection, calibration, and test windows.")],
            support, disclosures,
        )
    # v0.2 "unsupported" is always a data-insufficiency abstention: the
    # evaluation could not run, so the honest status is inconclusive — the
    # evidence does not argue against forecasting, it is simply absent.
    # "best_effort" is the same abstention with disclosed fallback rows
    # attached, so it shares the status and the recovery path; the rows
    # change what was published, not what the evidence supports.
    recovery = [SupportReason("provide_more_history",
                              "Supply more observations, or lower the horizon, so rolling evaluation can run.")]
    reachable = assessment.max_supportable_horizon if assessment is not None else None
    if reachable is not None:
        # A refusal must not be a dead end: name the horizon that trades
        # reach for an honest result with the data already supplied.
        recovery.insert(0, SupportReason(
            "reduce_horizon",
            f"Retry with horizon {reachable} or less: the observations "
            f"already supplied support evaluation at that horizon.",
        ))
    if support == "unsupported":
        # The disclosed fallback lane exists, but a caller who has never
        # read the CLI reference cannot know that. The refusal teaches
        # the path so recovery takes one step, not prior knowledge; the
        # fallback stays labelled wherever it is published.
        recovery.append(SupportReason(
            "retry_best_effort",
            "If rows are needed at this horizon anyway, retry with "
            "minimum_support 'best_effort' (the default; `--minimum-support "
            "best_effort` on the CLI): a naive fallback is published with "
            "support `best_effort` and a NO RELIABLE FORECAST warning — "
            "disclosed rows, never a supported forecast.",
        ))
    insufficiency = (
        [SupportReason("insufficient_evaluation", message) for message in warnings]
        or [SupportReason("insufficient_evaluation", "The evaluation protocol could not complete.")]
    )
    if support == "best_effort":
        insufficiency.insert(0, SupportReason(
            "no_reliable_forecast",
            "The published rows are a best-effort fallback, not a "
            "supported forecast: nothing was selected by a backtest and "
            "nothing was measured on a fold.",
        ))
    return SupportAssessment(
        "inconclusive",
        insufficiency,
        assumptions, sensitivity,
        recovery,
        support, disclosures,
    )


def disclose_epistemic_deviation(
    assessment: SupportAssessment, reason: SupportReason, *, cap: bool,
) -> SupportAssessment:
    """Stamp a non-default evidence rule onto an already-built assessment.

    An epistemic parameter moved off its default (see
    ``contracts.PARAMETER_AUTHORITY``) changes what the numbers mean, so
    the deviation rides in ``reasons`` where an agent already looks. With
    ``cap=True`` — the parameter was moved in the lax direction — a
    ``supported`` verdict is additionally capped at
    ``conditionally_supported``: the evidence may be fine, but it was
    judged by a weaker rule than the one the documentation promises.
    Applied after assembly so every branch of
    :func:`assess_forecast_support` is covered by construction.
    """
    assessment.reasons = [reason] + list(assessment.reasons)
    if cap and assessment.status == "supported":
        assessment.status = "conditionally_supported"
    return assessment
