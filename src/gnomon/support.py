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
        return SupportAssessment(
            "conditionally_supported",
            [SupportReason("degraded_evaluation",
                           "Model selection ran without separated calibration and test folds.")] + reasons,
            assumptions, sensitivity,
            [SupportReason("provide_more_history",
                           "Supply enough observations for separated selection, calibration, and test windows.")],
            support, disclosures,
        )
    # v0.2 "unsupported" is always a data-insufficiency abstention: the
    # evaluation could not run, so the honest status is inconclusive — the
    # evidence does not argue against forecasting, it is simply absent.
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
    return SupportAssessment(
        "inconclusive",
        [SupportReason("insufficient_evaluation", message) for message in warnings]
        or [SupportReason("insufficient_evaluation", "The evaluation protocol could not complete.")],
        assumptions, sensitivity,
        recovery,
        support, disclosures,
    )
