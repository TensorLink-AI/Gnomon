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

from .contracts import SupportAssessment, SupportReason
from .evaluation import Evaluation
from .temporal_store import KNOWN_TIME_ASSUMED_WARNING


def assess_forecast_support(
    support: str,
    warnings: list[str],
    assessment: Evaluation | None,
    *,
    known_time_assumed: bool = False,
    disclosures: list[SupportReason] | None = None,
) -> SupportAssessment:
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

    if support in ("supported", "supported_ensemble"):
        extra = [SupportReason("ensemble_selection",
                               "An ensemble of eligible models beat the strongest baseline.")] \
            if support == "supported_ensemble" else []
        return SupportAssessment(
            "supported", extra + reasons, assumptions, sensitivity, [], support,
            disclosures,
        )
    if support == "weakly_supported":
        return SupportAssessment(
            "conditionally_supported", reasons, assumptions, sensitivity,
            [SupportReason("review_warnings",
                           "Inspect the attached warnings; each names the condition under which this forecast holds.")],
            support, disclosures,
        )
    if support == "degraded":
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
