from __future__ import annotations

import pytest

from gnomon.breach import (
    BreachDecisionPolicy,
    aligned_residual_paths,
    apply_breach_policy,
    estimate_horizon_breach,
)
from gnomon.pipeline import bounded_threshold_assessment


def _bounded_rows(points, lows, highs):
    return [
        {"timestamp": f"2026-01-{index + 1:02d}", "point": point,
         "q10": low, "q50": point, "q90": high}
        for index, (point, low, high) in enumerate(zip(points, lows, highs))
    ]


def test_bounded_threshold_assessment_separates_point_answer_from_uncertainty():
    result = bounded_threshold_assessment(
        10, _bounded_rows([8, 9], [6, 7], [11, 12]))
    bounded = result["bounded_assessment"]
    assert result["probability_status"] == "unavailable_uncalibrated"
    assert result["probability_above"] == []
    assert bounded["best_estimate"] == "no"
    assert bounded["decision"] == "indeterminate"
    assert bounded["primary"]["published_range_relation"] == \
        "range_overlaps_threshold"
    assert bounded["automation_eligible"] is False


def test_bounded_threshold_assessment_can_make_a_nonprobabilistic_clear_call():
    below = bounded_threshold_assessment(
        10, _bounded_rows([7, 8], [5, 6], [8, 9]))
    above = bounded_threshold_assessment(
        10, _bounded_rows([11, 12], [10.5, 11], [13, 14]))
    assert below["bounded_assessment"]["decision"] == "no"
    assert above["bounded_assessment"]["decision"] == "yes"
    assert below["bounded_assessment"]["automation_eligible"] is False
    assert above["bounded_assessment"]["automation_eligible"] is False


def test_bounded_threshold_assessment_surfaces_candidate_conflict():
    result = bounded_threshold_assessment(
        10, _bounded_rows([7, 8], [5, 6], [8, 9]),
        alternate_paths=[{
            "path": "model_assisted", "points": [11, 12],
            "support": "prior_assisted"}])
    bounded = result["bounded_assessment"]
    assert bounded["best_estimate"] == "no"
    assert bounded["model_conflict"] is True
    assert bounded["decision"] == "indeterminate"
    assert bounded["alternatives"][0]["best_estimate"] == "yes"
    assert bounded["primary_forecast_unchanged"] is True
from gnomon.evaluation import evaluate


def _rows(horizon: int, centre: float = 10.0):
    return [
        {"point": centre, "q10": centre - 2, "q50": centre,
         "q90": centre + 2}
        for _ in range(horizon)
    ]


def _by_lead(paths: list[list[float]]) -> dict[int, list[float]]:
    return {
        step + 1: [path[step] for path in paths]
        for step in range(len(paths[0]))
    }


def test_aligned_paths_never_mix_origins() -> None:
    by_lead = {1: [1, 10], 2: [2, 20], 3: [3, 30]}
    assert aligned_residual_paths(by_lead, 3) == [[1, 2, 3], [10, 20, 30]]


def test_any_breach_preserves_within_path_dependence() -> None:
    # Eight of twenty origins breach, always across every lead together.
    # Multiplying marginals as independent would report ~78%; trajectory
    # replay correctly reports 40%.
    block = ([[3.0, 3.0, 3.0]] * 4
             + [[-1.0, -1.0, -1.0]] * 6)
    paths = block + block
    risk = estimate_horizon_breach(
        _rows(3), 12.0, _by_lead(paths),
        measured_interval_coverage=0.8,
        calibration_is_verifiable=True,
    )
    assert risk["probability_any_breach"] == 0.4
    assert risk["dependence_preserved"] is True
    assert risk["joint_path_count"] == 20
    assert risk["first_breach_step_probability"] == {"1": 0.4}
    assert risk["support"] == "supported"


def test_regime_change_demotes_probability_to_best_effort() -> None:
    paths = [[0.0, 0.0] for _ in range(8)] + [
        [20.0, 20.0] for _ in range(8)
    ]
    risk = estimate_horizon_breach(
        _rows(2), 12.0, _by_lead(paths),
        measured_interval_coverage=0.8,
        calibration_is_verifiable=True,
    )
    # The estimate still exists — hiding it entirely prices as never-act
    # under asymmetric costs — but it must not carry governed authority.
    assert risk["support"] == "best_effort"
    assert risk["probability_any_breach"] is not None
    assert risk["residual_regime"]["status"] == "changed"
    assert "residual_regime_changed" in {
        reason["code"] for reason in risk["reasons"]
    }


def test_starved_histories_climb_down_the_ladder_not_off_a_cliff() -> None:
    """One event-calibration origin (the short-history reality measured on
    every BreachBench case) must yield a labelled best-effort estimate
    from the richer selection folds, never silence."""
    event = _by_lead([[1.0, 1.0, 1.0]])
    selection = _by_lead([
        [3.0, 3.0, 3.0], [-1.0, -1.0, -1.0], [0.5, 0.5, 0.5],
        [2.5, 2.5, 2.5],
    ])
    risk = estimate_horizon_breach(
        _rows(3), 12.0, event,
        measured_interval_coverage=0.8,
        calibration_is_verifiable=True,
        fallback_residuals_by_lead=selection,
    )
    assert risk["support"] == "best_effort"
    assert risk["method"] == "blocked_residual_bootstrap_v1"
    assert risk["residual_source"] == "selection_folds_reused"
    assert risk["probability_any_breach"] is not None
    assert risk["joint_path_count"] == 1
    assert risk["bootstrap_path_count"] > 0
    assert risk["effective_origins"] == 4
    assert risk["dependence_preserved"] is False
    codes = {reason["code"] for reason in risk["reasons"]}
    assert {"insufficient_joint_paths", "bootstrap_synthesized_paths",
            "selection_folds_reused"} <= codes
    again = estimate_horizon_breach(
        _rows(3), 12.0, event,
        measured_interval_coverage=0.8,
        calibration_is_verifiable=True,
        fallback_residuals_by_lead=selection,
    )
    assert again == risk  # deterministic: same inputs, same paths


def test_no_residuals_at_all_still_withholds() -> None:
    risk = estimate_horizon_breach(
        _rows(3), 12.0, {},
        measured_interval_coverage=None,
        calibration_is_verifiable=False,
    )
    assert risk["support"] == "insufficient"
    assert risk["probability_any_breach"] is None


def test_policy_separates_likelihood_from_action() -> None:
    risk = {
        "probability_any_breach": 0.4,
        "probability_any_breach_interval_90": {"lower": 0.24, "upper": 0.58},
        "breach_more_likely_than_not": False,
        "support": "supported",
    }
    decision = apply_breach_policy(risk, BreachDecisionPolicy(2, 10))
    assert decision["breach_more_likely_than_not"] is False
    assert decision["break_even_probability"] == 0.2
    assert decision["recommended_action"] == "act"
    assert decision["decision_support"] == "supported"
    assert decision["automation_eligible"] is True
    assert decision["expected_loss_if_act"] == 2
    assert decision["expected_loss_if_monitor"] == 4


def test_unresolved_boundary_demotes_but_still_recommends() -> None:
    # An interval straddling the break-even loses governed authority, but
    # the operator still receives the expected-loss recommendation at the
    # point estimate: withholding-as-monitor was measured to invert the
    # 10:2 cost asymmetry and price as the worst constant policy.
    risk = {
        "probability_any_breach": 0.3,
        "probability_any_breach_interval_90": {"lower": 0.12, "upper": 0.55},
        "breach_more_likely_than_not": False,
        "support": "supported",
    }
    decision = apply_breach_policy(risk, BreachDecisionPolicy(2, 10))
    assert decision["recommended_action"] == "act"
    assert decision["decision_support"] == "best_effort"
    assert decision["automation_eligible"] is False
    assert decision["reason_code"] == \
        "policy_boundary_unresolved_point_estimate_used"


def test_best_effort_estimates_yield_best_effort_recommendations() -> None:
    risk = {
        "probability_any_breach": 0.05,
        "probability_any_breach_interval_90": {"lower": 0.01, "upper": 0.2},
        "breach_more_likely_than_not": False,
        "support": "best_effort",
        "reasons": [{"code": "bootstrap_synthesized_paths", "message": "x"}],
    }
    decision = apply_breach_policy(risk, BreachDecisionPolicy(2, 10))
    assert decision["recommended_action"] == "monitor"
    assert decision["decision_support"] == "best_effort"
    assert decision["reason_code"] == \
        "event_estimate_not_governed_point_estimate_used"
    assert decision["event_reasons"]


def test_policy_withholds_only_without_any_probability() -> None:
    decision = apply_breach_policy(
        {"probability_any_breach": None, "support": "insufficient"},
        BreachDecisionPolicy(2, 10))
    assert decision["recommended_action"] is None
    assert decision["decision_support"] == "insufficient"
    assert decision["reason_code"] == "event_probability_unavailable"


def test_policy_refuses_invalid_or_product_invented_costs() -> None:
    with pytest.raises(ValueError, match="action_cost"):
        BreachDecisionPolicy(-1, 10).validate()
    with pytest.raises(ValueError, match="miss_cost"):
        BreachDecisionPolicy(1, 0).validate()
    with pytest.raises(ValueError, match="mitigation_effectiveness"):
        BreachDecisionPolicy(1, 10, 0).validate()


def test_threshold_evaluation_reserves_post_selection_event_origins() -> None:
    values = [100 + index * 0.5 + (index % 3) for index in range(60)]
    threshold = evaluate(
        values, 3, 1, 0.02, threshold_job=True, strict_abstention=True)
    ordinary = evaluate(
        values, 3, 1, 0.02, threshold_job=False, strict_abstention=True)
    assert threshold.event_residual_fold_count == 8
    assert all(len(items) == 8
               for items in threshold.event_residuals_by_lead.values())
    # Ordinary forecast partitioning is unchanged; the event-only reserve is
    # paid for only when the caller asks a threshold question.
    assert ordinary.event_residual_fold_count == 1
