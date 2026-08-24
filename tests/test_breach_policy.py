from __future__ import annotations

import pytest

from gnomon.breach import (
    BreachDecisionPolicy,
    aligned_residual_paths,
    apply_breach_policy,
    estimate_horizon_breach,
)
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


def test_regime_change_withholds_probability_support() -> None:
    paths = [[0.0, 0.0] for _ in range(8)] + [
        [20.0, 20.0] for _ in range(8)
    ]
    risk = estimate_horizon_breach(
        _rows(2), 12.0, _by_lead(paths),
        measured_interval_coverage=0.8,
        calibration_is_verifiable=True,
    )
    assert risk["support"] == "insufficient"
    assert risk["residual_regime"]["status"] == "changed"
    assert "residual_regime_changed" in {
        reason["code"] for reason in risk["reasons"]
    }


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
    assert decision["expected_loss_if_act"] == 2
    assert decision["expected_loss_if_monitor"] == 4


def test_policy_withholds_when_interval_crosses_break_even() -> None:
    risk = {
        "probability_any_breach": 0.2,
        "probability_any_breach_interval_90": {"lower": 0.08, "upper": 0.38},
        "breach_more_likely_than_not": False,
        "support": "supported",
    }
    decision = apply_breach_policy(risk, BreachDecisionPolicy(2, 10))
    assert decision["recommended_action"] is None
    assert decision["decision_support"] == "insufficient"
    assert decision["reason_code"] == "policy_boundary_not_resolved"


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
