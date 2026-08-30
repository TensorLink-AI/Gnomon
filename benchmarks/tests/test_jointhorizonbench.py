from __future__ import annotations

from benchmarks.jointhorizonbench.run import _immutable_surface, _independence


def test_immutable_surface_excludes_only_candidate_cumulative_block() -> None:
    packet = {
        "forecast": [{"q10": 1, "q50": 2, "q90": 3}],
        "threshold_analysis": {
            "probability_above_per_step": [.25],
            "horizon_event": {
                "probability_any_breach": .4,
                "support": "best_effort",
                "cumulative_horizon": {"status": "available"},
            },
        },
        "governed_decision": {"advisory_action": "act"},
    }
    surface = _immutable_surface(packet)
    assert surface["horizon_event"] == {
        "probability_any_breach": .4,
        "support": "best_effort",
    }
    assert surface["forecast"] == packet["forecast"]
    assert surface["probability_above_per_step"] == [.25]
    assert surface["governed_decision"] == packet["governed_decision"]


def test_independence_diagnostic_is_mechanical_and_bounded() -> None:
    assert _independence([]) is None
    assert _independence([.25, .25]) == 1.0 - .75**2
    assert _independence([-2, 2]) == 1.0
