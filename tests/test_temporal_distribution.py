from __future__ import annotations

import pytest

from gnomon.temporal_distribution import (
    TemporalDecisionPolicy,
    TemporalPropertyDistribution,
)


def _distribution(**overrides):
    values = {
        "quantity": "future_to_reference_scale_ratio",
        "estimate": 1.4, "lower": 1.1, "upper": 1.8,
        "probabilities": {"decreased": .05, "stable": .1, "increased": .85},
        "point_state": "increased", "folds": 12,
        "balanced_accuracy": .7, "brier_skill": .15,
        "support": "supported",
    }
    values.update(overrides)
    return TemporalPropertyDistribution(**values)


def test_distribution_only_automates_when_every_policy_gate_passes() -> None:
    assert _distribution().decide(TemporalDecisionPolicy())["automation_eligible"]
    assert not _distribution(brier_skill=0).decide(
        TemporalDecisionPolicy())["automation_eligible"]
    assert not _distribution(probabilities={
        "decreased": .1, "stable": .2, "increased": .7,
    }).decide(TemporalDecisionPolicy())["automation_eligible"]


def test_weak_distribution_keeps_point_state_and_discloses_probability_mode() -> None:
    decision = _distribution(
        probabilities={"decreased": .7, "stable": .2, "increased": .1},
        point_state="stable", brier_skill=0,
    ).decide(TemporalDecisionPolicy())
    assert decision["best_state"] == "stable"
    assert decision["probability_mode"] == "decreased"


def test_distribution_rejects_invalid_probability_contract() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        _distribution(probabilities={"stable": .9, "increased": .9})
