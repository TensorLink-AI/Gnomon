import pytest

from gnomon.temporal_distribution import bounded_decision_policy
from gnomon.temporal_question import compile_temporal_question


def test_named_profiles_are_bounded_and_ordered():
    exploratory = bounded_decision_policy("exploratory")
    conservative = bounded_decision_policy("conservative")
    assert exploratory.minimum_probability == .7
    assert conservative.minimum_folds > exploratory.minimum_folds


@pytest.mark.parametrize("raw", [
    {"minimum_probability": .1}, {"minimum_folds": 1},
    {"minimum_balanced_accuracy": .2}, {"unknown": 2}, "anything",
])
def test_policy_cannot_weaken_below_public_evidence_floor(raw):
    with pytest.raises(ValueError):
        bounded_decision_policy(raw)


def test_question_compiler_returns_repairable_failure_for_bad_policy():
    with pytest.raises(Exception) as caught:
        compile_temporal_question({"verb": "predict", "property": "volatility",
                                   "target": "x", "decision_policy": {
                                       "minimum_probability": .2}},
                                  available_targets=["x"])
    assert "decision_policy" in caught.value.details["fields"]
