"""Question meaning survives public compilation, execution and rendering."""

from datetime import datetime, timedelta

import pytest

from gnomon.contracts import GnomonError
from gnomon.temporal_question import compile_temporal_question
from gnomon.temporal_reasoning import answer_descriptive_question
from gnomon.toolspec import runner_for


@pytest.mark.parametrize("quantity, expected", [
    ("average", 127 / 28), ("mean", 127 / 28), ("median", 1),
    ("latest", 100), ("minimum", 1), ("maximum", 100), ("sum", 127),
])
def test_public_describe_preserves_exact_statistic(tmp_path, quantity, expected):
    values = [1.0] * 27 + [100.0]
    rows = [{"time": (datetime(2025, 1, 1) + timedelta(days=i)).isoformat(),
             "value": value} for i, value in enumerate(values)]
    result = runner_for("gnomon_describe")({
        "observations": rows, "time_column": "time", "target_column": "value",
        "questions": [{"property": quantity, "verb": "describe", "target": "value"}],
        "output_dir": str(tmp_path),
    })
    answer = result["answers"][0]
    assert answer["answer"]["estimate"] == pytest.approx(expected)
    assert answer["best_estimate"]["value"] == pytest.approx(expected)
    assert answer["answer"]["automation_eligible"] is False
    assert "calibrated" not in answer["support"]["meaning"]
    assert answer["answer"]["executable"]["measure"] == (
        "mean" if quantity == "average" else quantity)
    assert "reasoning" not in answer["answer"]


def test_quantity_alias_cannot_conflict_with_measure():
    with pytest.raises(GnomonError, match="ambiguous or unsupported"):
        compile_temporal_question({"property": "average", "measure": "latest"},
                                  available_targets=["value"])


def test_explicit_mean_comparison_does_not_compute_median():
    question = compile_temporal_question(
        {"property": "mean", "verb": "compare", "horizon": 3},
        available_targets=["value"])
    answer = answer_descriptive_question(question, report={},
        values=[1, 1, 100], season=1, forecast_values=[3, 3, 3])
    estimate = answer["answer"]["estimate"]
    assert estimate["history_mean"] == 34
    assert estimate["forecast_mean"] == 3
    assert answer["answer"]["direction"] == "lower"


def test_explicit_future_statistic_requires_a_forecast_path():
    question = compile_temporal_question(
        {"property": "mean", "verb": "predict", "horizon": 3},
        available_targets=["value"])
    answer = answer_descriptive_question(question, report={}, values=[1, 1, 100], season=1)
    assert answer["answer"]["support"] == "abstained"


def test_observed_change_does_not_substitute_latest_or_invent_windows():
    question = compile_temporal_question(
        {"property": "level", "measure": "change", "verb": "describe"},
        available_targets=["value"])
    result = answer_descriptive_question(question, report={}, values=[1, 2, 3], season=1)
    assert result["answer"]["support"] == "abstained"
    assert result["answer"]["estimate"] is None


@pytest.mark.parametrize("extra", [{"window": "last_week"}, {"unit": "kg"}, {"aggregation": "mean"},
                                  {"comparison": {"left": "last_week", "right": "last_month"}},
                                  {"validation": {"folds": 100}}, {"period": 12},
                                  {"target": {"kind": "series", "members": ["value"], "unit": "kg"}},
                                  {"target": {"kind": "each", "members": ["value", "value"]}}])
def test_legacy_question_rejects_scope_it_cannot_execute(extra):
    with pytest.raises(GnomonError):
        compile_temporal_question({"property": "mean", "verb": "describe", **extra}, available_targets=["value"])


def test_observed_subquestion_does_not_inherit_forecast_horizon():
    question = compile_temporal_question({"property": "mean", "verb": "describe"},
                                         available_targets=["value"], default_verb="predict", default_horizon=7)
    assert question.horizon is None
    result = answer_descriptive_question(question, report={}, values=[1, 2, 3], season=1)
    assert result["best_estimate"]["value"] == 2
    assert result["action_authorized"] is False
    assert not result["best_estimate"]["automation_eligible"]
    assert "calibrated" not in result["support"]["meaning"]


def test_explicit_observed_future_horizon_is_a_conflict_not_permission():
    with pytest.raises(GnomonError):
        compile_temporal_question({"property": "mean", "verb": "describe", "horizon": 7}, available_targets=["value"])


def test_a_supported_fallback_is_not_a_calibration_or_permission_receipt():
    question = compile_temporal_question({"property": "extreme", "verb": "describe"}, available_targets=["value"])
    result = answer_descriptive_question(question, report={"level": {"minimum": 1, "maximum": 3}}, values=[1, 2, 3], season=1)
    assert result["support"]["state"] == "supported"
    assert not result["answer"]["automation_eligible"] and not result["action_authorized"]
