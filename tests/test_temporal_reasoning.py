from __future__ import annotations

from gnomon.temporal_question import TemporalQuestion
from gnomon.temporal_reasoning import (
    _seasonality_alignment, answer_descriptive_question,
    answer_scoped_question,
)


REPORT = {
    "level": {"latest": 14.0, "minimum": 10.0, "maximum": 14.0},
    "trend": {"slope_per_step": 1.0, "direction": "up"},
    "seasonality": {"period": 2},
    "changepoints": [],
}


def test_seasonality_alignment_distinguishes_fixed_shifted_and_flat() -> None:
    import math
    season = 24
    history = [10 + 4 * math.sin(2 * math.pi * index / season)
               for index in range(8 * season)]
    fixed = [10 + 4 * math.sin(2 * math.pi * (len(history) + index) / season)
             for index in range(2 * season)]
    shifted = [10 + 4 * math.sin(
        2 * math.pi * (len(history) + index - 7) / season)
               for index in range(2 * season)]

    assert _seasonality_alignment(history, fixed, season)["direction"] == "continued"
    shift = _seasonality_alignment(history, shifted, season)
    assert shift["direction"] == "shifting"
    assert shift["phase_shift_steps"] in {7, 17}
    assert _seasonality_alignment(history, [10.0] * 48, season)[
        "direction"] == "absent"


def test_level_comparison_returns_weak_direction_from_immutable_path() -> None:
    answer = answer_descriptive_question(
        TemporalQuestion("q", "compare", "x", "level"),
        report=REPORT, values=[10.0, 12.0, 14.0], season=2,
        forecast_values=[15.0, 17.0],
    )
    assert answer["answer"]["estimate"] == {
        "history_median": 12.0, "forecast_median": 16.0,
        "absolute_change": 4.0, "relative_change": 1 / 3,
    }
    assert answer["answer"]["direction"] == "higher"
    assert answer["answer"]["support"] == "weak"
    assert answer["answer"]["automation_eligible"] is False
    assert "threshold" in answer["limitations"][0]


def test_seasonality_comparison_reports_alignment_and_weak_best_label() -> None:
    answer = answer_descriptive_question(
        TemporalQuestion("q", "compare", "x", "seasonality"),
        report=REPORT, values=[1.0, 3.0, 1.0, 3.0], season=2,
        forecast_values=[1.0, 3.0, 1.0, 3.0],
    )
    assert answer["answer"]["estimate"][
        "forecast_vs_repeated_history_correlation"] == 1.0
    assert answer["answer"]["direction"] == "continued"
    assert answer["answer"]["support"] == "weak"


def test_predictive_properties_use_fitted_executable_without_forecast() -> None:
    values = [20 + .1 * index + (index % 7) for index in range(300)]
    for prop in ("level", "trend", "seasonality", "regime", "extreme"):
        answer = answer_descriptive_question(
            TemporalQuestion("q", "predict", "x", prop, horizon=14),
            report=REPORT, values=values, season=7)
        assert answer["answer"]["executable"]["kind"] == "fitted_temporal_property"


def test_volatility_exposes_distribution_and_separate_decision_policy() -> None:
    values = [20 + .1 * index + (index % 7) for index in range(300)]
    answer = answer_descriptive_question(
        TemporalQuestion("q", "predict", "x", "volatility", horizon=14),
        report=REPORT, values=values, season=7)
    distribution = answer["answer"]["property_distribution"]
    assert distribution["quantity"] == \
        "future_to_reference_residual_scale_ratio"
    assert abs(sum(distribution["probabilities"].values()) - 1) < 1e-9
    assert answer["answer"]["decision"]["automation_eligible"] is False
    assert answer["calibration"]["direction_candidate"]


def test_pair_dependence_routes_to_fitted_executable() -> None:
    question = TemporalQuestion(
        "q", "predict", "*", "dependence", horizon=8,
        scope="pair", members=("x", "y"))
    values = [float(index + index % 3) for index in range(240)]
    answer = answer_scoped_question(
        question, reports={"x": REPORT, "y": REPORT},
        execution_inputs={"x": (values, 1), "y": (values, 1)})
    assert answer["answer"]["executable"]["kind"] == "fitted_dependence"
    assert answer["calibration"]["primary_forecast_unchanged"] is True
    assert answer["answer"]["reasoning"]["mode"] == "predictive"
    assert answer["answer"]["reasoning"]["llm_role"] == \
        "explain_and_qualify_only"


def test_answer_exposes_canonical_and_explicit_display_vocabulary() -> None:
    question = TemporalQuestion(
        "q", "compare", "x", "level",
        answer_vocabulary={"higher": "Above history"})
    answer = answer_descriptive_question(
        question, report=REPORT, values=[10.0, 12.0, 14.0], season=2,
        forecast_values=[15.0, 17.0])
    assert answer["best_estimate"] == {
        "value": "higher", "display_value": "Above history",
        "support": "weak", "automation_eligible": False,
    }
    assert answer["decision_rule"]["kind"] == "published_forecast_projection"


def test_seasonality_aggregate_returns_one_quotable_answer() -> None:
    question = TemporalQuestion(
        "q", "compare", "*", "seasonality", scope="aggregate",
        members=("x", "y"), aggregation="median_alignment")
    answer = answer_scoped_question(
        question, reports={"x": REPORT, "y": REPORT},
        execution_inputs={
            "x": ([1.0, 3.0, 1.0, 3.0], 2),
            "y": ([2.0, 4.0, 2.0, 4.0], 2),
        }, forecast_values={
            "x": [1.0, 3.0, 1.0, 3.0],
            "y": [2.0, 4.0, 2.0, 4.0],
        })
    assert answer["best_estimate"]["value"] == "continued"
    assert answer["answer"]["estimate"]["contributing_series"] == 2
    assert answer["decision_rule"]["aggregation"] == "median_alignment"


def test_volatility_aggregate_preserves_observed_panel_evidence() -> None:
    import random
    question = TemporalQuestion(
        "q", "predict", "*", "volatility", horizon=69,
        scope="aggregate", members=tuple(f"x{i}" for i in range(6)))
    inputs = {}
    reports = {}
    for seed in range(6):
        rng = random.Random(seed)
        values = ([rng.gauss(0, 1) for _ in range(48)]
                  + [rng.gauss(0, 3) for _ in range(48)])
        inputs[f"x{seed}"] = (values, 1)
        reports[f"x{seed}"] = REPORT
    answer = answer_scoped_question(
        question, reports=reports, execution_inputs=inputs)
    evidence = answer["observed_panel_evidence"]
    assert evidence["direction"] == "increased"
    assert evidence["effective_series"] >= 5
    assert answer["identifiability"]["future_direction"] == \
        "conditional_on_persistence"
    assert answer["identifiability"]["primary_forecast_unchanged"] is True
