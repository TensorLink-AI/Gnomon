import math
import random

import pytest

from gnomon.statistical_executables import (
    fit_decomposition_executable,
    fit_regression_executable,
    fit_stationarity_executable,
)
from gnomon.temporal_contracts import (
    classify_dataset_contract,
    plan_execution,
)
from gnomon.temporal_question import compile_temporal_question
from gnomon.temporal_reasoning import answer_scoped_question


def test_dataset_contract_distinguishes_wide_waveform_and_multivariate():
    wide = classify_dataset_contract([f"t_{index}" for index in range(20)])
    multi = classify_dataset_contract(["temperature", "humidity"])
    assert wide.shape == "wide_waveform"
    assert multi.shape == "multivariate"


def test_planner_refuses_semantic_method_substitution():
    question = compile_temporal_question({
        "id": "q1", "verb": "decompose", "property": "decomposition",
        "target": "solar", "method": "stl", "period": 365,
    }, available_targets=["solar"])
    dataset = classify_dataset_contract(["solar"])
    plan = plan_execution(question, dataset)
    assert plan.status == "unsupported"
    assert "does not implement requested method stl" in str(plan.reason)

    regression = compile_temporal_question({
        "id": "linear", "verb": "regress", "property": "regression",
        "target": "load", "explanatory_variables": ["weather"],
        "method": "linear",
    }, available_targets=["load", "weather"])
    regression_plan = plan_execution(
        regression, classify_dataset_contract(["load", "weather"]))
    assert regression_plan.status == "unsupported"
    assert "ridge_linear" in str(regression_plan.reason)


def test_adf_separates_stationary_noise_from_random_walk():
    rng = random.Random(404)
    stationary = [rng.gauss(0, 1) for _ in range(400)]
    walk, level = [], 0.0
    for _ in range(400):
        level += rng.gauss(0, 1)
        walk.append(level)
    stationary_result = fit_stationarity_executable(
        stationary, target="x", method="adf").execute()
    walk_result = fit_stationarity_executable(
        walk, target="x", method="adf").execute()
    assert stationary_result["direction"] == "stationary"
    assert walk_result["direction"] == "unit_root_not_rejected"


def test_fixed_period_decomposition_recovers_period_and_strength():
    values = [10 + 3 * math.sin(2 * math.pi * index / 12) + .01 * index
              for index in range(120)]
    result = fit_decomposition_executable(
        values, target="load", period=12).execute()
    assert result["estimate"]["period"] == 12
    assert result["estimate"]["seasonal_strength"] > .9
    assert result["executable"]["method"] == "centered_moving_average_additive"


def test_exogenous_regression_is_expanding_window_and_recovers_coefficients():
    rng = random.Random(405)
    x1 = [rng.uniform(-2, 2) for _ in range(180)]
    x2 = [rng.uniform(-1, 1) for _ in range(180)]
    y = [4 + 2 * a - 3 * b + rng.gauss(0, .1) for a, b in zip(x1, x2)]
    result = fit_regression_executable(
        y, {"x1": x1, "x2": x2}, target="y").execute()
    assert result["support"] == "supported"
    assert result["estimate"]["validation"]["scheme"] == \
        "expanding_window_one_step"
    coefficients = result["estimate"]["coefficients"]
    assert coefficients["x1"] == pytest.approx(2, abs=.05)
    assert coefficients["x2"] == pytest.approx(-3, abs=.05)


def test_scoped_runtime_publishes_stationarity_through_common_envelope():
    rng = random.Random(406)
    values = [rng.gauss(0, 1) for _ in range(200)]
    question = compile_temporal_question({
        "id": "stationarity", "verb": "test", "property": "stationarity",
        "target": "value", "method": "adf",
    }, available_targets=["value"])
    answer = answer_scoped_question(
        question,
        reports={"value": {"frequency": "daily"}},
        execution_inputs={"value": (values, 1)},
    )
    assert answer["best_estimate"]["value"] == "stationary"
    assert answer["answer"]["executable"]["kind"] == \
        "fitted_stationarity_test"
    assert answer["execution_plan"]["status"] == "ready"
    assert answer["answer"]["reasoning"]["authority"] == "fitted_executable"


def test_scoped_runtime_returns_one_terminal_unsupported_answer():
    values = [float(index) for index in range(800)]
    question = compile_temporal_question({
        "id": "stl", "verb": "decompose", "property": "decomposition",
        "target": "value", "method": "stl", "period": 365,
    }, available_targets=["value"])
    answer = answer_scoped_question(
        question,
        reports={"value": {"frequency": "daily"}},
        execution_inputs={"value": (values, 365)},
    )
    assert answer["support"]["state"] == "abstained"
    assert "No substitute operation was run" in answer["headline"]
    assert len(answer["next_actions"]) == 1
