"""The bridge to TSAIA's own evaluators.

These tests run TSAIA's real ``gen_eval.py``, fetched and pinned. They are
skipped when it cannot be fetched, because a network failure is not a
regression in this repository.

The assertions are deliberately about *their* criteria, not ours. Each one
that pins a specific number is documenting a place where the harness used to
invent its own bar and got it wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from aionbench.contracts import BenchError
from aionbench import tsaia_eval


@pytest.fixture(scope="module", autouse=True)
def gen_eval():
    try:
        return tsaia_eval.load_gen_eval()
    except BenchError as exc:
        pytest.skip(f"TSAIA evaluator unavailable: {exc.code}")


# -- provenance ------------------------------------------------------------

def test_evaluator_is_pinned_and_checksummed():
    described = tsaia_eval.describe()
    assert described["source"].endswith(tsaia_eval.TSAIA_COMMIT)
    assert described["sha256"] == tsaia_eval.TSAIA_GEN_EVAL_SHA256
    # Not vendored: their repo has no LICENSE, so we may not redistribute it.
    assert described["fetched_not_vendored"] is True


def test_checksum_mismatch_refuses_to_grade(monkeypatch, tmp_path):
    monkeypatch.setattr(tsaia_eval, "_module", None)
    monkeypatch.setattr(tsaia_eval, "TSAIA_GEN_EVAL_SHA256", "0" * 64)
    with pytest.raises(BenchError, match="does not match the pinned checksum"):
        tsaia_eval.load_gen_eval()
    monkeypatch.setattr(tsaia_eval, "_module", None)


# -- registry --------------------------------------------------------------

def test_every_registered_question_type_resolves(gen_eval):
    for question_type in gen_eval.Question_Type_MAP:
        assert tsaia_eval.evaluator_for(question_type) is gen_eval.Question_Type_MAP[question_type][1]


def test_sub_case_suffix_is_stripped_before_lookup(gen_eval):
    assert tsaia_eval.evaluator_for("easy_stock-future price").__name__ == "Easy_Stock_Evaluator"
    assert tsaia_eval.evaluator_for("climate_anomaly-point").__name__ == "Anomaly_Evaluator"


def test_unknown_question_type_is_a_typed_error():
    with pytest.raises(BenchError, match="no evaluator"):
        tsaia_eval.evaluator_for("something_invented")


@pytest.mark.parametrize("question_type,metric", [
    ("easy_stock-future price", "mape"),
    ("easy_stock-future volatility", "mape"),
    ("easy_stock-future trend", "accuracy"),
    ("climate_anomaly-x", "f1"),
    ("causal_relation-y", "accuracy"),
    ("stock_rv_estimation-z", "absolute_diff"),
    ("stock_investment", "result"),
])
def test_sub_metric_names_match_their_map(question_type, metric):
    assert tsaia_eval.metric_name_for(question_type) == metric


# -- the criteria we previously got wrong ----------------------------------

def test_mape_bar_defaults_to_1_0_not_a_number_we_chose():
    """A 30% forecast error passes TSAIA. The harness used to fail it at 0.15."""
    truth = np.array([[100.0]] * 9)
    prediction = np.array([[130.0]] * 9)
    result = tsaia_eval.evaluate(
        "easy_stock-future price", response=prediction, ground_truth=truth,
        context={"names": ["AAA"]}, constraint={"future price": True},
    )
    assert result["status"] == 1
    assert result["mape"] == pytest.approx(0.30, abs=1e-3)


def test_the_bar_is_read_from_the_questions_own_constraint():
    truth = np.array([[100.0]] * 9)
    prediction = np.array([[130.0]] * 9)
    tightened = tsaia_eval.evaluate(
        "easy_stock-future price", response=prediction, ground_truth=truth,
        context={"names": ["AAA"]}, constraint={"future price": True, "threshold": 0.15},
    )
    assert tightened["status"] == 0
    assert tightened["mape"] == pytest.approx(0.30, abs=1e-3)


def test_anomaly_f1_bar_is_1e_minus_6_so_any_hit_passes():
    """Their bar is 'found something'. The harness used to demand F1 >= 0.5."""
    truth = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0])
    barely = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    result = tsaia_eval.evaluate(
        "climate_anomaly-x", response=barely, ground_truth=truth, context={}, constraint={},
    )
    assert result["status"] == 1
    assert result["f1"] > 0


def test_anomaly_f1_is_point_adjusted_not_raw_pointwise():
    """One detected point inside an event credits the whole event.

    This is the standard time-series anomaly convention TSAIA uses. Raw
    pointwise F1 — what the harness computed before — gives a much lower
    number for the same prediction.
    """
    truth = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    one_point = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    adjusted = tsaia_eval.evaluate(
        "climate_anomaly-x", response=one_point.copy(), ground_truth=truth,
        context={}, constraint={},
    )
    assert adjusted["f1"] == pytest.approx(1.0)

    from aionbench.metrics import binary_f1
    assert binary_f1(truth.tolist(), one_point.tolist()).f1 == pytest.approx(0.4)


def test_detecting_nothing_fails():
    truth = np.array([0, 1, 1, 0, 0])
    result = tsaia_eval.evaluate(
        "climate_anomaly-x", response=np.zeros(5, dtype=int), ground_truth=truth,
        context={}, constraint={},
    )
    assert result["status"] == 0


def test_trend_passes_on_any_correct_answer():
    """`if acc > 0` — one right out of three is a pass upstream."""
    truth = np.array([[1.0, 10.0, 5.0], [2.0, 9.0, 5.0], [3.0, 8.0, 5.0], [4.0, 7.0, 5.0]])
    result = tsaia_eval.evaluate(
        "easy_stock-future trend",
        response=["INCREASING", "INCREASING", "INCREASING"],
        ground_truth=truth, context={"names": ["A", "B", "C"]},
        constraint={"future trend": True},
    )
    assert result["status"] == 1
    assert 0 < result["accuracy"] < 1


# -- failure handling ------------------------------------------------------

def test_an_evaluator_exception_scores_zero_rather_than_propagating():
    result = tsaia_eval.evaluate(
        "easy_stock-future price", response=None, ground_truth=np.array([[1.0]]),
        context={"names": ["A"]}, constraint={"future price": True},
    )
    assert result["status"] == 0


def test_a_missing_answer_scores_zero():
    """A model that never bound `predictions` hands the evaluator None."""
    result = tsaia_eval.evaluate(
        "climate_anomaly-x", response=None, ground_truth=np.array([0, 1, 0]),
        context={}, constraint={},
    )
    assert result["status"] == 0


def test_wrong_shape_scores_zero_rather_than_being_truncated():
    truth = np.array([[100.0]] * 9)
    short = np.array([[100.0]] * 4
                     )
    result = tsaia_eval.evaluate(
        "easy_stock-future price", response=short, ground_truth=truth,
        context={"names": ["AAA"]}, constraint={"future price": True},
    )
    assert result["status"] == 0


# -- problematic-question skipping ----------------------------------------

def test_nan_ground_truth_is_problematic():
    assert tsaia_eval.is_problematic("climate_anomaly-x", np.array([1.0, np.nan]), {}) is True


def test_near_zero_electricity_ground_truth_is_problematic():
    assert tsaia_eval.is_problematic("electricity_prediction-x", np.array([1.0, 1e-9]), {}) is True
    assert tsaia_eval.is_problematic("electricity_prediction-x", np.array([1.0, 5.0]), {}) is False


def test_clean_ground_truth_is_not_problematic():
    assert tsaia_eval.is_problematic("climate_anomaly-x", np.array([0.0, 1.0]), {}) is False


def test_non_numeric_ground_truth_is_not_problematic():
    """Causal tasks key on matrices/dicts their NaN check cannot handle."""
    assert tsaia_eval.is_problematic("causal_relation-x", {"a": "b"}, {}) is False
