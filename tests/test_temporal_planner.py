from __future__ import annotations

from gnomon.temporal_planner import build_evidence_plan, inference_mode
from gnomon.temporal_question import TemporalQuestion


def _result(direction: str = "higher", support: str = "weak") -> dict:
    return {
        "best_estimate": {"value": direction, "support": support},
        "answer": {"direction": direction, "support": support,
                   "estimate": 2.0, "executable": {
                       "kind": "published_forecast_projection"}},
    }


def test_planner_distinguishes_observed_predictive_and_conditional() -> None:
    assert inference_mode(TemporalQuestion(
        "a", "describe", "x", "level")) == "observed"
    assert inference_mode(TemporalQuestion(
        "b", "predict", "x", "level", horizon=4)) == "predictive"
    assert inference_mode(TemporalQuestion(
        "c", "predict", "x", "level", horizon=4,
        context_policy="scenario")) == "conditional"


def test_planner_names_missing_evidence_instead_of_changing_claim() -> None:
    question = TemporalQuestion(
        "q", "predict", "x", "volatility", horizon=20)
    result = _result("uncertain", "abstained")
    plan = build_evidence_plan(question, result)
    assert plan["authority"] == "fitted_executable"
    assert plan["missing_evidence"] == ["rolling_origin_scale_fit"]
    assert plan["recovery"] == [
        "collect more history or request a shorter horizon"]
    assert plan["identifiable"] is False
    assert plan["llm_role"] == "explain_and_qualify_only"


def test_planner_discloses_observed_predictive_contradiction() -> None:
    question = TemporalQuestion(
        "q", "predict", "*", "volatility", horizon=10,
        scope="aggregate", members=("a", "b"),
        aggregation="median_normalized_scale_ratio")
    result = _result("decreased")
    result["observed_panel_evidence"] = {
        "direction": "increased", "support": "weak",
        "effective_series": 2, "agreement": 1.0}
    plan = build_evidence_plan(question, result)
    assert result["best_estimate"]["value"] == "decreased"
    assert plan["contradictions"][0]["observed"] == "increased"
    assert plan["primary_forecast_unchanged"] is True


def test_planner_recognizes_calibrated_volatility_without_a_kind_tag() -> None:
    question = TemporalQuestion(
        "q", "predict", "x", "volatility", horizon=4)
    result = _result("stable")
    result["answer"]["executable"] = {"candidate": "constant", "horizon": 4}
    result["answer"]["property_distribution"] = {"folds": 9}
    plan = build_evidence_plan(question, result)
    assert "missing_evidence" not in plan
    assert plan["basis"] == ["rolling_origin_scale_fit"]
    assert plan["evidence"][0]["kind"] == "fitted_volatility"


def test_reasoning_pack_has_a_hard_small_shape() -> None:
    import json
    question = TemporalQuestion(
        "q", "predict", "*", "volatility", horizon=10,
        scope="aggregate", members=tuple(f"s{i}" for i in range(100)),
        aggregation="median_normalized_scale_ratio")
    result = _result("increased")
    result["observed_panel_evidence"] = {
        "direction": "increased", "support": "supported",
        "effective_series": 100, "agreement": .91}
    plan = build_evidence_plan(question, result)
    assert len(json.dumps(plan, separators=(",", ":"))) < 1400
    assert len(plan.get("evidence", [])) <= 3
    assert len(plan.get("missing_evidence", [])) <= 3
    assert len(plan.get("contradictions", [])) <= 2


def test_predictive_plan_preserves_conflicting_observed_transition() -> None:
    question = TemporalQuestion(
        "q", "predict", "x", "trend", horizon=10)
    result = _result("upward")
    observed = {
        "direction": "downward", "support": "supported",
        "identifiable": True, "estimate": -2.0,
        "diagnostics": {"window_steps": 24},
    }
    plan = build_evidence_plan(
        question, result, observed_evidence=observed)
    assert plan["contradictions"][0] == {
        "between": ["canonical_predictive_answer", "observed_transition"],
        "canonical": "upward", "observed": "downward",
        "resolution": "retain canonical answer; disclose observed disagreement",
    }
