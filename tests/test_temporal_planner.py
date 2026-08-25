from __future__ import annotations

from gnomon.temporal_planner import (
    build_evidence_plan, compact_evidence_plan, inference_mode,
)
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
    # The full immutable receipt includes bounded adjudication provenance
    # and, since the two-lane decision, the evidence dossier (packet);
    # only its compact projection is paid on every agent turn.
    assert len(json.dumps(plan, separators=(",", ":"))) < 3000
    assert len(json.dumps(compact_evidence_plan(plan),
                          separators=(",", ":"))) < 850
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
    assert plan["contrast"]["against"][0]["direction"] == "downward"
    assert plan["suggested_next"]
    assert plan["adjudication"]["relationship"] == "alternative_preferred"
    assert plan["adjudication"]["primary_forecast_unchanged"] is True


def test_analogue_evidence_is_contrastive_not_authoritative() -> None:
    question = TemporalQuestion(
        "q", "predict", "x", "trend", horizon=10)
    result = _result("upward", "supported")
    analogues = {
        "version": "0.1", "available": True, "window_steps": 12,
        "consensus_direction": "downward", "agreement": .67,
        "matches": [{"state_end_offset": -24, "distance": .1,
                     "outcome_direction": "downward",
                     "outcome_support": "weak"}],
    }
    plan = build_evidence_plan(question, result, analogues=analogues)
    assert plan["authority"] == "fitted_executable"
    assert result["best_estimate"]["value"] == "upward"
    assert plan["contrast"]["against"] == [{
        "evidence": "historical_analogues", "direction": "downward",
        "agreement": .67,
    }]
    assert plan["primary_forecast_unchanged"] is True


def test_compact_projection_keeps_reasoning_not_receipt_bulk() -> None:
    plan = {
        "version": "0.2", "authority": "fitted_executable",
        "primary_forecast_unchanged": True,
        "evidence": [{"kind": "large", "values": list(range(100))}],
        "historical_analogues": {"matches": list(range(20))},
        "contrast": {
            "because": [{"evidence": str(i)} for i in range(4)],
            "against": [{"evidence": str(i)} for i in range(3)],
            "unknown": ["a", "b", "c"],
        },
        "suggested_next": ["observe", "refit"],
        "adjudication": {
            "relationship": "canonical_preferred",
            "candidates": [
                {"value": "stable", "support": "weak",
                 "evidence_weight": .7, "conditional_only": False},
                {"value": "increased", "support": "weak",
                 "evidence_weight": .4, "conditional_only": True},
            ],
            "synthesis_eligibility": {"eligible": False},
            "what_would_flip": ["more outcomes"],
        },
    }
    compact = compact_evidence_plan(plan)
    assert len(compact["because"]) == 1
    assert len(compact["against"]) == 1
    assert compact["unknown"] == ["a", "b"]
    assert compact["next"] == ["observe"]
    assert "evidence" not in compact
    assert "historical_analogues" not in compact
    assert compact["details_in_answer_receipt"] is True
    hypotheses = compact["adjudication"]["ranked_hypotheses"]
    assert [row["value"] for row in hypotheses] == ["stable", "increased"]
    assert compact["adjudication"]["weight_meaning"].endswith(
        "not probability")

    import json
    assert len(json.dumps(compact)) < 1600
    assert compact["adjudication"]["relationship"] == "canonical_preferred"
