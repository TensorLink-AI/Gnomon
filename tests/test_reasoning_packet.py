from __future__ import annotations

from gnomon.reasoning_packet import verify_packet_selection
from gnomon.temporal_planner import build_evidence_plan, compact_evidence_plan
from gnomon.temporal_question import TemporalQuestion


def _result(direction: str = "upward", support: str = "weak") -> dict:
    return {
        "best_estimate": {"value": direction, "support": support},
        "answer": {"direction": direction, "support": support,
                   "estimate": 2.0, "interval": None,
                   "executable": {"kind": "published_forecast_projection"}},
    }


def _plan(support: str = "weak", vocabulary: dict | None = None) -> dict:
    question = TemporalQuestion(
        "q", "predict", "x", "trend", horizon=10,
        answer_vocabulary=vocabulary)
    observed = {
        "direction": "downward", "support": "supported",
        "identifiable": True, "estimate": -2.0,
        "diagnostics": {"window_steps": 24},
    }
    return build_evidence_plan(
        question, _result("upward", support), observed_evidence=observed)


def test_packet_presents_interpretations_and_evidence_not_a_command() -> None:
    packet = _plan()["packet"]
    assert packet["selection_contract"]["selector"] == "model"
    assert packet["selection_contract"]["canonical"] == {
        "value": "upward", "support": "weak", "role": "default_not_command"}
    values = {row["value"]: row for row in packet["interpretations"]}
    assert values["upward"]["compatible"] is True
    assert values["downward"]["compatible"] is True
    assert "observed_transition" in values["downward"]["supporting"]
    assert "observed_transition" in values["upward"]["conflicting"]
    assert packet["evidence_sufficiency"]["level"] == "mixed"
    assert packet["observations"]["direction"] == "upward"
    assert packet["temporal_properties"] == {
        "property": "trend", "inference_mode": "predictive"}
    assert packet["discriminators"]
    assert packet["selection_contract"]["primary_forecast_unchanged"] is True


def test_a_supported_canonical_stays_binding() -> None:
    question = TemporalQuestion("q", "predict", "x", "trend", horizon=10)
    result = _result("upward", "supported")
    result["answer"]["property_distribution"] = {"folds": 6}
    result["answer"]["executable"] = {"kind": "fitted_temporal_property"}
    plan = build_evidence_plan(question, result)
    packet = plan["packet"]
    assert packet["selection_contract"]["selector"] == "gnomon_canonical"
    assert packet["selection_contract"]["canonical"]["role"] == "binding"
    violations = verify_packet_selection(packet, {"value": "downward"})
    assert any(item["code"] == "SELECTION_OVERRIDES_BINDING"
               for item in violations)
    assert verify_packet_selection(packet, {"value": "upward"}) == []


def test_supported_observation_does_not_bind_a_predictive_answer() -> None:
    question = TemporalQuestion(
        "q", "predict", "x", "trend", horizon=10,
        answer_vocabulary={
            "upward": "Up", "downward": "Down", "constant": "Flat"})
    result = _result("upward", "supported")
    result["answer"]["executable"] = {
        "kind": "observed_multi_resolution_windows"}
    packet = build_evidence_plan(question, result)["packet"]

    assert packet["selection_contract"]["selector"] == "model"
    assert packet["selection_contract"]["canonical"] == {
        "value": "upward", "support": "supported",
        "role": "default_not_command"}
    assert packet["selection_contract"]["inference_authority"] == {
        "mode": "predictive",
        "requirements_satisfied": False,
        "missing_evidence": ["rolling_origin_property_fit"],
    }
    interpretations = {
        row["value"]: row for row in packet["interpretations"]}
    assert all(row["compatible"] for row in interpretations.values())
    assert not any(
        item["code"] == "SELECTION_OVERRIDES_BINDING"
        for item in verify_packet_selection(
            packet, {"value": "downward", "cited_evidence": []}))


def test_discrimination_only_interpretation_is_selectable_in_model_lane() -> None:
    question = TemporalQuestion("q", "predict", "x", "trend", horizon=10)
    packet = build_evidence_plan(
        question, _result("uncertain", "abstained"),
        discrimination={
            "identifiable": True,
            "best": "downward",
            "separation": "clear",
            "holdout_steps": 12,
            "hypotheses": [{"value": "downward", "relative_weight": .9}],
        })["packet"]

    row = next(item for item in packet["interpretations"]
               if item["value"] == "downward")
    assert row["decision_eligible"] is True
    assert packet["selection_contract"]["canonical"]["role"] == \
        "default_not_command"
    assert verify_packet_selection(packet, {
        "value": "downward",
        "cited_evidence": ["held_out_hypothesis_fit"],
    }) == []


def test_a_selection_outside_the_packet_is_rejected() -> None:
    packet = _plan()["packet"]
    violations = verify_packet_selection(
        packet, {"value": "sideways", "cited_evidence": []})
    assert violations[0]["code"] == "SELECTION_NOT_IN_PACKET"


def test_an_interpretation_excluded_by_supported_evidence_is_incompatible() -> None:
    vocabulary = {"upward": "Up", "downward": "Down", "constant": "Flat"}
    question = TemporalQuestion(
        "q", "describe", "x", "trend", answer_vocabulary=vocabulary)
    observed = {
        "direction": "downward", "support": "supported",
        "identifiable": True, "estimate": -2.0,
        "diagnostics": {"window_steps": 24},
    }
    packet = build_evidence_plan(
        question, _result("upward", "weak"),
        observed_evidence=observed)["packet"]
    values = {row["value"]: row for row in packet["interpretations"]}
    assert values["constant"]["compatible"] is False
    violations = verify_packet_selection(
        packet, {"value": "constant",
                 "cited_evidence": ["observed_transition"]})
    codes = {item["code"] for item in violations}
    assert "SELECTION_INCOMPATIBLE" in codes
    assert "SELECTION_EVIDENCE_CONTRARY" in codes


def test_a_non_binding_selection_must_cite_packet_evidence() -> None:
    packet = _plan()["packet"]
    assert verify_packet_selection(packet, {"value": "downward"}) == [{
        "code": "SELECTION_UNCITED",
        "message": ("A non-binding selection must cite the packet evidence "
                    "it rests on."),
    }]
    missing = verify_packet_selection(
        packet, {"value": "downward", "cited_evidence": ["tea_leaves"]})
    assert missing[0]["code"] == "SELECTION_EVIDENCE_MISSING"
    assert verify_packet_selection(
        packet, {"value": "downward",
                 "cited_evidence": ["observed_transition"]}) == []


def test_sufficiency_names_what_the_evidence_can_and_cannot_settle() -> None:
    question = TemporalQuestion("q", "predict", "x", "volatility", horizon=20)
    abstained = build_evidence_plan(question, _result("uncertain", "abstained"))
    assert abstained["packet"]["evidence_sufficiency"]["level"] == "insufficient"
    assert abstained["packet"]["evidence_sufficiency"]["missing_evidence"]

    supported = TemporalQuestion("q", "predict", "x", "trend", horizon=4)
    result = _result("upward", "supported")
    result["answer"]["property_distribution"] = {"folds": 6}
    result["answer"]["executable"] = {"kind": "fitted_temporal_property"}
    plan = build_evidence_plan(supported, result)
    assert plan["packet"]["evidence_sufficiency"]["level"] == "sufficient"


def test_compact_projection_carries_the_dossier_within_bounds() -> None:
    plan = _plan()
    compact = compact_evidence_plan(plan)
    packet = compact["packet"]
    assert len(packet["interpretations"]) <= 4
    assert set(packet["interpretations"][0]) == {
        "value", "support", "compatible", "decision_eligible",
        "supporting", "conflicting"}
    assert packet["selection_must_cite_evidence"] is True
    assert packet["sufficiency"] == "mixed"
    assert packet["selector"] == "model"
    assert packet["requirements_satisfied"] is False
    assert packet["discriminator"] is not None


def test_repair_selection_accepts_a_grounded_conclusion() -> None:
    from gnomon.reasoning_packet import repair_selection

    packet = _plan()["packet"]
    verdict = repair_selection(
        packet, {"value": "downward",
                 "cited_evidence": ["observed_transition"]})
    assert verdict == {"accepted": True, "violations": []}


def test_repair_selection_builds_one_complete_repair_turn() -> None:
    from gnomon.reasoning_packet import MAX_REPAIR_ROUNDS, repair_selection

    packet = _plan()["packet"]
    verdict = repair_selection(
        packet, {"value": "downward", "cited_evidence": ["tea_leaves"]})
    assert verdict["accepted"] is False
    assert verdict["violations"][0]["code"] == "SELECTION_EVIDENCE_MISSING"
    repair = verdict["repair"]
    assert repair["rounds"] == MAX_REPAIR_ROUNDS == 1
    assert "downward" in repair["allowed_values"]
    assert "observed_transition" in repair["citable_evidence"]["downward"]
    assert repair["canonical_default"] == {"value": "upward",
                                           "support": "weak"}
    assert repair["after_failed_repair"] == \
        "publish_canonical_default_labelled"


def test_repair_of_a_binding_override_restates_the_canonical() -> None:
    from gnomon.reasoning_packet import repair_selection

    question = TemporalQuestion("q", "predict", "x", "trend", horizon=10)
    result = _result("upward", "supported")
    result["answer"]["property_distribution"] = {"folds": 6}
    result["answer"]["executable"] = {"kind": "fitted_temporal_property"}
    packet = build_evidence_plan(question, result)["packet"]
    verdict = repair_selection(packet, {"value": "downward"})
    assert verdict["accepted"] is False
    assert "binding" in verdict["repair"]["instruction"]
    assert "'upward'" in verdict["repair"]["instruction"]
