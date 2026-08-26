from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.llm_dossier import validate_temporal_dossier
from gnomon.publication import (build_scenario_catalog, publish_result,
                                scenario_selection_contract, select_publication,
                                verify_publication, write_publication)
from gnomon.publication import record_publication
from gnomon.tracking import TrackingStore
from gnomon.artifacts import verify_artifact_integrity
from gnomon.toolspec import runner_for
from gnomon.toolspec import enforce_response_budget


TIMES = ["2026-01-03T00:00:00+00:00", "2026-01-04T00:00:00+00:00"]


def _result():
    return {
        "support": "supported",
        "forecast": [
            {"timestamp": TIMES[0], "point": 10.0, "q10": 9, "q50": 10, "q90": 11},
            {"timestamp": TIMES[1], "point": 10.0, "q10": 9, "q50": 10, "q90": 11},
        ],
    }


def _dossier():
    raw = {
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": TIMES[0], "effective_end": TIMES[1],
                    "confidence": .8}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": TIMES[0], "q10": 10, "q50": 11, "q90": 12},
            {"timestamp": TIMES[1], "q10": 11, "q50": 12, "q90": 13},
        ], "rationale": "promotion"},
    }
    return validate_temporal_dossier(
        raw, context_text="promotion begins tomorrow",
        cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
        history=[8, 9, 10], compiler_model="test-model")[0]


def test_strict_never_promotes_prior_assisted_candidate():
    payload = publish_result(_result(), mode="strict", dossiers=[_dossier()])
    assert payload["recommended_scenario_id"] == "primary"
    assert payload["recommended_forecast"] == _result()["forecast"]
    assert payload["automation"]["eligible"] is False
    assert verify_publication(payload)


def test_best_effort_promotes_candidate_but_not_authority():
    payload = publish_result(
        _result(), mode="best_effort", dossiers=[_dossier()],
        automation_policy={"authorize": True})
    assert payload["recommended_scenario_id"] == "prior-assisted-1"
    assert payload["recommended_support"] == "prior_assisted"
    assert payload["primary_forecast"] == _result()["forecast"]
    assert payload["automation"]["eligible"] is False
    authority = payload["recommendation_authority"]
    assert authority["selection_method"] == "default_prior_assisted_lane"
    assert authority["independent_selection_performed"] is False
    assert authority["historically_admitted"] is False
    assert authority["prior_assisted"] is True
    assert authority["human_review_required"] is True
    assert verify_publication(payload)


def test_replay_admitted_observation_counterfactual_has_truthful_authority():
    start = datetime(2025, 10, 5, tzinfo=timezone.utc)
    history_times = [(start + timedelta(days=index)).isoformat()
                     for index in range(90)]
    history = []
    for index in range(90):
        disrupted = index % 6 in {0, 1, 2}
        history.append(-8.0 if disrupted else 20.0 + (index % 3 - 1))
    claim = (
        "Maintenance lasted for 3 days every 6 days starting from "
        "2025-10-05 00:00:00, resulting in no requests recorded.")
    context = claim + " There will be no future maintenance."
    dossier, reasons = validate_temporal_dossier(
        {"claims": [{"source_span": claim, "relation": "unknown",
                     "effective_start": history_times[0],
                     "effective_end": history_times[-1], "confidence": 1}]},
        context_text=context, cutoff=history_times[-1],
        future_timestamps=TIMES, history=history,
        history_timestamps=history_times, compiler_model="test")
    assert not reasons
    assert dossier["candidate_critique"]["selection_eligible"] is True

    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])

    assert payload["recommended_scenario_id"] == "prior-assisted-1"
    candidate = next(item for item in payload["candidate_portfolio"]
                     if item["scenario_id"] == "prior-assisted-1")
    assert candidate["role"] == "observation_counterfactual"
    assert candidate["support"] == "conditionally_supported"
    assert candidate["effect"]["conditional_replay"][
        "selection_eligible"] is True
    authority = payload["recommendation_authority"]
    assert authority["selection_method"] == "conditional_replay_evidence"
    assert authority["conditional_replay_admitted"] is True
    assert authority["historically_admitted"] is False
    assert authority["human_review_required"] is True
    assert payload["automation"]["eligible"] is False
    assert verify_publication(payload)


def test_governed_selection_repoints_sealed_path_without_reforecasting():
    original = publish_result(
        _result(), mode="best_effort", dossiers=[_dossier()])
    original_seals = [item["scenario_seal_sha256"]
                      for item in original["candidate_portfolio"]]
    selected = select_publication(original, {
        "selected_scenario_id": "primary",
        "ranking": ["primary", "prior-assisted-1"],
        "cited_claim_ids": [],
        "counterevidence_claim_ids": ["claim-1"],
        "confidence": .6,
        "rationale": "The conditional claim is not independently validated.",
        "what_would_change_selection": "A resolved outcome history.",
    })
    assert original["recommended_scenario_id"] == "prior-assisted-1"
    assert selected["recommended_scenario_id"] == "primary"
    assert [item["scenario_seal_sha256"]
            for item in selected["candidate_portfolio"]] == original_seals
    assert selected["primary_forecast"] == original["primary_forecast"]
    assert selected["primary_forecast_unchanged"] is True
    assert selected["recommended_support"] == "supported"
    assert selected["automation"]["eligible"] is False
    assert selected["recommendation_authority"][
        "independent_selection_performed"] is True
    assert selected["supersedes_publication_seal_sha256"] == original[
        "publication_seal_sha256"]
    assert verify_publication(original)
    assert verify_publication(selected)


def test_governed_selection_refuses_strict_or_tampered_publication():
    strict = publish_result(_result(), mode="strict")
    with pytest.raises(ValueError, match="strict"):
        select_publication(strict, {})
    broken = deepcopy(publish_result(
        _result(), mode="scenario", dossiers=[_dossier()]))
    broken["primary_forecast"][0]["point"] = -999
    with pytest.raises(ValueError, match="invalid publication"):
        select_publication(broken, {})


def test_mcp_selector_persists_new_sidecar_without_reforecasting(tmp_path):
    original = publish_result(
        _result(), mode="best_effort", dossiers=[_dossier()], artifact_id="f1")
    original_path = write_publication(tmp_path / "forecast_f1", original)
    payload = runner_for("gnomon_select_scenario")({
        "publication_path": str(original_path),
        "scenario_selection": {
            "selected_scenario_id": "primary",
            "ranking": ["primary", "prior-assisted-1"],
            "cited_claim_ids": [],
            "counterevidence_claim_ids": ["claim-1"],
            "confidence": .6,
            "rationale": "Prefer the governed primary pending outcomes.",
            "what_would_change_selection": "Resolved candidate outcomes.",
        },
    })
    assert payload["recommended_scenario_id"] == "primary"
    assert payload["artifact_id"] == "f1"
    assert payload["primary_forecast_unchanged"] is True
    assert payload["automation"]["eligible"] is False
    assert payload["reasoning"]["canonical_source"] == "/headline"
    assert payload["reasoning"]["sufficiency"] == {
        "sufficient_for": [
            "select_scenario:canonical_answer", "explain_support"],
        "further_calls_add_nothing_for": [
            "select_scenario:canonical_answer", "explain_support"],
        "requires_follow_up": False,
    }
    assert payload["reasoning"]["resolution"]["kind"] == "complete"
    selected_path = Path(payload["publication_path"])
    assert selected_path.is_file()
    assert selected_path != original_path
    assert original_path.read_text(encoding="utf-8") == (
        __import__("json").dumps(original, indent=2, sort_keys=True) + "\n")


def test_model_authored_path_cannot_bypass_governed_transform_authority():
    result = _result()
    result["transformation_candidates"] = [{
        "transformation_id": "equation-1",
        "forecast": [{**row, "point": 20, "q10": 19, "q50": 20,
                      "q90": 21} for row in result["forecast"]],
        "lane": "historically_testable", "claim_ids": ["claim-1"],
        "known_at": "2026-01-02T00:00:00+00:00",
        "output_unit": "value", "source_seal_sha256": "sealed",
        "primary_forecast_unchanged": True, "automation_eligible": False,
        "validation": {"recurrence_plausibility_passed": True,
                       "recurrence_replay_admitted": False,
                       "recurrence_replay_reason": "did_not_beat_last_value"},
    }]
    payload = publish_result(result, mode="best_effort", dossiers=[_dossier()])
    assert payload["recommended_scenario_id"] == "primary"
    by_id = {item["scenario_id"]: item
             for item in payload["candidate_portfolio"]}
    assert by_id["transformation-1"]["selection_eligible"] is False
    assert by_id["prior-assisted-1"]["selection_eligible"] is False
    assert "owns recommendation authority" in " ".join(
        by_id["prior-assisted-1"]["assumptions"])


def test_scenario_selection_can_rank_but_not_authorize_or_edit():
    selection = {
        "selected_scenario_id": "prior-assisted-1",
        "ranking": ["prior-assisted-1", "primary"],
        "cited_claim_ids": ["claim-1"], "counterevidence_claim_ids": [],
        "confidence": .7, "rationale": "dated promotion supports uplift",
        "what_would_change_selection": "promotion cancellation",
        "automation_authorized": True,
        "forecast": [{"point": 999}],
    }
    payload = publish_result(_result(), mode="scenario", dossiers=[_dossier()],
                             scenario_selection=selection)
    assert payload["scenario_selection"]["automation_authorized"] is False
    assert "forecast" not in payload["scenario_selection"]
    assert payload["recommended_forecast"][0]["q50"] == 11
    assert verify_publication(payload)


def test_validated_context_event_is_citable_without_a_dossier_claim():
    result = _result()
    result["primary_forecast"] = [
        {**row, "point": 9.0, "q10": 8, "q50": 9, "q90": 10}
        for row in result["forecast"]]
    result["context_outcome"] = {
        "status": "applied", "admission_basis": "future_context_contract",
        "events": ["event-1"],
    }
    selection = {
        "selected_scenario_id": "context_conditioned",
        "ranking": ["context_conditioned", "primary"],
        "cited_claim_ids": ["event-1"], "counterevidence_claim_ids": [],
        "confidence": .8, "rationale": "validated event applies",
        "what_would_change_selection": "event cancellation",
    }
    payload = publish_result(result, mode="best_effort",
                             scenario_selection=selection)
    assert payload["scenario_selection"]["cited_claim_ids"] == ["event-1"]
    authority = payload["recommendation_authority"]
    assert authority["selection_method"] == "governed_scenario_selection"
    assert authority["independent_selection_performed"] is True
    assert verify_publication(payload)


def test_selector_cannot_displace_context_trusted_path_with_weaker_primary():
    result = _result()
    result["support"] = "context_trusted"
    result["primary_forecast"] = [
        {**row, "point": 9.0, "q10": 8, "q50": 9, "q90": 10}
        for row in result["forecast"]]
    result["context_outcome"] = {
        "status": "applied", "admission_basis": "future_context_contract",
        "events": ["event-1"],
    }
    # Force the primary's own path tier below the deterministically validated
    # context path, matching a best-effort primary plus a literal future rule.
    for row in result["primary_forecast"]:
        row["tier"] = "conditionally_supported"
    with pytest.raises(ValueError, match="evidence-dominant path"):
        publish_result(result, mode="best_effort", scenario_selection={
            "selected_scenario_id": "primary",
            "ranking": ["primary", "context_conditioned"],
            "cited_claim_ids": ["event-1"], "counterevidence_claim_ids": [],
            "confidence": .7, "rationale": "prefer history",
            "what_would_change_selection": "more context",
        })
    scenarios, _ = build_scenario_catalog(result)
    contract = scenario_selection_contract(scenarios=scenarios)
    assert contract["selection_required"] is False
    assert contract["deterministic_scenario_id"] == "context_conditioned"


def test_unknown_citations_and_tampering_fail_loudly():
    selection = {
        "selected_scenario_id": "primary",
        "ranking": ["primary", "prior-assisted-1"],
        "cited_claim_ids": ["made-up"], "counterevidence_claim_ids": [],
        "confidence": .5, "rationale": "x",
        "what_would_change_selection": "y",
    }
    with pytest.raises(ValueError, match="unknown claim"):
        publish_result(_result(), mode="scenario", dossiers=[_dossier()],
                       scenario_selection=selection)
    payload = publish_result(_result(), mode="strict")
    damaged = deepcopy(payload)
    damaged["recommended_forecast"][0]["point"] = 999
    assert not verify_publication(damaged)


def test_invalid_context_is_typed_rejection_not_silent_drop():
    broken = _dossier()
    broken["compiler_model"] = "tampered"
    payload = publish_result(_result(), mode="scenario", dossiers=[broken])
    assert len(payload["context_dispositions"]) == 1
    rejection = payload["context_dispositions"][0]
    assert rejection["context_id"] == "dossier-1"
    assert rejection["disposition"] == "rejected"
    assert rejection["reason_code"] == "invalid_candidate_seal"
    assert rejection["reason"] == \
        "The dossier seal does not authenticate its body."
    assert rejection["recovery_action"]["code"] == "recompile_from_source"
    assert rejection["recovery_action"]["automation_eligible"] is False


def test_candidate_constraint_failure_is_typed_and_actionable():
    raw = {
        "claims": [{"source_span": "values stay between 4.79 and 9.13",
                    "relation": "constrains_range",
                    "effective_start": TIMES[0], "effective_end": TIMES[1],
                    "confidence": .9}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": timestamp, "q10": 0, "q50": 0, "q90": 0}
            for timestamp in TIMES], "rationale": "bounded path"},
    }
    dossier, _ = validate_temporal_dossier(
        raw, context_text="values stay between 4.79 and 9.13",
        cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
        history=[8, 9, 10], compiler_model="test-model")
    payload = publish_result(_result(), mode="best_effort", dossiers=[dossier])
    rejection = next(item for item in payload["context_dispositions"]
                     if item["reason_code"] == "forecast_candidate_rejected")
    assert rejection["disposition"] == "rejected"
    assert "violates cited lower bound" in rejection["reason"]
    assert rejection["recovery_action"]
    assert payload["recommended_scenario_id"] == "primary"
    assert verify_publication(payload)


def test_publication_reuses_synthesis_tracking_and_scores_numeric_uplift(tmp_path):
    store = TrackingStore(tmp_path / "tracking.db")
    payload = publish_result(_result(), mode="best_effort", dossiers=[_dossier()])
    synthesis_id = record_publication(
        store, project="p", forecast_id="f", series="x", payload=payload)
    score = store.resolve_temporal_synthesis(
        project="p", forecast_id="f", series="x", question_id="publication",
        synthesis_id=synthesis_id, outcome={"points": [11.0, 12.0]})
    assert score["rule"] == "numeric_path_wape_v1"
    assert score["synthesis_won"] is True
    assert score["synthesis_delta"] > 0


def test_mode_invariants_hold_across_varied_bounded_paths():
    for offset in (-2.0, -0.25, 0.0, 0.25, 2.0):
        # Rebuild rather than tamper with the seal.
        raw = {
            "claims": [{"source_span": "promotion begins tomorrow",
                        "relation": "supports_increase",
                        "effective_start": TIMES[0], "effective_end": TIMES[1],
                        "confidence": .6}],
            "forecast_candidate": {"quantiles": [
                {"timestamp": stamp, "q10": 9 + offset + i,
                 "q50": 10 + offset + i, "q90": 11 + offset + i}
                for i, stamp in enumerate(TIMES)]},
        }
        dossier = validate_temporal_dossier(
            raw, context_text="promotion begins tomorrow",
            cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
            history=[8, 9, 10], compiler_model="property-test")[0]
        strict = publish_result(_result(), mode="strict", dossiers=[dossier])
        best = publish_result(_result(), mode="best_effort", dossiers=[dossier])
        assert strict["recommended_forecast"] == strict["primary_forecast"]
        assert best["primary_forecast"] == strict["primary_forecast"]
        assert best["recommended_support"] == "prior_assisted"
        assert best["automation"]["eligible"] is False
        assert verify_publication(strict) and verify_publication(best)


def test_mcp_forecast_persists_verified_sidecar_without_mutating_artifact(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    future = [f"2026-02-{day:02d}T00:00:00+00:00" for day in (10, 11)]
    dossier = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": future[0], "effective_end": future[-1],
                    "confidence": .7}],
        "forecast_candidate": {"quantiles": [
            {"timestamp": future[0], "q10": 139, "q50": 140, "q90": 141},
            {"timestamp": future[1], "q10": 140, "q50": 141, "q90": 142},
        ]}}, context_text="promotion begins tomorrow",
        cutoff="2026-02-09T00:00:00+00:00", future_timestamps=future,
        history=list(range(100, 140)), compiler_model="test")[0]
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "publication_mode": "best_effort", "temporal_dossiers": [dossier],
    })
    assert payload["publication"]["recommended_scenario_id"] == "prior-assisted-1"
    assert payload["publication"]["projection"] == "compact"
    assert "recommended_forecast" not in payload["publication"]
    assert "candidate_portfolio" not in payload["publication"]
    assert payload["publication"]["selection_contract"]
    assert payload["publication"]["receipt_is_complete_and_sealed"] is True
    receipt = __import__("json").loads(
        Path(payload["publication_path"]).read_text(encoding="utf-8"))
    assert verify_publication(receipt)
    assert Path(payload["publication_path"]).is_file()
    assert verify_artifact_integrity(payload["artifact_path"])


def test_full_format_explicitly_returns_complete_signed_publication(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 12, "format": "full",
        "output_dir": str(tmp_path / "out"), "publication_mode": "scenario",
    })
    assert payload["publication"]["candidate_portfolio"]
    assert payload["publication"]["recommended_forecast"]
    assert "projection" not in payload["publication"]
    assert verify_publication(payload["publication"])


def test_scenario_overflow_is_bounded_with_typed_dispositions():
    result = _result()
    result["sensitivity_scenarios"] = [{
        "forecast": [{**row, "q50": 10 + index, "point": 10 + index}
                     for row in result["forecast"]],
        "support": "hypothetical_sensitivity", "assumptions": [f"case {index}"],
    } for index in range(12)]
    payload = publish_result(result, mode="scenario", dossiers=[_dossier()])
    assert len(payload["candidate_portfolio"]) == 8
    assert payload["candidate_portfolio"][0]["scenario_id"] == "primary"
    assert any(item["reason_code"] == "bounded_portfolio_overflow"
               for item in payload["context_dispositions"])
    assert verify_publication(payload)


def test_mcp_context_transformation_rejection_is_typed_and_primary_is_intact(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "format": "full",
        "publication_mode": "scenario",
        "context_submission": {
            "known_at": "2026-02-09T00:00:00+00:00",
            "transformations": [{
                "known_at": "2026-02-09T00:00:00+00:00",
                "claim_ids": ["invented"], "lane": "prior_assisted",
                "expression": {"op": "python", "code": "raise SystemExit"},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    assert publication["primary_forecast"] == publication["recommended_forecast"]
    rejection = next(item for item in publication["context_dispositions"]
                     if item["reason_code"] == "transformation_validation_failed")
    assert rejection["violations"][0]["code"] == "UNVERIFIED_CLAIMS"
    assert publication["automation"]["eligible"] is False
    assert verify_publication(publication)


def test_mcp_compiler_rejection_is_visible_in_publication(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "known_at": "2026-02-09T00:00:00+00:00",
            "rejections": [
                "context_unresolved: no grounded numeric relationship was found"],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    rejection = next(item for item in publication["context_dispositions"]
                     if item["reason_code"] == "context_unresolved")
    assert rejection["disposition"] == "rejected"
    assert rejection["reason"] == "no grounded numeric relationship was found"
    assert rejection["recovery_action"]["code"] == "provide_grounded_context"
    assert "effective dates" in rejection["recovery_action"]["required_evidence"]
    assert publication["primary_forecast_unchanged"] is True
    assert verify_publication(publication)


def test_every_rejected_context_disposition_has_bounded_recovery():
    result = _result()
    result["transformation_rejections"] = [{
        "transformation_id": "bad-transform",
        "reason_code": "transformation_validation_failed",
        "reason": "unknown series",
    }]
    result["context_rejections"] = [{
        "context_id": "unresolved-document",
        "reason_code": "context_unresolved",
        "reason": "no executable facts",
    }]
    payload = publish_result(result, mode="scenario", dossiers=[{
        **_dossier(), "compiler_model": "tampered"}])
    rejected = [item for item in payload["context_dispositions"]
                if item["disposition"] == "rejected"]
    assert len(rejected) == 3
    for item in rejected:
        action = item["recovery_action"]
        assert action["code"]
        assert action["message"]
        assert action["required_evidence"]
        assert action["automation_eligible"] is False
    assert verify_publication(payload)


def test_rejected_candidate_normalizes_legacy_string_recovery():
    dossier = _dossier()
    dossier["forecast_candidate"] = None
    dossier["candidate_critique"] = {
        "status": "rejected", "reasons": ["candidate is incomplete"],
        "recovery_action": "try again",
    }
    import hashlib
    import json
    body = {key: value for key, value in dossier.items()
            if key != "seal_sha256"}
    dossier["seal_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()
    payload = publish_result(_result(), mode="scenario", dossiers=[dossier])
    rejected = next(item for item in payload["context_dispositions"]
                    if item["reason_code"] == "forecast_candidate_rejected")
    assert rejected["recovery_action"]["code"] == "correct_context_proposal"
    assert rejected["recovery_action"]["automation_eligible"] is False


def test_mcp_recursive_transformation_binds_history_but_requires_replay_skill(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "wide.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,campaign,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{i + 1},{100 + i}" for i in range(40))
        + "\n")
    formula = "value[t] = 0.5 value[t-1] + 2 campaign[t-1]"
    schedule = "future campaign values are 41 then 42"
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "text": formula + ". " + schedule,
            "known_at": "2026-02-09T00:00:00+00:00",
            "compiler": "test-agent",
            "proposal": {"claims": [
                {"source_span": formula, "relation": "supports_increase",
                 "effective_start": "2026-02-10T00:00:00+00:00",
                 "effective_end": "2026-02-11T00:00:00+00:00"},
                {"source_span": schedule, "relation": "supports_increase",
                 "effective_start": "2026-02-10T00:00:00+00:00",
                 "effective_end": "2026-02-11T00:00:00+00:00"},
            ]},
            "transformations": [{
                "transformation": {
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "claim_ids": ["claim-1", "claim-2"],
                    "lane": "prior_assisted", "output_unit": "value",
                    "expression": {
                        "op": "recursive_linear", "output_unit": "value",
                        "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
                        "driver_terms": [{"series": "campaign", "lag": 1,
                                          "coefficient": 2}],
                    },
                },
                "units": {"primary": "value", "campaign": "campaign"},
                "series_values": {"campaign": {
                    "values": [41, 42],
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "source_claim_ids": ["claim-2"]}},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    scenario = next(item for item in publication["candidate_portfolio"]
                    if item["scenario_id"] == "transformation-1")
    assert [row["q50"] for row in scenario["forecast"]] == [149.5, 156.75]
    assert scenario["selection_eligible"] is False
    assert scenario["effect"]["validation"][
        "recurrence_replay_reason"] == "did_not_beat_last_value"
    assert publication["primary_forecast_unchanged"] is True
    assert publication["automation"]["eligible"] is False
    assert verify_publication(publication)


def test_missing_recursive_driver_rejects_scenario_not_primary(tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "series.csv"
    start = date(2026, 1, 1)
    source.write_text("timestamp,value\n" + "\n".join(
        f"{start + timedelta(days=i)},{100 + i}" for i in range(40)) + "\n")
    formula = "value[t] = 2 missing_driver[t-1]"
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"), "publication_mode": "scenario",
        "context_submission": {
            "text": formula, "known_at": "2026-02-09T00:00:00+00:00",
            "compiler": "test", "proposal": {"claims": [{
                "source_span": formula, "relation": "supports_increase",
                "effective_start": "2026-02-10T00:00:00+00:00",
                "effective_end": "2026-02-11T00:00:00+00:00"}]},
            "transformations": [{
                "transformation": {
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "claim_ids": ["claim-1"], "lane": "prior_assisted",
                    "output_unit": "value", "expression": {
                        "op": "recursive_linear", "output_unit": "value",
                        "autoregressive_terms": [],
                        "driver_terms": [{"series": "missing_driver", "lag": 1,
                                          "coefficient": 2}]},
                },
                "units": {"primary": "value", "missing_driver": "driver"},
                "series_values": {"missing_driver": {
                    "values": [1, 1],
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "source_claim_ids": ["claim-1"]}},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "primary"
    assert publication["primary_forecast_unchanged"] is True
    rejection = next(item for item in publication["context_dispositions"]
                     if item["reason_code"] == "MISSING_COLUMNS")
    assert "missing_driver" in rejection["reason"]


def test_documented_history_can_replay_encoded_driver_without_guessing_scale(
        tmp_path):
    from datetime import date, timedelta
    source = tmp_path / "encoded.csv"
    start = date(2026, 1, 1)
    target = [10.0]
    for _ in range(1, 40):
        target.append(.5 * target[-1] + 2 * 2.0)
    source.write_text("timestamp,campaign,value\n" + "\n".join(
        f"{start + timedelta(days=i)},999,{target[i]}" for i in range(40))
        + "\n")
    formula = "value[t] = 0.5 value[t-1] + 2 campaign[t-1]"
    history = "campaign is 2 from 2026-01-01 to 2026-02-09"
    schedule = "future campaign values are 2 then 2"
    claims = [
        {"source_span": text, "relation": "unknown",
         "effective_start": "2026-02-10T00:00:00+00:00",
         "effective_end": "2026-02-11T00:00:00+00:00"}
        for text in (formula, history, schedule)]
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "format": "full",
        "publication_mode": "best_effort",
        "context_submission": {
            "text": ". ".join((formula, history, schedule)),
            "known_at": "2026-02-09T00:00:00+00:00",
            "compiler": "test", "proposal": {"claims": claims},
            "transformations": [{
                "transformation": {
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "claim_ids": ["claim-1", "claim-2", "claim-3"],
                    "lane": "historically_testable", "output_unit": "value",
                    "expression": {"op": "recursive_linear",
                        "output_unit": "value", "intercept": 0,
                        "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
                        "driver_terms": [{"series": "campaign", "lag": 1,
                                          "coefficient": 2}]},
                },
                "units": {"primary": "value", "campaign": "campaign"},
                "series_values": {"campaign": {"values": [2, 2],
                    "known_at": "2026-02-09T00:00:00+00:00",
                    "source_claim_ids": ["claim-3"]}},
                "historical_series_segments": {"campaign": [{
                    "start": "2026-01-01", "end": "2026-02-09", "value": 2,
                    "source_claim_ids": ["claim-2"]}]},
            }],
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "transformation-1"
    scenario = next(item for item in publication["candidate_portfolio"]
                    if item["scenario_id"] == "transformation-1")
    validation = scenario["effect"]["validation"]
    assert validation["recurrence_replay_admitted"] is True
    assert validation["recurrence_replay_candidate_mae"] == pytest.approx(0)
    assert validation["recurrence_history_source"] == "document_cited_segments"
    assert publication["automation"]["eligible"] is False


def test_response_budget_never_breaks_a_publication_seal():
    result = _result()
    result["forecast"] = [
        {"timestamp": f"2026-01-{day:02d}T00:00:00+00:00",
         "point": float(day), "q10": day - 1, "q50": day, "q90": day + 1}
        for day in range(3, 28)]
    publication = publish_result(result, mode="strict")
    trimmed = enforce_response_budget({
        "artifact_path": "/tmp/artifact", "publication": publication,
        "bulk": list(range(1000)),
    }, budget_bytes=1000)
    assert trimmed["truncated"] is True
    assert trimmed["publication"] == publication
    assert verify_publication(trimmed["publication"])
