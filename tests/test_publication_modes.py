from copy import deepcopy
from pathlib import Path

import pytest

from gnomon.llm_dossier import validate_temporal_dossier
from gnomon.publication import (build_scenario_catalog, publish_result,
                                scenario_selection_contract,
                                verify_publication)
from gnomon.publication import record_publication
from gnomon.tracking import TrackingStore
from gnomon.artifacts import verify_artifact_integrity
from gnomon.toolspec import runner_for


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
    assert verify_publication(payload)


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
    assert payload["context_dispositions"] == [{
        "context_id": "dossier-1", "disposition": "rejected",
        "reason_code": "invalid_candidate_seal",
        "reason": "The dossier seal does not authenticate its body.",
    }]


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
    assert Path(payload["publication_path"]).is_file()
    assert verify_artifact_integrity(payload["artifact_path"])


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
