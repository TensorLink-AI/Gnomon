from pathlib import Path

import pytest

from benchmarks.gfr import score_observation
from benchmarks.gfr_smoke import (assemble, calibration_relationship_raw,
                                  conditional_calibration_candidate,
                                  constraint_observations,
                                  outcome_observations,
                                  preservation_observations,
                                  matched_latency_seconds,
                                  prior_classified_without_skill,
                                  validate_matched_identities)


def _identity(method: str) -> dict:
    return {
        "method": method,
        "model": "deepseek", "base_url": "https://example.test/v1",
        "temperature": 1.0, "selected_tasks": ["one"],
        "seed_start": 7, "seeds": 1, "n_samples": 50,
        "fail_on_invalid": True,
        "sample_parallelism": 2,
        "mcp_profile": "evidence" if method == "gnomon-mcp" else None,
    }


def test_matched_cik_identities_are_accepted() -> None:
    validate_matched_identities(_identity("control"),
                                _identity("gnomon-mcp"))


def test_temperature_mismatch_is_rejected() -> None:
    control = _identity("control")
    treatment = _identity("gnomon-mcp")
    treatment["temperature"] = 0.0

    with pytest.raises(ValueError, match="temperature"):
        validate_matched_identities(control, treatment)


def test_sample_parallelism_mismatch_is_rejected() -> None:
    control = _identity("control")
    treatment = _identity("gnomon-mcp")
    treatment["sample_parallelism"] = 1

    with pytest.raises(ValueError, match="sample_parallelism"):
        validate_matched_identities(control, treatment)


def test_matched_latency_uses_one_clock_for_cached_arms() -> None:
    control = {"sample_cache_hits": 55, "request_latency_seconds": 477.0}
    treatment = {"sample_cache_hits": 0, "request_latency_seconds": 25.0}

    assert matched_latency_seconds(control, treatment, .007, .17) == (
        477.0, 25.0)
    assert matched_latency_seconds(
        {"sample_cache_hits": 0}, {"sample_cache_hits": 0}, 9.0, 4.0,
    ) == (9.0, 4.0)
    assert matched_latency_seconds(control, treatment, .007, None) == (
        477.0, 25.0)
    assert matched_latency_seconds(
        {"sample_cache_hits": 1}, {"sample_cache_hits": 0}, .1, .1,
    ) is None


def test_wide_mcp_profile_is_rejected() -> None:
    treatment = _identity("gnomon-mcp")
    treatment["mcp_profile"] = "full"

    with pytest.raises(ValueError, match="Evidence MCP profile"):
        validate_matched_identities(_identity("control"), treatment)


def test_retained_unskilled_prior_is_still_authority_classified() -> None:
    assert prior_classified_without_skill([{
        "support": "prior_assisted",
        "effect": {"recommendation_stability": {
            "reason_code": "sampled_prior_has_no_historical_skill"}},
    }])
    assert not prior_classified_without_skill([{
        "support": "supported", "effect": {},
    }])


def test_conditional_calibration_case_uses_governed_candidate_not_selected():
    candidates = [
        {"role": "immutable_primary", "selected": True, "wis": 1.0},
        {"role": "governed_categorical_state_mapping", "selected": False,
         "wis": 2.0},
    ]
    assert conditional_calibration_candidate(candidates) is candidates[1]
    assert conditional_calibration_candidate(candidates[:1]) is None


def test_calibration_relationship_projects_typed_no_distinct_path():
    publication = {
        "primary_forecast_unchanged": True,
        "candidate_portfolio": [{
            "role": "governed_categorical_state_mapping",
            "effect": {"distribution": {
                "kind": "under_evidence_no_distinct_numeric_path",
                "numeric_authority": "withheld_no_distinct_path",
            }},
        }],
    }
    assert calibration_relationship_raw(publication) == {
        "candidate_relationship": "no_distinct_numeric_path",
        "primary_preserved": True,
        "numeric_path_withheld": True,
    }


def test_assembler_rejects_unknown_scope_before_reading_evidence():
    with pytest.raises(ValueError, match="scope must be smoke or full"):
        assemble(
            root=Path.cwd(), protocol_path=Path("benchmarks/gfr_protocol.json"),
            control_dir=Path("missing"), treatment_dir=Path("missing"),
            context_dir=Path("missing"), short_history=Path("missing"),
            decision_contract=Path("missing"), outcome=Path("missing"),
            boundary=Path("missing"), calibration_action=Path("missing"),
            output_dir=Path("missing"), scope="invalid")


def test_preservation_cases_are_bound_to_semantic_rows():
    rows = [{
        "case": f"decision-{index:02d}", "exact": True,
        "complete": True, "canonical_valid": True,
    } for index in range(12)]
    observed = preservation_observations({"rows": rows})

    assert set(observed) == {
        "preservation:conditional-scenario",
        "preservation:no-distinct-numeric-path",
        "preservation:best-effort", "preservation:typed-choice",
        "preservation:invalid-citation-repair", "preservation:abstention",
    }
    rows[4]["exact"] = False
    assert preservation_observations({"rows": rows})[
        "preservation:conditional-scenario"]["support_preserved"] is False


def test_outcome_cases_preserve_transition_and_automation_evidence():
    family = {"outcome_informed_selections": 0,
              "automation_violations": 0}
    summary = {"gates": {}, "families": {
        "stable_beneficial": {**family, "outcome_informed_selections": 2},
        "delayed_outcomes": dict(family),
        "regime_reversal": {
            **family, "first_demoted_after_regime_change": 11,
            "bad_recommendations_before_demotion": 2},
        "stable_harmful": dict(family),
        "proposer_identity_change": dict(family),
    }}
    observed = outcome_observations(summary)

    assert all(item["expected_transition"] == item["actual_transition"]
               for item in observed.values())
    assert all(item["automatic_model_switch"] is False
               for item in observed.values())


def test_full_constraint_cases_pass_without_inventing_or_forcing_bounds():
    observed = constraint_observations()

    assert set(observed) == {
        "constraint:declared-min", "constraint:declared-max",
        "constraint:declared-window", "constraint:undeclared-min",
        "constraint:contradicted-min",
        "constraint:post-context-reassertion",
    }
    assert all(score_observation("domain_constraints", raw) == 1
               for raw in observed.values())
