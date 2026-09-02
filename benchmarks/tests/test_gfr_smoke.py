import pytest

from benchmarks.gfr_smoke import (conditional_calibration_candidate,
                                  prior_classified_without_skill,
                                  validate_matched_identities)


def _identity(method: str) -> dict:
    return {
        "method": method,
        "model": "deepseek", "base_url": "https://example.test/v1",
        "temperature": 1.0, "selected_tasks": ["one"],
        "seed_start": 7, "seeds": 1, "n_samples": 50,
        "fail_on_invalid": True,
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
