import pytest

from gnomon.effect_proposals import compose_effect, validate_effect_proposal
from gnomon.llm_dossier import validate_temporal_dossier
from gnomon.publication import publish_result, verify_publication
from gnomon.toolspec import runner_for


TIMES = ["2026-01-03T00:00:00+00:00", "2026-01-04T00:00:00+00:00"]
PRIMARY = [
    {"timestamp": TIMES[0], "point": 10.0, "q10": 9.0, "q50": 10.0, "q90": 11.0},
    {"timestamp": TIMES[1], "point": 10.0, "q10": 9.0, "q50": 10.0, "q90": 11.0},
]


def _proposal(**updates):
    value = {
        "shape": "level_shift", "unit": "target_units",
        "location": 2.0, "lower": 1.0, "upper": 3.0,
        "confidence": .7, "delay_steps": 0, "duration_steps": None,
        "scope": {"kind": "single_series", "series": ["sales"]},
        "claim_ids": ["claim-1"], "rationale": "promotion",
        "uncertainty_basis": "operator estimate",
    }
    value.update(updates)
    return value


def test_typed_effect_is_composed_by_engine_not_model():
    proposal, critique = validate_effect_proposal(
        _proposal(), claim_ids={"claim-1"})
    rows = compose_effect(PRIMARY, proposal)
    assert critique["status"] == "accepted"
    assert [row["q50"] for row in rows] == [12.0, 12.0]
    assert [row["q10"] for row in rows] == [10.0, 10.0]
    assert PRIMARY[0]["q50"] == 10.0


def test_repair_is_bounded_and_typed():
    accepted, critique = validate_effect_proposal(
        _proposal(shape="magic"), claim_ids={"claim-1"}, repair=_proposal())
    assert accepted["shape"] == "level_shift"
    assert critique["status"] == "accepted_after_repair"
    assert critique["attempts_used"] == 2
    rejected, critique = validate_effect_proposal(
        _proposal(shape="magic"), claim_ids={"claim-1"},
        repair=_proposal(shape="still_magic"))
    assert rejected is None and critique["attempts_remaining"] == 0
    assert critique["attempts"][0]["violations"][0]["code"] == "UNKNOWN_EFFECT_SHAPE"


def test_unknown_claim_and_bad_scope_fail_closed():
    proposal, critique = validate_effect_proposal(
        _proposal(claim_ids=["invented"], scope={"kind": "galaxy", "series": []}),
        claim_ids={"claim-1"})
    assert proposal is None
    assert {item["code"] for item in critique["attempts"][0]["violations"]} == {
        "UNVERIFIED_EFFECT_CLAIMS", "INVALID_EFFECT_SCOPE"}


def test_rejected_proposal_is_visible_in_publication_dispositions():
    dossier, _ = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1]}],
        "effect_proposal": _proposal(shape="magic"),
    }, context_text="promotion begins tomorrow",
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
       history=[8, 9, 10], compiler_model="test")
    payload = publish_result({"support": "supported", "forecast": PRIMARY},
                             mode="scenario", dossiers=[dossier])
    rejected = [item for item in payload["context_dispositions"]
                if item["disposition"] == "rejected"]
    assert rejected[0]["reason_code"] == "effect_proposal_rejected"
    assert rejected[0]["violations"][0]["code"] == "UNKNOWN_EFFECT_SHAPE"


def test_publication_prefers_effect_composition_and_retains_portfolio():
    dossier, reasons = validate_temporal_dossier({
        "claims": [{"source_span": "promotion begins tomorrow",
                    "relation": "supports_increase",
                    "effective_start": TIMES[0], "effective_end": TIMES[-1],
                    "confidence": .8}],
        "effect_proposal": _proposal(),
        # A full path is accepted for compatibility but must not outrank the
        # safer engine-composed representation of the same context.
        "forecast_candidate": {"quantiles": [
            {"timestamp": stamp, "q10": 99, "q50": 100, "q90": 101}
            for stamp in TIMES]},
    }, context_text="promotion begins tomorrow",
       cutoff="2026-01-02T00:00:00+00:00", future_timestamps=TIMES,
       history=[8, 9, 10], compiler_model="test")
    assert dossier["effect_proposal_critique"]["status"] == "accepted"
    payload = publish_result({"support": "supported", "forecast": PRIMARY},
                             mode="best_effort", dossiers=[dossier])
    assert payload["recommended_scenario_id"] == "effect-composed-1"
    assert payload["recommended_forecast"][0]["q50"] == 12
    assert payload["primary_forecast"][0]["q50"] == 10
    assert payload["automation"]["eligible"] is False
    assert payload["temporal_state"]["trend"]["direction"] == "flat_or_unknown"
    assert verify_publication(payload)


def test_mcp_one_call_validates_and_composes_raw_context(tmp_path):
    source = tmp_path / "series.csv"
    source.write_text("timestamp,value\n" + "\n".join(
        f"2026-01-{day:02d},{100 + day}" for day in range(1, 21)) + "\n")
    proposal = {
        "claims": [{"source_span": "capacity doubles next week",
                    "relation": "supports_increase",
                    "effective_start": "2026-01-21T00:00:00+00:00",
                    "effective_end": "2026-01-22T00:00:00+00:00",
                    "confidence": .8}],
        "effect_proposal": {
            **_proposal(scope={"kind": "single_series", "series": ["value"]}),
        },
    }
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "target_column": "value", "horizon": 2,
        "output_dir": str(tmp_path / "out"),
        "publication_mode": "best_effort",
        "context_submission": {
            "text": "capacity doubles next week",
            "known_at": "2026-01-20T12:00:00+00:00",
            "compiler": "test-agent", "proposal": proposal,
        },
    })
    publication = payload["publication"]
    assert publication["recommended_scenario_id"] == "effect-composed-1"
    assert publication["primary_forecast_unchanged"] is True
    assert publication["selection_contract"]["temporal_state"]
    assert verify_publication(publication)


@pytest.mark.parametrize("shape", [
    "temporary_pulse", "level_shift", "trend_change", "variance_change",
    "ramp_recovery", "seasonal_amplitude", "seasonal_phase",
    "cross_series_relationship", "saturation_bound",
])
def test_all_composable_shapes_preserve_horizon_and_quantile_order(shape):
    updates = {"shape": shape, "duration_steps": 2}
    if shape == "variance_change":
        updates["unit"] = "fraction_of_level"
    if shape == "cross_series_relationship":
        updates["scope"] = {"kind": "shared", "series": ["sales", "traffic"]}
    if shape.startswith("seasonal_"):
        updates["period_steps"] = 2
    proposal, _ = validate_effect_proposal(
        _proposal(**updates), claim_ids={"claim-1"})
    rows = compose_effect(PRIMARY, proposal)
    assert len(rows) == len(PRIMARY)
    assert all(row["q10"] <= row["q50"] <= row["q90"] for row in rows)
