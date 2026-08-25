from __future__ import annotations

from gnomon.llm_dossier import (
    validate_temporal_dossier,
    verify_temporal_dossier_seal,
)


def _raw(span: str, rows=None):
    return {
        "claims": [{
            "source_span": span,
            "relation": "supports_decrease",
            "effective_start": "2026-01-05T00:00:00+00:00",
            "effective_end": "2026-01-06T00:00:00+00:00",
            "mechanism": "closure",
            "confidence": 0.9,
        }],
        "forecast_candidate": {"quantiles": rows or [
            {"q10": 8, "q50": 9, "q90": 10},
            {"q10": 7, "q50": 8, "q90": 9},
        ]},
    }


def test_valid_dossier_is_cited_sealed_and_non_automatable():
    span = "The site will be closed on Monday."
    dossier, reasons = validate_temporal_dossier(
        _raw(span), context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert not reasons
    assert dossier["forecast_candidate"] is not None
    assert dossier["candidate_support"] == "prior_assisted"
    assert dossier["automation_eligible"] is False
    assert dossier["primary_forecast_unchanged"] is True
    assert len(dossier["seal_sha256"]) == 64
    assert verify_temporal_dossier_seal(dossier)
    dossier["claims"][0]["confidence"] = 0.1
    assert not verify_temporal_dossier_seal(dossier)


def test_uncited_claim_cannot_author_a_candidate():
    dossier, reasons = validate_temporal_dossier(
        _raw("invented"), context_text="The site remains open.",
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert dossier["forecast_candidate"] is None
    assert any("verbatim source_span" in reason for reason in reasons)
    assert any("requires a verified cited claim" in reason for reason in reasons)


def test_bad_quantile_order_and_implausible_jump_are_rejected():
    span = "The site will be closed on Monday."
    bad_order = [{"q10": 10, "q50": 9, "q90": 8}] * 2
    dossier, reasons = validate_temporal_dossier(
        _raw(span, bad_order), context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert dossier["forecast_candidate"] is None
    assert any("quantiles are invalid" in reason for reason in reasons)

    huge = [{"q10": 999, "q50": 1000, "q90": 1001}] * 2
    dossier, reasons = validate_temporal_dossier(
        _raw(span, huge), context_text=span,
        cutoff="2026-01-04T00:00:00+00:00",
        future_timestamps=["2026-01-05T00:00:00+00:00",
                           "2026-01-06T00:00:00+00:00"],
        history=[8, 9, 10, 11], compiler_model="model-x")
    assert dossier["forecast_candidate"] is None
    assert any("boundary-jump" in reason for reason in reasons)
