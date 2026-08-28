from datetime import datetime, timedelta, timezone
import math

from gnomon.calibration_counterfactual import (
    compile_additive_drift_repair, deterministic_additive_drift_claim,
)
from gnomon.llm_dossier import validate_temporal_dossier
from gnomon.publication import publish_result


def _case():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    drift_start_index = 48
    rate = 0.2
    history = []
    timestamps = []
    for index in range(24 * 8):
        timestamps.append((start + timedelta(hours=index)).isoformat())
        clean = 20 + 5 * math.sin(2 * math.pi * (index % 24) / 24)
        drift = rate * (index - drift_start_index + 1) \
            if index >= drift_start_index else 0
        history.append(clean + drift)
    future = [(start + timedelta(hours=len(history) + index)).isoformat()
              for index in range(24)]
    drift_start = start + timedelta(hours=drift_start_index)
    context = (
        "The sensor had a calibration problem starting from "
        f"{drift_start:%Y-%m-%d %H:%M:%S} which resulted in an additive "
        f"trend in the series that increases by {rate:.4f} at every hour. "
        f"At timestep {future[0]}, the sensor was repaired and this additive "
        "trend will disappear.")
    claims = [{"claim_id": "claim-1", "source_span": context}]
    return history, timestamps, future, context, claims


def test_source_determined_drift_repair_recovers_clean_daily_path():
    history, timestamps, future, context, claims = _case()
    original = list(history)

    candidate, evidence = compile_additive_drift_repair(
        context_text=context, claims=claims, history=history,
        history_timestamps=timestamps, future_timestamps=future)

    assert candidate is not None
    assert evidence["status"] == "source_determined_prior_assisted"
    assert evidence["human_recommendation_eligible"] is True
    assert evidence["automation_eligible"] is False
    assert evidence["raw_observations_mutated"] is False
    assert history == original
    expected = [20 + 5 * math.sin(2 * math.pi * index / 24)
                for index in range(24)]
    actual = [row["q50"] for row in candidate["quantiles"]]
    assert max(abs(left - right) for left, right in zip(actual, expected)) < 0.1


def test_fully_specified_drift_claim_has_deterministic_transcription_fallback():
    _, timestamps, _, context, _ = _case()

    claim = deterministic_additive_drift_claim(
        context, history_start=timestamps[0], cutoff=timestamps[-1])

    assert claim is not None
    assert claim["effective_start"] == timestamps[48]
    assert claim["effective_end"] == timestamps[-1]
    assert claim["compiler_binding"] == \
        "deterministic_additive_drift_fallback"


def test_drift_claim_fallback_refuses_qualitative_or_multiplicative_repairs():
    _, timestamps, _, context, _ = _case()
    for invalid in (
        context.replace("increases by 0.2000 at every hour", "increases"),
        context.replace("additive trend", "multiplicative trend"),
        context.replace("At timestep", "Around"),
    ):
        assert deterministic_additive_drift_claim(
            invalid, history_start=timestamps[0], cutoff=timestamps[-1]) is None


def test_dossier_and_best_effort_publish_calibration_without_mutating_primary():
    history, timestamps, future, context, _ = _case()
    raw = {"claims": [{
        "source_span": context.split(" At timestep")[0],
        "relation": "unknown", "effective_start": timestamps[48],
        "effective_end": timestamps[-1], "confidence": "high",
    }]}
    dossier, reasons = validate_temporal_dossier(
        raw, context_text=context, cutoff=timestamps[-1],
        future_timestamps=future, history=history,
        history_timestamps=timestamps, compiler_model="test",
        candidate_selection_eligible=False,
        candidate_selection_reason="model transformation failed")
    assert not reasons
    assert dossier["candidate_critique"]["candidate_origin"] == \
        "calibration_counterfactual"
    assert dossier["candidate_critique"]["selection_eligible"] is True
    primary = [{"timestamp": stamp, "point": 50, "q10": 49,
                "q50": 50, "q90": 51} for stamp in future]

    strict = publish_result(
        {"support": "supported", "forecast": primary},
        mode="strict", dossiers=[dossier])
    best = publish_result(
        {"support": "supported", "forecast": primary},
        mode="best_effort", dossiers=[dossier])

    assert strict["recommended_scenario_id"] == "primary"
    assert best["recommended_scenario_id"] == "prior-assisted-1"
    assert best["recommendation_authority"]["selection_method"] == \
        "source_determined_calibration_best_effort"
    assert best["automation"]["eligible"] is False
    assert best["primary_forecast"] == primary


def test_drift_repair_requires_exact_repair_boundary():
    history, timestamps, future, context, claims = _case()
    vague = context.replace(f"At timestep {future[0]}", "Soon")

    candidate, evidence = compile_additive_drift_repair(
        context_text=vague, claims=claims, history=history,
        history_timestamps=timestamps, future_timestamps=future)

    assert candidate is None
    assert evidence["status"] == "not_applicable"


def test_drift_repair_does_not_generalize_additive_rule_to_multiplicative_drift():
    history, timestamps, future, context, claims = _case()
    context = context.replace("additive trend", "multiplicative trend")

    candidate, evidence = compile_additive_drift_repair(
        context_text=context, claims=claims, history=history,
        history_timestamps=timestamps, future_timestamps=future)

    assert candidate is None
    assert evidence["status"] == "not_applicable"
