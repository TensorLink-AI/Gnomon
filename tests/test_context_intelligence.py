from datetime import datetime, timedelta, timezone

import pytest

from gnomon.context_intelligence import (
    align_vintage_rows, candidate_evidence_score, compile_context_hypotheses,
    fit_historical_analogue, fit_lagged_relationship, fit_vintage_exogenous,
)
from gnomon.publication import publish_result, verify_publication


UTC = timezone.utc


def _stamp(index):
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)).isoformat()


CLAIMS = [{"claim_id": "claim-1"}, {"claim_id": "claim-2"}]


def test_multi_hypothesis_compilation_is_stable_and_repair_is_bounded():
    valid = {"kind": "relationship", "claim_ids": ["claim-1"],
             "target_series": ["sales"], "predictor_series": "traffic",
             "known_at": _stamp(4), "lag_steps": 1}
    invalid = {**valid, "claim_ids": ["invented"]}
    repaired = {**valid, "claim_ids": ["claim-2"], "lag_steps": 2}
    first, critique = compile_context_hypotheses(
        [invalid, valid], claims=CLAIMS, series=["sales", "traffic"],
        cutoff=_stamp(5), repair=[repaired])
    second, _ = compile_context_hypotheses(
        [valid, invalid], claims=CLAIMS, series=["traffic", "sales"],
        cutoff=_stamp(5), repair=[repaired])
    assert [item["hypothesis_id"] for item in first] == [
        item["hypothesis_id"] for item in second]
    assert len(first) == 2
    assert critique["attempts_used"] == 2
    assert all(item["validation"]["known_at_cutoff"] for item in first)


def test_hypothesis_rejects_future_knowledge_and_unknown_series():
    hypotheses, critique = compile_context_hypotheses([{
        "kind": "relationship", "claim_ids": ["claim-1"],
        "target_series": ["profit"], "predictor_series": "traffic",
        "known_at": _stamp(9),
    }], claims=CLAIMS, series=["sales", "traffic"], cutoff=_stamp(5))
    assert hypotheses == []
    codes = {item["code"] for item in critique["rejected"][0]["violations"]}
    assert codes == {"UNKNOWN_SERIES", "NOT_KNOWN_AT_CUTOFF"}


def test_vintage_alignment_uses_latest_knowable_revision_only():
    rows = [
        {"timestamp": _stamp(1), "known_at": _stamp(1), "x": 1},
        {"timestamp": _stamp(1), "known_at": _stamp(3), "x": 2},
        {"timestamp": _stamp(1), "known_at": _stamp(7), "x": 999},
        {"timestamp": _stamp(8), "known_at": _stamp(1), "x": 999},
    ]
    assert align_vintage_rows(rows, cutoff=_stamp(5)) == [rows[1]]


def test_vintage_exogenous_rejects_revisions_that_arrived_after_origin():
    rows = []
    for i in range(45):
        rows.append({"timestamp": _stamp(i), "known_at": _stamp(i),
                     "sales": 3 + 2 * i, "traffic": i})
        rows.append({"timestamp": _stamp(i), "known_at": _stamp(i + 1),
                     "sales": 9999, "traffic": 9999})
    fitted = fit_vintage_exogenous(
        rows, target_key="sales", predictor_keys=["traffic"],
        cutoff=_stamp(44), hypothesis_id="h")
    result = fitted.execute()
    assert result["validation"]["per_origin_knowledge_checked"] is True
    assert result["validation"]["beats_baseline"] is True
    assert result["automation_eligible"] is False


def test_lagged_relationship_selects_real_lag_on_expanding_origins():
    predictor, target = [], []
    values = [float((i * 7) % 19 + i / 7) for i in range(80)]
    for i, value in enumerate(values):
        predictor.append({"timestamp": _stamp(i), "known_at": _stamp(i), "x": value})
        y = 10.0 if i < 2 else 4 + 3 * values[i - 2]
        target.append({"timestamp": _stamp(i), "known_at": _stamp(i), "y": y})
    fitted = fit_lagged_relationship(
        target, predictor, target_key="y", predictor_key="x",
        cutoff=_stamp(79), hypothesis_id="h", lags=[0, 1, 2, 3])
    result = fitted.execute()
    assert result["estimate"]["selected_lag_steps"] == 2
    assert result["support"] == "supported"
    assert result["automation_eligible"] is False


def test_historical_analogue_excludes_outcomes_unknown_at_cutoff():
    episodes = []
    for i in range(12):
        episodes.append({"episode_id": str(i), "features": {"load": float(i)},
                         "outcome": float(2 * i + 1),
                         "outcome_known_at": _stamp(i + 1)})
    episodes.append({"episode_id": "future", "features": {"load": 6.0},
                     "outcome": -9999, "outcome_known_at": _stamp(30)})
    fitted = fit_historical_analogue(
        episodes, query_features={"load": 7.0}, cutoff=_stamp(20),
        hypothesis_id="h", k=3)
    result = fitted.execute()
    assert "future" not in result["estimate"]["matched_episode_ids"]
    assert result["validation"]["episodes"] == 12


def test_evidence_selection_prefers_validated_candidate_but_never_automates():
    times = [_stamp(20), _stamp(21)]
    primary = [{"timestamp": t, "point": 10, "q10": 9, "q50": 10, "q90": 11}
               for t in times]
    weak = [{**row, "point": 50, "q50": 50} for row in primary]
    strong = [{**row, "point": 12, "q50": 12} for row in primary]
    result = {"support": "supported", "forecast": primary,
              "context_candidates": [
                  {"hypothesis_id": "weak", "kind": "relationship",
                   "forecast": weak, "validation": {"skill": -.2,
                   "validation_points": 30, "beats_baseline": False}},
                  {"hypothesis_id": "strong", "kind": "relationship",
                   "forecast": strong, "validation": {"skill": .2,
                   "validation_points": 30, "beats_baseline": True}},
              ]}
    payload = publish_result(result, mode="best_effort",
                             automation_policy={"authorize": True,
                                                "policy_id": "ops",
                                                "minimum_support": "supported"})
    assert payload["recommended_scenario_id"] == "fitted-context-2"
    assert payload["recommended_forecast"] == strong
    assert payload["primary_forecast"] == primary
    assert payload["automation"]["eligible"] is False
    assert verify_publication(payload)


def test_candidate_evidence_does_not_trust_model_confidence():
    score = candidate_evidence_score({
        "confidence": 1.0,
        "validation": {"skill": 0, "validation_points": 100,
                       "beats_baseline": False}})
    assert score["score"] == 0
    assert score["decisive"] is False


def test_llm_cannot_override_uniquely_decisive_candidate():
    times = [_stamp(20), _stamp(21)]
    primary = [{"timestamp": t, "point": 10, "q10": 9, "q50": 10, "q90": 11}
               for t in times]
    result = {"support": "supported", "forecast": primary,
              "context_candidates": [{
                  "hypothesis_id": "h", "kind": "relationship",
                  "forecast": [{**row, "q50": 12, "point": 12} for row in primary],
                  "validation": {"skill": .3, "validation_points": 30,
                                 "beats_baseline": True}}]}
    with pytest.raises(ValueError, match="decisive out-of-sample evidence"):
        publish_result(result, mode="best_effort", scenario_selection={
            "selected_scenario_id": "primary",
            "ranking": ["primary", "fitted-context-1"],
            "cited_claim_ids": ["h"], "counterevidence_claim_ids": [],
            "confidence": .9, "rationale": "I prefer the primary",
            "what_would_change_selection": "more evidence",
        })
