from datetime import datetime, timedelta, timezone
import math
import random

import pytest

from gnomon.context_intelligence import (
    align_vintage_rows, candidate_evidence_score, canonicalize_recursive_wrapper,
    compile_context_hypotheses,
    compile_transformation, execute_transformation, TransformationError,
    expand_cited_history_segments,
    fit_historical_analogue, fit_lagged_relationship,
    fit_companion_level_candidate, fit_categorical_state_candidate,
    fit_structured_arx_candidate, fit_vintage_exogenous,
    validate_transformation,
)
from gnomon.publication import publish_result, verify_publication


UTC = timezone.utc


def _stamp(index):
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)).isoformat()


def test_cited_history_segments_require_entailment_complete_nonoverlap():
    timestamps = [datetime(2026, 1, day, tzinfo=UTC) for day in range(1, 5)]
    span = "X is 2.5 from 2026-01-01 to 2026-01-02 and 3.5 from 2026-01-03 to 2026-01-04"
    expanded = expand_cited_history_segments({"X": [
        {"start": "2026-01-01T00:00:00+00:00",
         "end": "2026-01-02T00:00:00+00:00", "value": 2.5,
         "source_claim_ids": ["claim-1"]},
        {"start": "2026-01-03", "end": "2026-01-04", "value": 3.5,
         "source_claim_ids": ["claim-1"]},
    ]}, timestamps=timestamps, cutoff=timestamps[-1],
        claim_spans={"claim-1": span}, allowed_claim_ids=["claim-1"])
    assert expanded == {"X": [2.5, 2.5, 3.5, 3.5]}

    with pytest.raises(TransformationError) as caught:
        expand_cited_history_segments({"X": [{
            "start": "2026-01-01", "end": "2026-01-02", "value": 9,
            "source_claim_ids": ["claim-1"]}]}, timestamps=timestamps,
            cutoff=timestamps[-1], claim_spans={"claim-1": span},
            allowed_claim_ids=["claim-1"])
    assert caught.value.code == "UNENTAILED_HISTORY_RANGE"


CLAIMS = [{"claim_id": "claim-1"}, {"claim_id": "claim-2"}]


def test_absent_hypothesis_is_not_reported_as_rejected():
    hypotheses, critique = compile_context_hypotheses(
        None, claims=CLAIMS, series=["sales"], cutoff=_stamp(5))

    assert hypotheses == []
    assert critique["status"] == "not_proposed"
    assert critique["accepted"] == 0
    assert critique["rejected"] == []


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


def test_companion_level_mapping_replays_against_last_value_and_stays_manual():
    companion = [10, 12, 11, 14, 13, 16, 15, 18]
    target = [value + 2 for value in companion]
    primary = [{"timestamp": _stamp(8 + index), "q10": 14, "q50": 15,
                "q90": 16} for index in range(3)]
    candidate = fit_companion_level_candidate(
        target, companion, [17, 20, 19], primary=primary,
        claim_ids=["claim-1"], hypothesis_id="companion-1")
    assert [row["q50"] for row in candidate["forecast"]] == [19, 22, 21]
    assert candidate["validation"]["beats_baseline"] is True
    assert candidate["selection_eligible"] is True
    assert candidate["automation_eligible"] is False
    assert all(row["q10"] < row["q50"] < row["q90"]
               for row in candidate["forecast"])


def test_companion_level_mapping_rejects_unaligned_or_short_inputs():
    primary = [{"timestamp": _stamp(5), "q50": 1}]
    with pytest.raises(ValueError, match="align"):
        fit_companion_level_candidate(
            [1, 2, 3, 4], [1, 2, 3], [4], primary=primary,
            claim_ids=[], hypothesis_id="h")
    with pytest.raises(ValueError, match="overlapping"):
        fit_companion_level_candidate(
            [1, 2, 3], [1, 2, 3], [4], primary=primary,
            claim_ids=[], hypothesis_id="h")


def test_short_noisy_companion_is_shrunk_and_interval_keeps_raw_displacement():
    companion = [10, 11, 12, 13, 14, 15]
    target = [12.0, 12.8, 14.1, 14.9, 16.2, 16.8]
    primary = [{"timestamp": _stamp(6), "q50": target[-1]}]
    candidate = fit_companion_level_candidate(
        target, companion, [20], primary=primary,
        claim_ids=["claim-1"], hypothesis_id="short")
    validation = candidate["validation"]
    assert validation["validation_points"] == 3
    assert validation["publication_evidence_weight"] == pytest.approx(3 / 8)
    assert validation["publication_shrunk_to_baseline"] is True
    raw_point = 20 + candidate["executable"]["offset"]
    point = candidate["forecast"][0]["q50"]
    assert target[-1] < point < raw_point
    assert candidate["forecast"][0]["q90"] - point >= raw_point - point


def test_governed_companion_candidate_keeps_validation_and_origin():
    span = "On 2026-01-03 the companion value is 12."
    raw = {"claims": [{
        "source_span": span, "relation": "unknown",
        "effective_start": _stamp(2), "effective_end": _stamp(2),
        "confidence": 1.0,
    }]}
    primary = [{"timestamp": _stamp(2), "q10": 9, "q50": 10, "q90": 11}]
    governed = fit_companion_level_candidate(
        [8, 10, 9, 12], [6, 8, 7, 10], [12], primary=primary,
        claim_ids=["claim-1"], hypothesis_id="companion")
    dossier, reasons = __import__(
        "gnomon.llm_dossier", fromlist=["validate_temporal_dossier"]
    ).validate_temporal_dossier(
        raw, context_text=span, cutoff=_stamp(1),
        future_timestamps=[_stamp(2)], history=[8, 10, 9, 12],
        compiler_model="test", governed_candidate=governed)
    assert not reasons
    assert dossier["candidate_critique"]["candidate_origin"] == (
        "governed_companion_mapping")
    assert dossier["forecast_candidate"]["validation"]["mapping"] == (
        "companion_plus_robust_level_difference")
    assert dossier["automation_eligible"] is False
    publication = publish_result({
        "support": "best_effort",
        "forecast": [{**primary[0], "point": primary[0]["q50"]}],
    }, mode="best_effort", dossiers=[dossier])
    candidate = next(item for item in publication["candidate_portfolio"]
                     if item["scenario_id"] != "primary")
    assert candidate["role"] == "governed_companion_mapping"
    assert candidate["effect"]["validation"]["mapping"] == (
        "companion_plus_robust_level_difference")


def test_companion_mapping_admits_signal_and_rejects_independent_walks():
    outcomes = {"signal": 0, "null": 0}
    for seed in range(100):
        generator = random.Random(seed)
        companion, value = [], 0.0
        for _ in range(16):
            value += generator.gauss(0, 1)
            companion.append(value)
        signal = [item + 2 + generator.gauss(0, .15)
                  for item in companion]
        independent, value = [], 0.0
        for _ in range(16):
            value += generator.gauss(0, 1)
            independent.append(value)
        primary = [{"timestamp": f"future-{index}"} for index in range(4)]
        future = [companion[-1] + generator.gauss(0, 1) for _ in range(4)]
        for label, target in (("signal", signal), ("null", independent)):
            candidate = fit_companion_level_candidate(
                target, companion, future, primary=primary,
                claim_ids=["claim-1"], hypothesis_id=label)
            outcomes[label] += int(candidate["selection_eligible"])
    assert outcomes["signal"] >= 95
    assert outcomes["null"] <= 5


def test_categorical_state_mapping_replays_levels_and_stays_manual():
    states = ["open", "open", "closed", "closed"] * 6
    target = [20.0 if state == "open" else 5.0 for state in states]
    primary = [{"timestamp": _stamp(24 + index)} for index in range(4)]

    candidate = fit_categorical_state_candidate(
        target, states, ["closed", "open", "closed", "open"],
        primary=primary, claim_ids=["claim-1"], hypothesis_id="hours")

    points = [row["q50"] for row in candidate["forecast"]]
    assert points[0] < points[1] and points[2] < points[3]
    assert candidate["validation"]["beats_baseline"] is True
    assert candidate["validation"]["all_future_states_observed_twice"] is True
    assert candidate["provenance_class"] == (
        "governed_categorical_state_mapping")
    assert candidate["support"] == "prior_assisted"
    assert candidate["automation_eligible"] is False
    assert candidate["primary_forecast_unchanged"] is True


def test_categorical_state_mapping_rejects_unseen_and_most_null_schedules():
    primary = [{"timestamp": _stamp(20 + index)} for index in range(2)]
    unseen = fit_categorical_state_candidate(
        list(range(20)), ["a", "b"] * 10, ["new", "a"], primary=primary,
        claim_ids=[], hypothesis_id="unseen")
    assert unseen["selection_eligible"] is False
    assert unseen["validation"]["future_state_counts"]["new"] == 0

    admitted = 0
    for seed in range(100):
        generator = random.Random(seed)
        states = [generator.choice(["a", "b"]) for _ in range(32)]
        value, target = 0.0, []
        for _ in states:
            value += generator.gauss(0, 1)
            target.append(value)
        candidate = fit_categorical_state_candidate(
            target, states, ["a", "b"], primary=primary,
            claim_ids=[], hypothesis_id=f"null-{seed}")
        admitted += int(candidate["selection_eligible"])
    assert admitted <= 10


def test_structured_arx_fits_coefficients_and_beats_last_value():
    driver = [float((index * 7) % 17) for index in range(100)]
    target = [2.0, 3.0, 4.0]
    for index in range(3, 100):
        target.append(
            1.5 + .4 * target[index - 1] - .2 * target[index - 2]
            + 1.2 * driver[index - 1])
    future_driver = [3.0, 7.0, 2.0, 9.0]
    primary = [{"timestamp": _stamp(100 + index), "point": target[-1],
                "q10": target[-1] - 2, "q50": target[-1],
                "q90": target[-1] + 2} for index in range(4)]

    candidate = fit_structured_arx_candidate(
        target, {"campaign": driver},
        future_drivers={"campaign": future_driver}, primary=primary,
        autoregressive_lags=[1, 2], driver_lags={"campaign": [1]},
        hypothesis_id="relationship-1")

    validation = candidate["validation"]
    assert validation["beats_baseline"] is True
    assert validation["skill_vs_last_value_baseline"] > .5
    assert validation["specification_known_at_each_origin"] is False
    assert candidate["primary_forecast_unchanged"] is True
    assert candidate["automation_eligible"] is False
    assert len(candidate["forecast"]) == len(primary)
    assert candidate["forecast"][0]["q10"] < candidate["forecast"][0]["q50"] \
        < candidate["forecast"][0]["q90"]


def test_structured_arx_rejects_missing_or_misaligned_drivers():
    primary = [{"timestamp": _stamp(30), "point": 1, "q10": 0,
                "q50": 1, "q90": 2}]
    with pytest.raises(ValueError, match="matching historical and future"):
        fit_structured_arx_candidate(
            [float(index) for index in range(30)], {"x": [1.0] * 30},
            future_drivers={}, primary=primary, autoregressive_lags=[1],
            driver_lags={"x": [1]}, hypothesis_id="h")


def test_fitted_recursive_structure_accepts_lags_but_not_coefficients():
    raw = {
        "known_at": _stamp(20), "claim_ids": ["relationship"],
        "lane": "historically_testable", "output_unit": "requests",
        "expression": {
            "op": "fit_recursive_linear", "output_unit": "requests",
            "autoregressive_lags": [2, 1, 2],
            "driver_lags": [{"series": "campaign", "lags": [2, 1]}],
        },
    }
    compiled = validate_transformation(
        raw, series=["campaign"], claim_ids=["relationship"],
        cutoff=_stamp(20), units={"campaign": "spend"},
        claim_spans={"relationship": "campaign affects requests at lags 1 and 2"})
    assert compiled["expression"]["autoregressive_lags"] == [1, 2]
    assert compiled["expression"]["driver_lags"] == [
        {"series": "campaign", "lags": [1, 2]}]

    raw["expression"]["coefficients"] = [1.0]
    with pytest.raises(TransformationError) as error:
        validate_transformation(
            raw, series=["campaign"], claim_ids=["relationship"],
            cutoff=_stamp(20), units={"campaign": "spend"},
            claim_spans={"relationship": "campaign affects requests at lags 1 and 2"})
    assert error.value.code == "MODEL_AUTHORED_FIT_PARAMETERS"


def test_fitted_recursive_structure_executes_only_with_versioned_future_path():
    driver = [float((index * 5) % 13) for index in range(90)]
    target = [4.0, 5.0]
    for index in range(2, 90):
        target.append(2 + .25 * target[index - 1] + 1.5 * driver[index - 1])
    claim = "driver values will be 3 then 8; driver affects target at lag 1"
    compiled = validate_transformation({
        "known_at": _stamp(89), "claim_ids": ["c1"],
        "lane": "historically_testable", "output_unit": "units",
        "expression": {"op": "fit_recursive_linear",
                       "output_unit": "units", "autoregressive_lags": [1],
                       "driver_lags": [{"series": "driver", "lags": [1]}]},
    }, series=["driver"], claim_ids=["c1"], cutoff=_stamp(89),
       units={"driver": "index"}, claim_spans={"c1": claim})
    primary = [{"timestamp": _stamp(90 + index), "point": target[-1],
                "q10": target[-1] - 2, "q50": target[-1],
                "q90": target[-1] + 2} for index in range(2)]
    candidate = execute_transformation(
        compiled, primary=primary,
        series_values={"driver": {"values": [3, 8], "known_at": _stamp(89),
                                   "source_claim_ids": ["c1"]}},
        claim_spans={"c1": claim}, history_values=target,
        history_series={"driver": driver})
    assert candidate["kind"] == "fitted_structured_arx"
    assert candidate["validation"]["beats_baseline"] is True
    assert candidate["automation_eligible"] is False
    assert candidate["source_seal_sha256"] == compiled["seal_sha256"]

    with pytest.raises(TransformationError) as short:
        execute_transformation(
            compiled, primary=primary,
            series_values={"driver": {"values": [3, 8],
                                       "known_at": _stamp(89),
                                       "source_claim_ids": ["c1"]}},
            claim_spans={"c1": claim}, history_values=target[:12],
            history_series={"driver": driver[:12]})
    assert short.value.code == "INSUFFICIENT_RELATIONSHIP_HISTORY"
    assert "at least" in short.value.message
    assert "12 are available" in short.value.message


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
    assert payload["automation"]["reason_code"] == \
        "recommendation_not_automation_eligible"
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
    with pytest.raises(ValueError, match="evidence-dominant path"):
        publish_result(result, mode="best_effort", scenario_selection={
            "selected_scenario_id": "primary",
            "ranking": ["primary", "fitted-context-1"],
            "cited_claim_ids": ["h"], "counterevidence_claim_ids": [],
            "confidence": .9, "rationale": "I prefer the primary",
            "what_would_change_selection": "more evidence",
        })


def _transform(lane="scenario_only"):
    return {
        "known_at": _stamp(5), "claim_ids": ["claim-1"], "lane": lane,
        "output_unit": "usd",
        "expression": {"op": "add", "args": [
            {"op": "primary", "quantile": "q50"},
            {"op": "literal", "value": 2, "unit": "usd"},
        ]},
    }


def test_safe_transformation_executes_without_code_surface():
    compiled = validate_transformation(
        _transform(), series=[], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"primary": "usd"})
    primary = [{"timestamp": _stamp(6), "q10": 9, "q50": 10,
                "q90": 11, "point": 10},
               {"timestamp": _stamp(7), "q10": 10, "q50": 11,
                "q90": 12, "point": 11}]
    result = execute_transformation(compiled, primary=primary)
    assert [row["q50"] for row in result["forecast"]] == [12, 13]
    assert result["forecast"][0]["q10"] == 11
    assert result["forecast"][0]["q90"] == 13
    assert result["lane"] == "scenario_only"
    assert result["automation_eligible"] is False
    assert result["primary_forecast_unchanged"] is True


def test_transformation_rejects_code_unknown_series_and_future_knowledge():
    for raw, code in [
        ({**_transform(), "expression": {"op": "python", "code": "open('/etc/passwd')"}},
         "UNSAFE_OR_UNKNOWN_OPERATOR"),
        ({**_transform(), "expression": {"op": "series", "name": "secret"}},
         "UNKNOWN_SERIES"),
        ({**_transform(), "known_at": _stamp(9)}, "NOT_KNOWN_AT_CUTOFF"),
    ]:
        with pytest.raises(TransformationError) as caught:
            validate_transformation(raw, series=["sales"],
                                    claim_ids=["claim-1"], cutoff=_stamp(5))
        assert caught.value.code == code


def test_transformation_repair_is_once_and_field_bounded():
    broken = {**_transform(), "output_unit": "items"}
    repaired = _transform()
    compiled, critique = compile_transformation(
        broken, series=[], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"primary": "usd"}, repair=repaired)
    assert compiled is not None
    assert critique["status"] == "repaired"
    assert critique["attempts_used"] == 2

    future = {**broken, "known_at": _stamp(9)}
    rewrite = {**_transform(), "known_at": _stamp(5), "lane": "prior_assisted"}
    compiled, critique = compile_transformation(
        future, series=[], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"primary": "usd"}, repair=rewrite)
    assert compiled is None
    assert critique["violations"][-1]["code"] == "REPAIR_CHANGED_UNRELATED_FIELDS"


def test_historical_transformation_requires_fold_safe_evidence():
    compiled = validate_transformation(
        _transform("historically_testable"), series=[],
        claim_ids=["claim-1"], cutoff=_stamp(5), units={"primary": "usd"})
    primary = [{"timestamp": _stamp(6), "q50": 10, "point": 10}]
    with pytest.raises(TransformationError, match="per-origin"):
        execute_transformation(
            compiled, primary=primary,
            historical_validation={"skill": .2, "validation_points": 20,
                                   "beats_baseline": True})
    result = execute_transformation(
        compiled, primary=primary,
        historical_validation={"skill": .2, "validation_points": 20,
                               "beats_baseline": True,
                               "per_origin_knowledge_checked": True})
    assert candidate_evidence_score(result)["decisive"] is True


def test_transformation_lanes_use_existing_publication_authority_ladder():
    primary = [{"timestamp": _stamp(6), "q50": 10, "point": 10}]
    compiled = validate_transformation(
        _transform("prior_assisted"), series=[], claim_ids=["claim-1"],
        cutoff=_stamp(5), units={"primary": "usd"})
    candidate = execute_transformation(compiled, primary=primary)
    result = {"support": "supported", "forecast": primary,
              "transformation_candidates": [candidate]}
    strict = publish_result(result, mode="strict")
    best = publish_result(result, mode="best_effort")
    assert strict["recommended_scenario_id"] == "primary"
    assert best["recommended_scenario_id"] == "transformation-1"
    assert best["recommended_support"] == "prior_assisted"
    assert best["automation"]["eligible"] is False
    assert best["primary_forecast"] == primary
    assert verify_publication(strict) and verify_publication(best)


def test_future_series_requires_point_in_time_claim_provenance():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "usd",
        "expression": {"op": "multiply", "args": [
            {"op": "series", "name": "units"},
            {"op": "literal", "value": 3, "unit": "usd/items"},
        ]},
    }
    compiled = validate_transformation(
        raw, series=["units"], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"units": "items"})
    primary = [{"timestamp": _stamp(6), "q50": 0, "point": 0},
               {"timestamp": _stamp(7), "q50": 0, "point": 0}]
    with pytest.raises(TransformationError) as caught:
        execute_transformation(compiled, primary=primary,
                               series_values={"units": [4, 5]})
    assert caught.value.code == "UNVERSIONED_FUTURE_SERIES"
    with pytest.raises(TransformationError) as unentailed:
        execute_transformation(compiled, primary=primary, series_values={
            "units": {"values": [4, 5], "known_at": _stamp(5),
                      "source_claim_id": "claim-1"}},
            claim_spans={"claim-1": "units are 4 then 900"})
    assert unentailed.value.code == "UNENTAILED_FUTURE_SERIES_VALUES"
    result = execute_transformation(compiled, primary=primary, series_values={
        "units": {"values": [4, 5], "known_at": _stamp(5),
                  "source_claim_id": "claim-1"}},
        claim_spans={"claim-1": "units are 4 then 5"})
    assert [row["q50"] for row in result["forecast"]] == [12, 15]


def test_future_series_may_union_multiple_verified_schedule_claims():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1", "claim-2"],
        "lane": "prior_assisted", "output_unit": "items",
        "expression": {"op": "series", "name": "schedule"},
    }
    compiled = validate_transformation(
        raw, series=["schedule"], claim_ids=["claim-1", "claim-2"],
        cutoff=_stamp(5), units={"schedule": "items"})
    result = execute_transformation(
        compiled,
        primary=[{"timestamp": _stamp(6), "point": 0, "q50": 0},
                 {"timestamp": _stamp(7), "point": 0, "q50": 0}],
        series_values={"schedule": {
            "values": [4, 9], "known_at": _stamp(5),
            "source_claim_ids": ["claim-1", "claim-2"]}},
        claim_spans={"claim-1": "first period is 4",
                     "claim-2": "second period is 9"})
    assert [row["q50"] for row in result["forecast"]] == [4, 9]


def test_llm_common_ast_aliases_are_canonicalized_without_eval():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "usd^2",
        "expression": {"op": "add", "args": [
            {"op": "pow", "left": {"op": "series", "series": "price"},
             "right": {"op": "literal", "value": 2}},
            {"op": "literal", "value": 1, "unit": "usd^2"},
            {"op": "literal", "value": 2, "unit": "usd^2"},
        ]},
    }
    compiled = validate_transformation(
        raw, series=["price"], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"price": "usd"})
    assert compiled["expression"]["args"][0]["op"] == "power"
    result = execute_transformation(
        compiled,
        primary=[{"timestamp": _stamp(6), "point": 0, "q50": 0}],
        series_values={"price": {"values": [3], "known_at": _stamp(5),
                                 "source_claim_id": "claim-1"}},
        claim_spans={"claim-1": "price is 3"})
    assert result["forecast"][0]["q50"] == 12


def test_power_rejects_nonliteral_or_large_exponents():
    base = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "scenario_only", "output_unit": "unknown",
    }
    for exponent, code in [
        ({"op": "primary"}, "NON_LITERAL_EXPONENT"),
        ({"op": "literal", "value": 99}, "UNSAFE_EXPONENT"),
    ]:
        with pytest.raises(TransformationError) as caught:
            validate_transformation(
                {**base, "expression": {"op": "power", "args": [
                    {"op": "primary"}, exponent]}},
                series=[], claim_ids=["claim-1"], cutoff=_stamp(5))
        assert caught.value.code == code


def test_reference_power_macro_expands_to_safe_canonical_ast():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "Pa",
        "expression": {
            "op": "reference_power", "series": "speed",
            "input_reference": {"value": 3000, "unit": "rpm"},
            "output_reference": {"value": 37.5, "unit": "Pa"},
            "exponent": 2,
        },
    }
    compiled = validate_transformation(
        raw, series=["speed"], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"speed": "rpm"})
    assert compiled["expression"]["op"] == "multiply"
    result = execute_transformation(
        compiled,
        primary=[{"timestamp": _stamp(6), "point": 1, "q50": 1}],
        series_values={"speed": {
            "values": [1500], "known_at": _stamp(5),
            "source_claim_id": "claim-1"}},
        claim_spans={"claim-1": "speed will be 1500 rpm"})
    assert result["forecast"][0]["q50"] == 9.375


@pytest.mark.parametrize("alias", [
    "fraction", "proportion", "probability", "ratio", "share", "unitless",
])
def test_universal_dimensionless_alias_multiplies_a_physical_unit(alias):
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "rpm",
        "expression": {"op": "multiply", "args": [
            {"op": "series", "name": "load"},
            {"op": "literal", "value": 3000, "unit": "rpm"},
        ]},
    }
    compiled = validate_transformation(
        raw, series=["load"], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"load": alias},
        claim_spans={"claim-1": "load is 0.5; full load is 3000 rpm"})
    assert compiled["output_unit"] == "rpm"
    assert compiled["expression"]["args"][0]["name"] == "load"


def test_percent_is_not_silently_treated_as_a_fraction():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "rpm",
        "expression": {"op": "multiply", "args": [
            {"op": "series", "name": "load"},
            {"op": "literal", "value": 3000, "unit": "rpm"},
        ]},
    }
    with pytest.raises(TransformationError) as caught:
        validate_transformation(
            raw, series=["load"], claim_ids=["claim-1"], cutoff=_stamp(5),
            units={"load": "percent"},
            claim_spans={"claim-1": "load is 50 percent; max is 3000 rpm"})
    assert caught.value.code == "OUTPUT_UNIT_MISMATCH"


def test_linear_combination_macro_derives_conversion_units_and_executes():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "revenue",
        "expression": {
            "op": "linear_combination", "output_unit": "revenue",
            "terms": [
                {"coefficient": 3, "series": "units"},
                {"coefficient": -2, "series": "returns"},
            ],
            "intercept": 10,
        },
    }
    compiled = validate_transformation(
        raw, series=["units", "returns"], claim_ids=["claim-1"],
        cutoff=_stamp(5), units={"units": "items", "returns": "items"},
        claim_spans={"claim-1":
                     "revenue = 3 units - 2 returns + 10; future units 5, returns 1"})
    assert compiled["expression"]["op"] == "add"
    result = execute_transformation(
        compiled,
        primary=[{"timestamp": _stamp(6), "point": 0, "q50": 0}],
        series_values={
            "units": {"values": [5], "known_at": _stamp(5),
                      "source_claim_id": "claim-1"},
            "returns": {"values": [1], "known_at": _stamp(5),
                        "source_claim_id": "claim-1"},
        },
            claim_spans={"claim-1":
                         "revenue = 3 units - 2 returns + 10; future units 5, returns 1"})
    assert result["forecast"][0]["q50"] == 23


def test_linear_combination_rejects_unknown_series_and_unentailed_coefficient():
    base = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "revenue",
    }
    with pytest.raises(TransformationError) as unknown:
        validate_transformation({**base, "expression": {
            "op": "linear_combination", "output_unit": "revenue",
            "terms": [{"coefficient": 3, "series": "missing"}]}},
            series=["units"], claim_ids=["claim-1"], cutoff=_stamp(5),
            units={"units": "items"})
    assert unknown.value.code == "UNKNOWN_SERIES"
    with pytest.raises(TransformationError) as unentailed:
        validate_transformation({**base, "expression": {
            "op": "linear_combination", "output_unit": "revenue",
            "terms": [{"coefficient": 99, "series": "units"}]}},
            series=["units"], claim_ids=["claim-1"], cutoff=_stamp(5),
            units={"units": "items"},
            claim_spans={"claim-1": "revenue is three times units"})
    assert unentailed.value.code == "UNENTAILED_TRANSFORMATION_CONSTANT"


def test_percent_prose_entails_dimensionless_multiplier():
    compiled = validate_transformation({
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "scenario_only", "output_unit": "percent",
        "expression": {"op": "multiply", "args": [
            {"op": "primary"}, {"op": "literal", "value": 0.1},
        ]},
    }, series=[], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"primary": "percent"},
        claim_spans={"claim-1": "Traffic will be 10.0% of usual."})

    assert compiled["expression"]["args"][1]["value"] == 0.1


def test_inapplicable_null_lag_is_zero_for_historical_analogue():
    accepted, critique = compile_context_hypotheses([{
        "kind": "historical_analogue", "claim_ids": ["claim-1"],
        "target_series": ["*"], "predictor_series": None,
        "known_at": _stamp(5), "lag_steps": None,
        "direction": "unknown", "rationale": "One cited reference.",
    }], claims=[{"claim_id": "claim-1"}], series=["target"],
        cutoff=_stamp(5))

    assert critique["status"] == "accepted"
    assert accepted[0]["lag_steps"] == 0


def test_ordinary_additive_equation_derives_only_forced_coefficient_units():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "revenue",
        "expression": {"op": "add", "args": [
            {"op": "multiply", "args": [
                {"op": "literal", "value": 3},
                {"op": "lag", "args": [
                    {"op": "series", "name": "units"}], "steps": 1}]},
            {"op": "literal", "value": 10},
        ]},
    }
    compiled = validate_transformation(
        raw, series=["units"], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"units": "items"},
        claim_spans={"claim-1": "revenue = 3 units lagged 1 step + 10"})
    coefficient = compiled["expression"]["args"][0]["args"][0]
    assert coefficient["unit"] == "revenue/items"
    assert compiled["expression"]["args"][1]["unit"] == "revenue"
    assert compiled["validation"]["coefficient_units_derived"] is True


def test_standalone_product_does_not_guess_a_target_conversion_unit():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "revenue",
        "expression": {"op": "multiply", "args": [
            {"op": "literal", "value": 3},
            {"op": "series", "name": "units"},
        ]},
    }
    with pytest.raises(TransformationError) as caught:
        validate_transformation(
            raw, series=["units"], claim_ids=["claim-1"], cutoff=_stamp(5),
            units={"units": "items"},
            claim_spans={"claim-1": "revenue = 3 units"})
    assert caught.value.code == "OUTPUT_UNIT_MISMATCH"


def test_recursive_linear_uses_trusted_history_and_propagates_uncertainty():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "sales",
        "expression": {
            "op": "recursive_linear", "output_unit": "sales",
            "intercept": 2,
            "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
            "driver_terms": [{"series": "campaign", "lag": 1,
                              "coefficient": 3}],
        },
    }
    span = "sales[t] = 2 + 0.5 sales[t-1] + 3 campaign[t-1]"
    compiled = validate_transformation(
        raw, series=["campaign"], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"campaign": "spend", "primary": "sales"},
        claim_spans={"claim-1": span})
    assert compiled["expression"]["driver_terms"][0]["coefficient_unit"] \
        == "sales/spend"
    primary = [
        {"timestamp": _stamp(6), "point": 0, "q10": -1, "q50": 0, "q90": 1},
        {"timestamp": _stamp(7), "point": 0, "q10": -1, "q50": 0, "q90": 1},
    ]
    result = execute_transformation(
        compiled, primary=primary,
        series_values={"campaign": {
            "values": [5, 7], "known_at": _stamp(5),
            "source_claim_id": "claim-1"}},
        claim_spans={"claim-1": span + "; campaign schedule is 5 then 7"},
        history_values=[10], history_series={"campaign": [4]})
    # t0 = 2 + .5*10 + 3*4 = 19; t1 = 2 + .5*19 + 3*5 = 26.5.
    assert [row["q50"] for row in result["forecast"]] == [19, 26.5]
    assert result["forecast"][0]["q10"] == 18
    assert result["forecast"][1]["q10"] == pytest.approx(
        26.5 - math.sqrt(1.25))
    assert result["primary_forecast_unchanged"] is True
    assert result["validation"]["recurrence_uncertainty"] \
        == "linear_state_covariance"
    assert result["validation"]["recurrence_plausibility_passed"] is True
    assert result["validation"]["recurrence_replay_admitted"] is False


def test_recursive_linear_replay_is_selected_without_false_knowledge_claim():
    raw = {
        "known_at": _stamp(20), "claim_ids": ["claim-1"],
        "lane": "historically_testable", "output_unit": "sales",
        "expression": {
            "op": "recursive_linear", "output_unit": "sales",
            "intercept": 1,
            "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
            "driver_terms": [{"series": "campaign", "lag": 1,
                              "coefficient": 2}],
        },
    }
    span = "sales[t] = 1 + 0.5 sales[t-1] + 2 campaign[t-1]"
    compiled = validate_transformation(
        raw, series=["campaign"], claim_ids=["claim-1"], cutoff=_stamp(20),
        units={"campaign": "spend", "primary": "sales"},
        claim_spans={"claim-1": span})
    campaign = [float(index % 3) for index in range(20)]
    sales = [10.0]
    for index in range(1, 20):
        sales.append(1 + .5 * sales[-1] + 2 * campaign[index - 1])
    primary = [{"timestamp": _stamp(21 + index), "point": sales[-1],
                "q10": sales[-1] - 1, "q50": sales[-1],
                "q90": sales[-1] + 1} for index in range(2)]
    candidate = execute_transformation(
        compiled, primary=primary,
        series_values={"campaign": {"values": [1, 2],
                                     "known_at": _stamp(20),
                                     "source_claim_id": "claim-1"}},
        claim_spans={"claim-1": span + "; campaign schedule is 1 then 2"},
        history_values=sales, history_series={"campaign": campaign})
    validation = candidate["validation"]
    assert validation["recurrence_replay_points"] == 19
    assert validation["recurrence_replay_candidate_mae"] == pytest.approx(0)
    assert validation["recurrence_replay_admitted"] is True
    assert validation["per_origin_knowledge_checked"] is False
    assert validation["per_origin_observation_availability_checked"] is True
    assert validation["specification_known_at_each_origin"] is False
    assert validation["validation_interpretation"] == \
        "retrospective_fixed_specification_replay"
    publication = publish_result(
        {"support": "supported", "forecast": primary,
         "transformation_candidates": [candidate]}, mode="best_effort")
    assert publication["recommended_scenario_id"] == "transformation-1"
    admitted = next(item for item in publication["candidate_portfolio"]
                    if item["scenario_id"] == "transformation-1")
    assert admitted["role"] == "retrospectively_validated"
    assert admitted["support"] == "conditionally_supported"
    authority = publication["recommendation_authority"]
    assert authority["selection_method"] == \
        "retrospective_fixed_specification_evidence"
    assert authority["historically_admitted"] is False
    assert authority["human_review_required"] is True


def test_recursive_linear_that_loses_replay_remains_visible_not_recommended():
    raw = {
        "known_at": _stamp(20), "claim_ids": ["claim-1"],
        "lane": "historically_testable", "output_unit": "sales",
        "expression": {"op": "recursive_linear", "output_unit": "sales",
                       "autoregressive_terms": [
                           {"lag": 1, "coefficient": -.5}],
                       "driver_terms": []},
    }
    compiled = validate_transformation(
        raw, series=[], claim_ids=["claim-1"], cutoff=_stamp(20),
        units={"primary": "sales"},
        claim_spans={"claim-1": "sales[t] = -0.5 sales[t-1]"})
    history = [10 + index for index in range(20)]
    primary = [{"timestamp": _stamp(21), "point": 30, "q10": 29,
                "q50": 30, "q90": 31}]
    candidate = execute_transformation(
        compiled, primary=primary, history_values=history)
    assert candidate["validation"]["recurrence_replay_admitted"] is False
    publication = publish_result(
        {"support": "supported", "forecast": primary,
         "transformation_candidates": [candidate]}, mode="best_effort")
    assert publication["recommended_scenario_id"] == "primary"
    retained = next(item for item in publication["candidate_portfolio"]
                    if item["scenario_id"] == "transformation-1")
    assert retained["selection_eligible"] is False


def test_recursive_linear_refuses_missing_trusted_initial_state():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "sales",
        "expression": {
            "op": "recursive_linear", "output_unit": "sales",
            "autoregressive_terms": [{"lag": 2, "coefficient": .5}],
            "driver_terms": [],
        },
    }
    compiled = validate_transformation(
        raw, series=[], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"primary": "sales"},
        claim_spans={"claim-1": "sales[t] = 0.5 sales[t-2]"})
    with pytest.raises(TransformationError) as caught:
        execute_transformation(
            compiled,
            primary=[{"timestamp": _stamp(6), "q50": 0, "q10": -1, "q90": 1}],
            history_values=[10])
    assert caught.value.code == "MISSING_RECURSIVE_HISTORY"


def test_explosive_recurrence_is_retained_but_not_recommendable():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "sales",
        "expression": {"op": "recursive_linear", "output_unit": "sales",
                       "autoregressive_terms": [
                           {"lag": 1, "coefficient": 2}],
                       "driver_terms": []},
    }
    compiled = validate_transformation(
        raw, series=[], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"primary": "sales"},
        claim_spans={"claim-1": "sales[t] = 2 sales[t-1]"})
    primary = [{"timestamp": _stamp(6 + i), "q10": 9, "q50": 10,
                "q90": 11, "point": 10} for i in range(8)]
    candidate = execute_transformation(
        compiled, primary=primary, history_values=[10])
    assert candidate["validation"]["recurrence_stable"] is False
    publication = publish_result(
        {"support": "supported", "forecast": primary,
         "transformation_candidates": [candidate]}, mode="best_effort")
    assert publication["recommended_scenario_id"] == "primary"
    retained = next(item for item in publication["candidate_portfolio"]
                    if item["role"] == "model_authored_transformation")
    assert retained["selection_eligible"] is False


def test_zero_recursive_intercept_is_safe_identity_not_unsourced_effect():
    compiled = validate_transformation({
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "sales",
        "expression": {
            "op": "recursive_linear", "output_unit": "sales",
            "intercept": 0,
            "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
            "driver_terms": [],
        }}, series=[], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"primary": "sales"},
        claim_spans={"claim-1": "sales[t] = 0.5 sales[t-1]"})
    assert compiled["expression"]["intercept"] == 0


def test_verbose_lag_arrays_canonicalize_to_recursion_without_target_values():
    wrapper = {
        "transformation": {
            "known_at": _stamp(5), "claim_ids": ["claim-1"],
            "lane": "prior_assisted", "output_unit": "sales",
            "expression": {"op": "add", "args": [
                {"op": "multiply", "args": [
                    {"op": "literal", "value": .5},
                    {"op": "series", "name": "sales_lag1"}]},
                {"op": "multiply", "args": [
                    {"op": "literal", "value": 2},
                    {"op": "series", "name": "campaign_lag1"}]},
            ]},
        },
        "units": {"primary": "sales", "sales_lag1": "sales",
                  "campaign_lag1": "spend"},
        "series_values": {
            "sales_lag1": {"values": [999, 999]},
            "campaign_lag1": {"values": [4, 5], "known_at": _stamp(5),
                              "source_claim_ids": ["claim-1"]},
        },
    }
    canonical, status = canonicalize_recursive_wrapper(
        wrapper, target_name="sales", driver_names=["campaign"])
    assert status["status"] == "canonicalized"
    expression = canonical["transformation"]["expression"]
    assert expression["op"] == "recursive_linear"
    assert expression["autoregressive_terms"] == [{"lag": 1, "coefficient": .5}]
    assert expression["driver_terms"] == [
        {"series": "campaign", "lag": 1, "coefficient": 2}]
    assert set(canonical["series_values"]) == {"campaign"}
    assert 999 not in canonical["series_values"]["campaign"]["values"]


def test_verbose_recurrence_refuses_conflicting_driver_schedules():
    wrapper = {
        "transformation": {"output_unit": "y", "expression": {
            "op": "add", "args": [
                {"op": "multiply", "args": [
                    {"op": "literal", "value": .5},
                    {"op": "series", "name": "y_lag1"}]},
                {"op": "multiply", "args": [
                    {"op": "literal", "value": 2},
                    {"op": "series", "name": "x_lag1"}]},
                {"op": "multiply", "args": [
                    {"op": "literal", "value": 3},
                    {"op": "series", "name": "x_lag2"}]},
            ]}},
        "series_values": {
            "x_lag1": {"values": [1, 2]},
            "x_lag2": {"values": [9, 9]},
        },
    }
    unchanged, status = canonicalize_recursive_wrapper(
        wrapper, target_name="y", driver_names=["x"])
    assert unchanged == wrapper
    assert status["status"] == "rejected"


@pytest.mark.parametrize("alias", ["x_0_future", "future_x0"])
def test_recursive_future_alias_rebinds_to_governed_driver_identity(alias):
    wrapper = {
        "transformation": {"output_unit": "y", "expression": {
            "op": "recursive_linear", "output_unit": "y", "intercept": 0,
            "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
            "driver_terms": [{"series": alias, "lag": 1,
                              "coefficient": 2}]}},
        "units": {"primary": "y", alias: "x"},
        "series_values": {alias: {
            "values": [1, 2], "known_at": _stamp(5),
            "source_claim_ids": ["claim-1"]}},
    }
    canonical, status = canonicalize_recursive_wrapper(
        wrapper, target_name="X_1", driver_names=["X_0"])
    assert status["status"] == "canonicalized"
    assert canonical["transformation"]["expression"]["driver_terms"][0][
        "series"] == "X_0"


def test_recursive_duplicate_lag_series_names_collapse_only_when_schedules_agree():
    wrapper = {
        "transformation": {"expression": {
            "op": "recursive_linear", "output_unit": "y", "intercept": 0,
            "autoregressive_terms": [{"lag": 1, "coefficient": .5}],
            "driver_terms": [
                {"series": "X_0_lag1", "lag": 1, "coefficient": 2},
                {"series": "X_0_lag2", "lag": 2, "coefficient": 3},
            ]}},
        "units": {"primary": "y", "X_0_lag1": "x", "X_0_lag2": "x"},
        "series_values": {
            "X_0_lag1": {"values": [1, 2], "known_at": _stamp(5),
                           "source_claim_ids": ["claim-1"]},
            "X_0_lag2": {"values": [1, 2], "known_at": _stamp(5),
                           "source_claim_ids": ["claim-1"]},
        },
    }
    canonical, status = canonicalize_recursive_wrapper(
        wrapper, target_name="X_1",
        driver_names=["X_0_lag1", "X_0_lag2"])
    assert status == {"status": "canonicalized", "target": "X_1",
                      "drivers": ["X_0"]}
    assert {term["series"] for term in canonical["transformation"][
        "expression"]["driver_terms"]} == {"X_0"}
    assert set(canonical["series_values"]) == {"X_0"}

    wrapper["series_values"]["X_0_lag2"]["values"] = [9, 9]
    unchanged, rejected = canonicalize_recursive_wrapper(
        wrapper, target_name="X_1",
        driver_names=["X_0_lag1", "X_0_lag2"])
    assert unchanged is wrapper
    assert rejected["status"] == "rejected"
    assert "conflicting schedules" in rejected["reason"]
    assert set(canonical["series_values"]) == {"X_0"}
    assert canonical["units"]["X_0"] == "x"


def test_nested_lag_future_series_canonicalizes_to_feedback_and_driver():
    wrapper = {
        "transformation": {"output_unit": "y", "expression": {
            "op": "add", "args": [
                {"op": "multiply", "args": [
                    {"op": "literal", "value": .5},
                    {"op": "lag", "steps": 1, "args": [
                        {"op": "series", "name": "future_x1"}]}]},
                {"op": "multiply", "args": [
                    {"op": "literal", "value": 2},
                    {"op": "lag", "steps": 1, "args": [
                        {"op": "series", "name": "future_x0"}]}]},
            ]}},
        "units": {"primary": "y", "future_x0": "x", "future_x1": "y"},
        "series_values": {
            "future_x0": {"values": [1, 2], "source_claim_ids": ["claim-1"]},
            "future_x1": {"values": [999, 999], "source_claim_ids": ["claim-1"]},
        },
    }
    canonical, status = canonicalize_recursive_wrapper(
        wrapper, target_name="X_1", driver_names=["X_0"])
    assert status["status"] == "canonicalized"
    expression = canonical["transformation"]["expression"]
    assert expression["autoregressive_terms"] == [{"lag": 1, "coefficient": .5}]
    assert expression["driver_terms"] == [
        {"series": "X_0", "lag": 1, "coefficient": 2}]
    assert set(canonical["series_values"]) == {"X_0"}


def test_model_computed_constant_cannot_launder_through_claim_id():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "Pa",
        "expression": {"op": "multiply", "args": [
            {"op": "primary", "quantile": "q50"},
            {"op": "literal", "value": 31.2},
        ]},
    }
    with pytest.raises(TransformationError) as caught:
        validate_transformation(
            raw, series=[], claim_ids=["claim-1"], cutoff=_stamp(5),
            units={"primary": "Pa"},
            claim_spans={"claim-1": "pressure follows the square of speed"})
    assert caught.value.code == "UNENTAILED_TRANSFORMATION_CONSTANT"


def test_textual_square_entails_reference_law_exponent():
    raw = {
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "Pa",
        "expression": {
            "op": "reference_power", "series": "speed",
            "input_reference": {"value": 3000, "unit": "rpm"},
            "output_reference": {"value": 37.5, "unit": "Pa"},
            "exponent": 2,
        },
    }
    compiled = validate_transformation(
        raw, series=["speed"], claim_ids=["claim-1"], cutoff=_stamp(5),
        units={"speed": "rpm"}, claim_spans={
            "claim-1": "Pressure is 37.5 Pa at 3000 rpm and follows the square of speed."})
    assert compiled["validation"]["constants_entailed"] is True


def test_root_literal_inherits_explicit_output_unit_only():
    compiled = validate_transformation({
        "known_at": _stamp(5), "claim_ids": ["claim-1"],
        "lane": "prior_assisted", "output_unit": "items",
        "expression": {"op": "literal", "value": 12}},
        series=[], claim_ids=["claim-1"], cutoff=_stamp(5))
    assert compiled["expression"]["unit"] == "items"
