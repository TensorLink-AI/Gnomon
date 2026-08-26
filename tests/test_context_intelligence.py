from datetime import datetime, timedelta, timezone
import math

import pytest

from gnomon.context_intelligence import (
    align_vintage_rows, candidate_evidence_score, canonicalize_recursive_wrapper,
    compile_context_hypotheses,
    compile_transformation, execute_transformation, TransformationError,
    fit_historical_analogue, fit_lagged_relationship, fit_vintage_exogenous,
    validate_transformation,
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
