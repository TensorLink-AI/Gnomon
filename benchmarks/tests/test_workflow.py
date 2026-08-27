import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.workflow.run_workflow import DEFAULT_CASES, case_payload
from benchmarks.workflow.schema import Case, Observation, load_cases
from benchmarks.workflow.scoring import score_run
from benchmarks.workflow.compare import compare
from benchmarks.workflow.audit import audit
from benchmarks.workflow.provenance import corpus_sha256
from benchmarks.workflow.generate import generate_publication_cases
from benchmarks.workflow.agent_adapter import (
    _compile_execution_arguments, _publication_recommendation_numbers)


def _observation(case, **overrides):
    values = {
        "case_id": case.id, "status": "answered", "support": "supported",
        "numbers": case.oracle.numbers, "choices": case.oracle.choices,
        "disclosures": list(case.oracle.required_disclosures),
        "publish_matches_evaluated": True,
        "temporal_leakage": False,
        "evaluated_fingerprint": "same", "published_fingerprint": "same",
        "artifact_numbers": case.oracle.numbers,
        "headline_numbers": case.oracle.numbers,
        "facts": case.oracle.required_facts,
        "stage_results": {
            "repair": {"completed": True, "followup_status": "answered"},
            "outcome": {"tracked": True, "absolute_error": 0.0,
                        "reported_absolute_error": 0.0}},
        "repair_completed": True, "tracking_completed": True,
        "quote_matches": True, "tool_calls": 1, "cumulative_tokens": 100,
        "metadata": {"leakage_measurement": "cutoff_projection_v1",
                     "cutoff_projection_sha256": "a" * 64},
    }
    if case.oracle.should_abstain:
        values.update(status="abstained", support="abstained")
    elif values["support"] not in case.oracle.allowed_support:
        values["support"] = case.oracle.allowed_support[0]
    values.update(overrides)
    return Observation.from_dict(values)


def test_smoke_corpus_covers_all_case_kinds_and_hides_oracle():
    cases = load_cases(DEFAULT_CASES)
    assert {case.kind for case in cases} == {
        "synthetic", "frozen", "messy", "longitudinal", "multiseries"
    }
    assert all("oracle" not in case_payload(case) for case in cases)


def test_context_interface_corpus_scores_the_engine_contract_separately():
    path = Path(__file__).parents[1] / "workflow" / "cases" / \
        "context-interface.jsonl"
    cases = load_cases(path)
    assert len(cases) == 4
    observations = []
    for case in cases:
        expected = case.oracle.context_behavior
        mode = (expected.get("publication_mode")
                or (expected.get("allowed_publication_modes") or [None])[0])
        status = (expected.get("status")
                  or (expected.get("allowed_statuses") or [None])[0])
        recommended = (expected.get("recommended_scenario_id")
                       or (expected.get("recommended_scenario_by_mode") or {}).get(
                           mode))
        observations.append(_observation(case, metadata={
            "leakage_measurement": "cutoff_projection_v1",
            "cutoff_projection_sha256": "a" * 64,
            "surface_required_calls": 1,
            "context_arguments": [expected["required_argument"],
                                  "publication_mode"],
            "context_behavior": {
                "publication_mode": mode, "status": status,
                "recommended_scenario_id": recommended,
                "primary_forecast_unchanged": expected[
                    "primary_forecast_unchanged"],
                "automation_eligible": expected["automation_eligible"],
                "scenario_count": expected["minimum_scenario_count"],
            },
        }))
    result = score_run(cases, observations, "fixture")
    assert result["context_contract"] == {
        "required_cases": 4, "passed_cases": 4, "pass_rate": 1.0,
    }
    assert all(row["context_contract"]["pass"] for row in result["rows"])


def test_adversarial_context_corpus_has_explicit_safe_dispositions():
    path = Path(__file__).parents[1] / "workflow" / "cases" / \
        "context-adversarial.jsonl"
    cases = load_cases(path)
    assert len(cases) == 4
    assert all("rejected" in case.oracle.context_behavior["allowed_statuses"]
               for case in cases)
    assert all(case.oracle.context_behavior["primary_forecast_unchanged"]
               is True for case in cases)
    assert all(case.oracle.context_behavior["automation_eligible"] is False
               for case in cases)
    assert all("context-interface" in case.tags for case in cases)


def test_mixed_context_corpus_requires_multiple_disposition_channels():
    path = Path(__file__).parents[1] / "workflow" / "cases" / \
        "context-mixed.jsonl"
    cases = load_cases(path)
    assert len(cases) == 3
    assert all(len(case.oracle.context_behavior["required_arguments"]) == 2
               for case in cases)
    assert {case.oracle.context_behavior["allowed_statuses"][0]
            for case in cases} == {
        "used", "partially_used", "partially_represented"}


def test_publication_recommendation_overrides_active_artifact_lane_for_answer():
    publication = {
        "recommended_scenario_id": "primary",
        "selection_contract": {"scenarios": [
            {"scenario_id": "primary", "summary": {"first_q50": 70.72}},
            {"scenario_id": "context_conditioned",
             "summary": {"first_q50": 40.0}},
        ]},
    }
    assert _publication_recommendation_numbers(publication, {"next"}) == {
        "next": 70.72}
    assert _publication_recommendation_numbers(publication, set()) == {}


def test_execution_compiler_binds_known_fields_but_preserves_ambiguity(tmp_path):
    base = {
        "kind": "synthetic",
        "available_at_cutoff": {"cutoff": "2026-01-01", "series": [1, 2]},
    }
    result = _compile_execution_arguments(
        base, "gnomon_forecast", {"input": "/invented"},
        tmp_path / "history.csv", tmp_path,
    )
    assert result["input"] == str(tmp_path / "history.csv")
    assert result["target_column"] == "value"
    assert result["horizon"] == 1
    assert result["minimum_support"] == "best_effort"
    assert "data_ref" not in _compile_execution_arguments(
        base, "gnomon_forecast", {"data_ref": "stale", "series_column": "x"},
        tmp_path / "history.csv", tmp_path,
    )

    governed = _compile_execution_arguments(
        base, "gnomon_forecast", {
            "input": "/invented", "output_dir": "/invented-output",
            "context_submission": {
                "text": "A closure is scheduled tomorrow.",
                "known_at": "2026-01-01T00:00:00+00:00",
                "compiler": "host-model", "proposal": {"events": []},
            },
            "publication_mode": "scenario",
            "automation_policy": {"allow": False},
            "future_events": True,
        }, tmp_path / "history.csv", tmp_path)
    assert governed["input"] == str(tmp_path / "history.csv")
    assert governed["output_dir"] == str(tmp_path / "gnomon-output")
    assert governed["context_submission"]["compiler"] == "host-model"
    assert governed["publication_mode"] == "scenario"
    assert governed["automation_policy"] == {"allow": False}
    assert governed["future_events"] is True

    scoped = _compile_execution_arguments(
        base, "gnomon_forecast", {
            "context_events": [{"event_id": "closure",
                                "entity_scope": ["demand"]}],
            "qualitative_context_events": [{"event_id": "campaign",
                                             "entity_scope": ["sales"]}],
        }, tmp_path / "history.csv", tmp_path)
    assert scoped["context_events"][0]["entity_scope"] == ["value"]
    assert scoped["qualitative_context_events"][0]["entity_scope"] == ["value"]

    ambiguous = {
        "kind": "messy",
        "available_at_cutoff": {
            "cutoff": "2026-01-01", "columns": {"a": [1], "b": [2]}
        },
    }
    result = _compile_execution_arguments(
        ambiguous, "gnomon_forecast", {"target_column": "a"},
        tmp_path / "history.csv", tmp_path,
    )
    assert "target_column" not in result


def test_perfect_matched_run_passes_release_gate():
    cases = load_cases(DEFAULT_CASES)
    result = score_run(cases, [_observation(case) for case in cases], "fixture")
    assert result["correctness_mean_all_cases"] == 1.0
    assert result["trust_pass_rate_all_cases"] == 1.0
    assert result["usability_pass_rate_all_cases"] == 1.0
    assert result["release_gate_pass"] is True
    assert result["economics"]["calls_median"] == 1


def test_leak_is_a_hard_gate_and_abstention_cannot_inflate_accuracy():
    cases = load_cases(DEFAULT_CASES)
    observations = [_observation(case) for case in cases]
    observations[0] = _observation(cases[0], temporal_leakage=True)
    observations[1] = _observation(cases[1], status="abstained", support="abstained")
    result = score_run(cases, observations, "bad")
    assert result["temporal_leaks"] == 1
    assert result["release_gate_pass"] is False
    assert result["correctness_mean_all_cases"] < 1.0


def test_missing_cases_are_in_all_case_denominator():
    cases = load_cases(DEFAULT_CASES)
    result = score_run(cases, [_observation(cases[0])], "partial")
    assert len(result["missing"]) == 4
    assert result["correctness_mean_all_cases"] == pytest.approx(0.2)
    assert result["release_gate_pass"] is False
    assert result["completeness_gate_pass"] is False
    assert result["leakage_safety_gate_pass"] is True


def test_correctness_and_trust_components_are_reported_separately():
    cases = load_cases(DEFAULT_CASES)
    observations = [_observation(case) for case in cases]
    observations[0] = _observation(cases[0], choices={"pattern": "wrong"})
    result = score_run(cases, observations, "component-report")
    assert result["correctness_components"]["numeric"] == 1.0
    assert result["correctness_components"]["semantic"] < 1.0
    assert set(result["trust_component_pass_rates"]) == {
        "leakage_measured_safe", "disclosures", "forbidden_claims",
        "support", "publish_parity", "quote_fidelity",
    }


def test_contract_rejects_future_key_and_bad_abstention_support():
    with pytest.raises(ValueError, match="future key"):
        Case.from_dict({"id": "x", "kind": "synthetic", "question": "x",
                        "available_at_cutoff": {"future": [1]},
                        "answer_schema": {}, "oracle": {}})
    with pytest.raises(ValueError, match="requires abstained support"):
        Observation.from_dict({"case_id": "x", "status": "abstained",
                               "support": "supported"})
    with pytest.raises(ValueError, match="unknown fields"):
        Observation.from_dict({"case_id": "x", "status": "answered",
                               "support": "supported", "tokenz": 3})


def test_jsonl_case_validation_reports_duplicate_ids(tmp_path):
    row = {"id": "x", "kind": "synthetic", "question": "x",
           "available_at_cutoff": {}, "answer_schema": {}, "oracle": {}}
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_corpus_hash_is_stable_and_oracle_sensitive():
    cases = load_cases(DEFAULT_CASES)
    assert corpus_sha256(cases) == corpus_sha256(load_cases(DEFAULT_CASES))
    changed = list(cases)
    source = json.loads(DEFAULT_CASES.read_text().splitlines()[0])
    source["oracle"]["numbers"]["next"] = 11
    changed[0] = Case.from_dict(source)
    assert corpus_sha256(changed) != corpus_sha256(cases)


def test_comparison_requires_matched_corpus_and_uses_conservative_gate():
    cases = load_cases(DEFAULT_CASES)
    base = score_run(cases, [_observation(case) for case in cases], "base")
    treatment = score_run(cases, [
        _observation(case, cumulative_tokens=1000, tool_calls=1)
        for case in cases
    ], "core")
    result = compare([base, treatment], "base")
    assert result["eligible_arms"] == ["base", "core"]
    treatment["corpus_sha256"] = "different"
    with pytest.raises(ValueError, match="corpus_sha256"):
        compare([base, treatment], "base")


def test_comparison_aggregates_replicates_by_case():
    cases = load_cases(DEFAULT_CASES)
    run1 = score_run(cases, [_observation(case) for case in cases], "base")
    run2 = score_run(cases, [_observation(case) for case in cases], "base")
    treatment = score_run(cases, [_observation(case) for case in cases], "evidence")
    result = compare([run1, run2, treatment], "base")
    by_arm = {row["arm"]: row for row in result["arms"]}
    assert by_arm["base"]["replicates"] == 2
    assert by_arm["evidence"]["replicates"] == 1


def test_comparison_averages_stage_economics_across_replicates():
    cases = load_cases(DEFAULT_CASES)
    first = score_run(cases, [_observation(case, stage_results={
        "initial": {"economics": {"tool_calls": 1, "cumulative_tokens": 10,
                                  "response_tokens": 2, "latency_seconds": 1}}})
                              for case in cases], "base")
    second = score_run(cases, [_observation(case, stage_results={
        "initial": {"economics": {"tool_calls": 3, "cumulative_tokens": 30,
                                  "response_tokens": 4, "latency_seconds": 3}}})
                               for case in cases], "base")
    arm = compare([first, second], "base")["arms"][0]
    assert arm["initial_calls_median"] == 2


def test_decision_readiness_requires_a_real_leaktrap_gate():
    cases = load_cases(DEFAULT_CASES)
    base = score_run(cases, [_observation(case) for case in cases], "base")
    assert compare([base], "base")["decision_ready_arms"] == []
    leaktrap = {"benchmark": "leakage-trap", "condition": "gnomon",
                "tasks": 40, "tasks_flagged_as_leaking": 0,
                "tasks_transcribing_the_future": 0,
                "structural_claim_proven": 40}
    assert compare([base], "base", leaktrap=leaktrap)["decision_ready_arms"] == []
    candidate = score_run(cases, [_observation(case) for case in cases], "mega")
    accuracy = {"mega": {"comparable": True, "benchmark": "temporalbench",
                         "matched_tasks": 40,
                         "metrics": {"SMAPE": {"lower_is_better": True,
                                                "baseline_mean": 10.0,
                                                "treatment_mean": 10.1}}}}
    heldout = {"generator": "workflow-synthetic-v2", "fresh_seed": True,
               "seed": 123, "corpus_sha256": base["corpus_sha256"]}
    # One stochastic sample is never enough to select a surface.
    assert compare([base, candidate], "base", leaktrap=leaktrap,
                   accuracy_comparisons=accuracy,
                   heldout_manifest=heldout)["decision_ready_arms"] == []
    assert compare([base] * 3 + [candidate] * 3, "base", leaktrap=leaktrap,
                   accuracy_comparisons=accuracy,
                   heldout_manifest=heldout)["decision_ready_arms"] == ["mega"]


def test_decision_readiness_rejects_reused_or_mismatched_corpus_manifest():
    cases = load_cases(DEFAULT_CASES)
    base = score_run(cases, [_observation(case) for case in cases], "base")
    candidate = score_run(cases, [_observation(case) for case in cases], "mega")
    leaktrap = {"benchmark": "leakage-trap", "condition": "gnomon", "tasks": 40,
                "tasks_flagged_as_leaking": 0, "tasks_transcribing_the_future": 0,
                "structural_claim_proven": 40}
    accuracy = {"mega": {"comparable": True, "benchmark": "temporalbench",
                         "matched_tasks": 40,
                         "metrics": {"SMAPE": {"lower_is_better": True,
                                                "baseline_mean": 10,
                                                "treatment_mean": 10}}}}
    reused = {"generator": "workflow-synthetic-v2", "fresh_seed": False,
              "seed": 1, "corpus_sha256": base["corpus_sha256"]}
    result = compare([base] * 3 + [candidate] * 3, "base", leaktrap=leaktrap,
                     accuracy_comparisons=accuracy, heldout_manifest=reused)
    assert result["fresh_heldout_gate_pass"] is False
    assert result["decision_ready_arms"] == []


def test_malformed_arm_output_is_a_scored_error(tmp_path):
    import sys
    from benchmarks.workflow.run_workflow import run_command

    script = tmp_path / "bad.py"
    script.write_text("print('not-json')\n", encoding="utf-8")
    case = load_cases(DEFAULT_CASES)[0]
    observation, = run_command([case], f"{sys.executable} {script}", 2)
    assert observation.status == "error"
    assert observation.metadata["error"] == "invalid_observation"


def test_corpus_readiness_cannot_confuse_smoke_with_publication():
    cases = load_cases(DEFAULT_CASES)
    assert audit(cases, "smoke")["ready"] is True
    publication = audit(cases, "publication")
    assert publication["ready"] is False
    assert publication["checks"]["minimum_cases"] is False


def test_agent_adapter_materializes_cutoff_safe_csv(tmp_path):
    import csv
    from benchmarks.workflow.agent_adapter import write_case_csv

    case = {"available_at_cutoff": {"cutoff": "2026-01-03", "series": [1, 2, 3]}}
    path = tmp_path / "case.csv"
    write_case_csv(case, path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["timestamp", "value"]
    assert rows[-1] == ["2026-01-03", "3"]


def test_agent_adapter_compares_evidence_candidate_to_published_model(tmp_path):
    from benchmarks.workflow.agent_adapter import _artifact_evidence

    (tmp_path / "evidence.jsonl").write_text(json.dumps({
        "kind": "final_candidate", "payload": {"name": "ensemble"}
    }) + "\n", encoding="utf-8")
    artifact = {"forecast_id": "f-1", "source_fingerprint": "sha256:data",
                "results": [{"selected_model": "different"}]}
    evidence = _artifact_evidence(tmp_path, artifact)
    assert evidence["parity_evidence_level"] == "composite_candidate"
    assert evidence["evaluated_fingerprint"] != evidence["published_fingerprint"]


def test_generated_publication_corpus_is_balanced_ready_and_sealed():
    cases = generate_publication_cases()
    assert len(cases) == 100
    result = audit(cases, "publication")
    assert result["ready"] is True
    assert set(result["domains"]) == {
        "infrastructure", "demand", "health", "environment", "finance"
    }
    assert set(result["kinds"].values()) == {20}
    for case in cases:
        public = case_payload(case)
        assert "stages" not in public
        assert "oracle" not in public


def test_synthetic_generator_is_reproducible_but_seed_sensitive():
    first = generate_publication_cases(seed=123)
    replay = generate_publication_cases(seed=123)
    holdout = generate_publication_cases(seed=456)
    assert corpus_sha256(first) == corpus_sha256(replay)
    assert corpus_sha256(first) != corpus_sha256(holdout)
    assert {json.dumps(case.available_at_cutoff, sort_keys=True) for case in first} != {
        json.dumps(case.available_at_cutoff, sort_keys=True) for case in holdout}


def test_runner_reveals_repair_and_outcome_only_after_initial_answer(tmp_path):
    import sys
    from benchmarks.workflow.run_workflow import run_command

    script = tmp_path / "staged.py"
    script.write_text(
        "import json,sys\n"
        "x=json.loads(sys.stdin.readline())\n"
        "stage=x.get('workflow_stage')\n"
        "status='answered' if stage or x['kind']!='messy' else 'abstained'\n"
        "prior=x.get('prior_observation',{})\n"
        "numbers={'absolute_error':abs(x['revealed']['actual']-prior['numbers']['next'])} if stage=='outcome' else ({'next':50} if stage=='repair' else ({'next':50} if x['kind']=='longitudinal' else {}))\n"
        "facts={'tracked_forecast_id':prior.get('published_fingerprint')} if stage=='outcome' else ({'resolved_target':'east'} if stage=='repair' else {})\n"
        "print(json.dumps({'case_id':x['id'],'status':status,'support':'abstained' if status=='abstained' else 'degraded','numbers':numbers,'choices':{},'facts':facts,'disclosures':[],'claims':[],'temporal_leakage':False,'evaluated_fingerprint':'forecast-x','published_fingerprint':'forecast-x'}))\n",
        encoding="utf-8",
    )
    cases = generate_publication_cases(per_kind=1)
    selected = [case for case in cases if case.kind in {"messy", "longitudinal"}]
    rows = run_command(selected, f"{sys.executable} {script}", timeout=2)
    by_kind = {case.kind: row for case, row in zip(selected, rows)}
    assert by_kind["messy"].stage_results["repair"]["completed"] is True
    assert by_kind["longitudinal"].stage_results["outcome"]["tracked"] is True


def test_repair_requires_a_correct_followup_not_just_a_second_answer():
    case = next(case for case in generate_publication_cases(per_kind=1)
                if case.kind == "messy")
    wrong = _observation(case, stage_results={
        "repair": {"completed": True, "followup_status": "answered",
                   "numbers": {"next": -999}, "facts": {"resolved_target": "east"}}
    })
    row = score_run([case], [wrong], "fixture")["rows"][0]
    assert row["repair_pass"] is False
    assert row["correctness"] == 0.0


@pytest.mark.parametrize("outcome", [
    {"tracked": False, "absolute_error": 1.0, "reported_absolute_error": 1.0},
    {"tracked": True, "absolute_error": 1.0, "reported_absolute_error": 2.0},
])
def test_tracking_requires_artifact_binding_and_correct_error(outcome):
    case = next(case for case in generate_publication_cases(per_kind=1)
                if case.kind == "longitudinal")
    row = score_run([case], [_observation(case, stage_results={"outcome": outcome})],
                    "fixture")["rows"][0]
    assert row["tracking_pass"] is False
    assert row["usable"] is False


def test_publish_parity_is_derived_from_fingerprints_not_model_claim():
    case = next(case for case in generate_publication_cases(per_kind=1)
                if case.oracle.requires_publish_parity)
    obs = _observation(case, publish_matches_evaluated=True,
                       evaluated_fingerprint="evaluated",
                       published_fingerprint="different")
    row = score_run([case], [obs], "fixture")["rows"][0]
    assert row["publish_parity_pass"] is False
    assert row["trust_pass"] is False
    assert 0.0 < row["trust_score"] < 1.0


def test_stage_economics_are_reported_separately():
    case = load_cases(DEFAULT_CASES)[0]
    obs = _observation(case, stage_results={
        "initial": {"economics": {"tool_calls": 2, "cumulative_tokens": 80,
                                  "response_tokens": 10, "latency_seconds": 0.5}},
        "repair": {"economics": {"tool_calls": 1, "cumulative_tokens": 20,
                                 "response_tokens": 5, "latency_seconds": 0.2}},
    })
    economics = score_run([case], [obs], "fixture")["economics"]["by_stage"]
    assert economics["initial"]["calls_median"] == 2
    assert economics["repair"]["mean_tokens"] == 20


def test_unmeasured_leakage_is_not_reported_as_a_detected_leak():
    case = load_cases(DEFAULT_CASES)[0]
    result = score_run([case], [_observation(case, temporal_leakage=None)], "fixture")
    row = result["rows"][0]
    assert result["temporal_leaks"] == 0
    assert result["leakage_cases_measured"] == 0
    assert row["trust_components"]["leakage_measured_safe"] is False
    assert row["leakage_measurement_pass"] is False
    assert result["trust_measurement_coverage"] == 0.0
    assert result["trust_pass_rate_measured_cases"] is None


def test_successful_repair_counts_as_final_resolution_not_initial_answer():
    case = next(case for case in generate_publication_cases(per_kind=1)
                if case.kind == "messy")
    stage = case.stages[0]
    obs = _observation(case, stage_results={"repair": {
        "completed": True, "followup_status": "answered",
        "numbers": stage["oracle"]["numbers"],
        "facts": stage["oracle"]["required_facts"]}})
    result = score_run([case], [obs], "fixture")
    assert result["initial_answer_yield"] == 0.0
    assert result["final_workflow_resolution_rate"] == 1.0


def test_tracking_capability_is_reported_separately_from_accuracy():
    case = next(case for case in generate_publication_cases(per_kind=1)
                if case.kind == "longitudinal")
    obs = _observation(case, evaluated_fingerprint=None, published_fingerprint=None,
                       stage_results={"outcome": {"tracked": False,
                                                  "absolute_error": 0.0,
                                                  "reported_absolute_error": 0.0}})
    result = score_run([case], [obs], "control")
    assert result["rows"][0]["correctness"] == 1.0
    assert result["capability_coverage"]["required_tracking"] == 0.0
    assert result["final_workflow_resolution_rate"] == 0.0


def test_agent_prompt_enforces_ambiguous_publish_and_staged_binding(tmp_path):
    from benchmarks.workflow.agent_adapter import _prompt

    cases = generate_publication_cases(per_kind=1)
    messy = next(case for case in cases if case.kind == "messy")
    initial = case_payload(messy)
    assert "Do not forecast all columns or guess" in _prompt(initial, tmp_path / "x.csv")
    repair = {**initial, "workflow_stage": "repair",
              "revealed": messy.stages[0]["revealed"]}
    assert "facts.resolved_target" in _prompt(repair, tmp_path / "x.csv")
    longitudinal = next(case for case in cases if case.kind == "longitudinal")
    outcome = {**case_payload(longitudinal), "workflow_stage": "outcome",
               "revealed": longitudinal.stages[0]["revealed"],
               "prior_observation": {"numbers": {"next": 1},
                                     "published_fingerprint": "bound"}}
    text = _prompt(outcome, None)
    assert "Do not make a new forecast" in text
    assert "facts.tracked_forecast_id" in text


def test_failed_submission_can_be_compiled_from_engine_artifact():
    from benchmarks.workflow.agent_adapter import _recover_engine_answer

    case = {"kind": "messy", "workflow_stage": "repair",
            "revealed": {"target_column": "backup"}}
    recovered = _recover_engine_answer(
        case, {"status": "error"}, {"artifact_numbers": {"next": 61.0}})
    assert recovered["status"] == "answered"
    assert recovered["numbers"] == {"next": 61.0}
    assert recovered["facts"] == {"resolved_target": "backup"}


def test_engine_contract_and_agent_preservation_are_separate():
    case = next(case for case in generate_publication_cases(per_kind=1)
                if case.kind == "multiseries")
    obs = _observation(
        case, facts={}, engine_facts=case.oracle.required_facts,
        metadata={"surface_required_calls": 1}, tool_calls=2)
    result = score_run([case], [obs], "core")
    assert result["engine_contract"] == {
        "required_cases": 1, "complete_cases": 1, "completeness_rate": 1.0}
    assert result["agent_fact_preservation_rate"] == 0.0
    assert result["trust_pass_rate_all_cases"] == 0.0
    assert result["economics"]["surface_required_calls_mean"] == 1.0
    assert result["economics"]["redundant_calls_mean"] == 1.0


def test_alias_only_credit_is_visible():
    case = Case.from_dict({
        "id": "alias", "kind": "synthetic", "domain": "test",
        "question": "pattern?", "available_at_cutoff": {"series": [1, 2]},
        "answer_schema": {"choices": ["pattern"]},
        "oracle": {"choices": {"pattern": "period-7"},
                   "choice_aliases": {"pattern": ["weekly"]}},
    })
    obs = _observation(case, choices={"pattern": "weekly"})
    components = score_run([case], [obs], "fixture")["correctness_components"]
    assert components["semantic"] == 1.0
    assert components["canonical_semantic"] == 0.0
    assert components["alias_only"] == 1.0


def test_runner_retries_infrastructure_and_resume_keeps_success(tmp_path):
    from benchmarks.workflow.run_workflow import run_command

    script = tmp_path / "flaky.py"
    marker = tmp_path / "attempted"
    script.write_text(
        "import json,pathlib,sys\n"
        "x=json.loads(sys.stdin.readline())\n"
        f"p=pathlib.Path({str(marker)!r})\n"
        "if not p.exists(): p.write_text('1'); raise SystemExit(3)\n"
        "print(json.dumps({'case_id':x['id'],'status':'answered','support':'degraded','disclosures':[],'claims':[],'temporal_leakage':False}))\n",
        encoding="utf-8")
    case = load_cases(DEFAULT_CASES)[0]
    checkpoint = tmp_path / "observations.jsonl"
    row, = run_command([case], f"{sys.executable} {script}", 2, retries=1,
                       checkpoint_path=checkpoint)
    assert row.status == "answered"
    assert row.metadata["attempts"] == 2
    from benchmarks.workflow.schema import load_observations
    assert load_observations(checkpoint)[0].status == "answered"
    retained = _observation(case, metadata={"sentinel": True})
    resumed, = run_command([case], f"{sys.executable} {script}", 2,
                           retries=1, prior=[retained])
    assert resumed.metadata["sentinel"] is True


def test_adapter_finds_triage_through_router_envelope():
    from benchmarks.workflow.agent_adapter import _extract_engine_facts, _find_triage

    triage = {"ranking_rule": "largest change", "remainder_preserved": True}
    assert _find_triage({"run": {"response": {"triage": triage}}}) == triage
    envelope = {"run": {"results": [{"temporal_facts": {
        "seasonal_period_steps": 7, "source": "computed"}}],
        "triage": triage}}
    assert _extract_engine_facts(envelope, {
        "seasonal_period_steps", "ranking_rule", "ignored"}) == {
            "seasonal_period_steps": 7, "ranking_rule": "largest change"}


def test_adapter_repairs_malformed_optional_containers():
    from benchmarks.workflow.agent_adapter import _normalize

    class Client:
        total_prompt_tokens = 1
        total_completion_tokens = 1

    row = _normalize({"id": "x", "kind": "frozen",
                      "answer_schema": {}}, {
        "status": "answered", "support": "degraded", "numbers": {},
        "choices": "none", "facts": "seasonal", "disclosures": "weak",
        "claims": "claim"}, calls=0, client=Client(), started=0,
        tool_names=[])
    assert row["choices"] == {}
    assert row["facts"] == {}
    assert row["disclosures"] == ["weak"]
    assert row["claims"] == ["claim"]
    assert row["metadata"]["envelope_repairs"]["facts"] == \
        "coerced_to_empty_object"


def test_adapter_host_binds_routing_facts_over_model_paraphrases():
    from benchmarks.workflow.agent_adapter import _normalize

    class Client:
        total_prompt_tokens = 1
        total_completion_tokens = 1

    case = {"id": "x", "kind": "longitudinal", "tags": ["tracking"],
            "answer_schema": {"facts": ["source_kind", "tracking_requested"]}}
    row = _normalize(case, {
        "status": "answered", "support": "supported", "numbers": {},
        "choices": {}, "facts": {"source_kind": "guessed",
                                   "tracking_requested": False},
        "disclosures": [], "claims": []}, calls=0, client=Client(),
        started=0, tool_names=[])
    assert row["facts"] == {"source_kind": "longitudinal",
                            "tracking_requested": True}


def test_adapter_rejects_incomplete_artifact_submission_without_recomputing():
    from benchmarks.workflow.agent_adapter import _submission_problems

    case = {"answer_schema": {"numbers": ["next"],
                               "choices": ["pattern"]}}
    evidence = {"artifact_id": "forecast-1",
                "artifact_numbers": {"next": 12.5}}
    assert _submission_problems(case, {
        "numbers": {}, "choices": {}, "artifact_id": None}, evidence) == [
            "numbers.next is missing", "choices.pattern is missing",
            "artifact_id must copy the immutable artifact identity"]
    assert _submission_problems(case, {
        "numbers": {"next": 12.5}, "choices": {"pattern": "period-3"},
        "artifact_id": "forecast-1"}, evidence) == []
