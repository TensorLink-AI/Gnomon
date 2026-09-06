import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.workflow.run_workflow import DEFAULT_CASES, case_payload
from benchmarks.workflow.schema import Case, Observation, load_cases
from benchmarks.workflow.scoring import score_run
from benchmarks.workflow.provenance import corpus_sha256


def _observation(case, **overrides):
    values = {
        "case_id": case.id, "status": "answered", "support": "supported",
        "numbers": case.oracle.numbers, "choices": case.oracle.choices,
        "disclosures": list(case.oracle.required_disclosures),
        "temporal_leakage": False,
        "facts": case.oracle.required_facts,
        "tool_calls": 1, "cumulative_tokens": 100,
        "metadata": {"leakage_measurement": "cutoff_projection_v1",
                     "cutoff_projection_sha256": "a" * 64},
    }
    if case.oracle.should_abstain:
        values.update(status="abstained", support="abstained")
    elif values["support"] not in case.oracle.allowed_support:
        values["support"] = case.oracle.allowed_support[0]
    values.update(overrides)
    return Observation.from_dict(values)


def test_current_single_phase_cases_hide_the_oracle():
    cases = [case for case in load_cases(DEFAULT_CASES) if not case.episode]
    assert {case.kind for case in cases} == {"synthetic", "frozen"}
    assert all("oracle" not in case_payload(case) for case in cases)


@pytest.mark.parametrize("field", ["stages"])
def test_retired_case_fields_are_rejected(field):
    from dataclasses import asdict
    value = asdict(load_cases(DEFAULT_CASES)[0])
    value[field] = []
    with pytest.raises(ValueError, match="unknown"):
        Case.from_dict(value)


@pytest.mark.parametrize("field", [
    "requires_repair", "requires_tracking", "requires_publish_parity",
    "requires_quote_match", "engine_required_facts",
])
def test_retired_oracle_fields_are_rejected(field):
    from dataclasses import asdict
    value = asdict(load_cases(DEFAULT_CASES)[0])
    value["oracle"][field] = False
    with pytest.raises(ValueError, match="unknown"):
        Case.from_dict(value)


@pytest.mark.parametrize("field", [
    "stage_results", "engine_facts", "published_fingerprint", "repair_completed",
    "publish_matches_evaluated", "tracking_completed", "quote_matches",
    "evaluated_fingerprint", "headline_numbers", "artifact_numbers",
])
def test_retired_observation_fields_are_rejected(field):
    with pytest.raises(ValueError, match="unknown"):
        Observation.from_dict({"case_id": "retired", field: None})


def test_old_case_schema_is_rejected_and_retired_runners_are_absent():
    from dataclasses import asdict
    value = asdict(load_cases(DEFAULT_CASES)[0])
    assert value["schema_version"] == 2
    value["schema_version"] = 1
    with pytest.raises(ValueError, match="schema"):
        Case.from_dict(value)
    root = Path(__file__).parents[1]
    for relative in ("workflow/compare.py", "workflow/audit.py", "workflow/generate.py",
                     "workflow/cases/smoke.jsonl", "common/envfile.py"):
        assert not (root / relative).exists()


def test_installed_agent_skill_is_compact_without_hiding_safety_contracts():
    path = Path(__file__).parents[2] / "skills" / "use-gnomon" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) <= 5_000
    assert not (path.parent / "references" / "legacy-workflows.md").exists()
    for required in ("provider", "statistic", "data_ref", "source-availability", "local-recording"):
        assert required in text
    assert "permission" in text and "ledger" in text


def test_perfect_single_phase_run_passes_measurement_gates():
    cases = [case for case in load_cases(DEFAULT_CASES) if not case.episode]
    result = score_run(cases, [_observation(case) for case in cases], "fixture")
    assert result["correctness_mean_all_cases"] == 1.0
    assert result["trust_pass_rate_all_cases"] == 1.0
    assert result["usability_pass_rate_all_cases"] == 1.0
    assert result["leakage_safety_gate_pass"] is True and result["completeness_gate_pass"] is True
    assert result["economics"]["calls_median"] == 1


def test_leak_is_a_hard_gate_and_abstention_cannot_inflate_accuracy():
    cases = [case for case in load_cases(DEFAULT_CASES) if not case.episode]
    observations = [_observation(case) for case in cases]
    observations[0] = _observation(cases[0], temporal_leakage=True)
    observations[1] = _observation(cases[1], status="abstained", support="abstained")
    result = score_run(cases, observations, "bad")
    assert result["temporal_leaks"] == 1
    assert result["leakage_safety_gate_pass"] is False
    assert result["correctness_mean_all_cases"] < 1.0


def test_missing_cases_are_in_all_case_denominator():
    cases = [case for case in load_cases(DEFAULT_CASES) if not case.episode]
    result = score_run(cases, [_observation(cases[0])], "partial")
    assert len(result["missing"]) == len(cases) - 1
    assert result["correctness_mean_all_cases"] == pytest.approx(1 / len(cases))
    assert result["completeness_gate_pass"] is False
    assert result["leakage_safety_gate_pass"] is False


def test_correctness_and_trust_components_are_reported_separately():
    cases = [case for case in load_cases(DEFAULT_CASES) if not case.episode]
    observations = [_observation(case) for case in cases]
    observations[5] = _observation(cases[5], choices={"action": "wrong"})
    result = score_run(cases, observations, "component-report")
    assert result["correctness_components"]["numeric"] == 1.0
    assert result["correctness_components"]["semantic"] < 1.0
    assert set(result["trust_component_pass_rates"]) == {
        "leakage_measured_safe", "disclosures", "forbidden_claims",
        "support",
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
    cases = [case for case in load_cases(DEFAULT_CASES) if not case.episode]
    assert corpus_sha256(cases) == corpus_sha256([case for case in load_cases(DEFAULT_CASES) if not case.episode])
    changed = list(cases)
    source = json.loads(DEFAULT_CASES.read_text().splitlines()[0])
    source["oracle"]["numbers"]["h1"] += 11
    changed[0] = Case.from_dict(source)
    assert corpus_sha256(changed) != corpus_sha256(cases)


def test_malformed_arm_output_is_a_scored_error(tmp_path):
    import sys
    from benchmarks.workflow.run_workflow import run_command

    script = tmp_path / "bad.py"
    script.write_text("print('not-json')\n", encoding="utf-8")
    case = load_cases(DEFAULT_CASES)[0]
    observation, = run_command([case], f"{sys.executable} {script}", 2)
    assert observation.status == "error"
    assert observation.metadata["error"] == "invalid_observation"


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
    from dataclasses import replace
    retained = replace(row, metadata={**row.metadata, "sentinel": True})
    resumed, = run_command([case], f"{sys.executable} {script}", 2,
                           retries=1, prior=[retained])
    assert resumed.metadata["sentinel"] is True


def test_workflow_resume_identity_rejects_changed_arm(tmp_path):
    import argparse
    from benchmarks.workflow.run_workflow import (
        _prepare_run_identity, _run_identity,
    )

    cases = [case for case in load_cases(DEFAULT_CASES) if not case.episode]
    args = argparse.Namespace(
        submission=None, arm="evidence", arm_command="python arm.py",
        timeout=120.0, jobs=1, infrastructure_retries=2)
    identity = _run_identity(args, cases)
    _prepare_run_identity(tmp_path, identity, resume=False)
    (tmp_path / "observations.jsonl").write_text("{}\n")
    changed = {**identity, "arm_command": "python other.py"}
    with pytest.raises(SystemExit, match="resume identity mismatch"):
        _prepare_run_identity(tmp_path, changed, resume=True)


def test_workflow_resume_refuses_legacy_checkpoint_without_identity(tmp_path):
    from benchmarks.workflow.run_workflow import _prepare_run_identity

    (tmp_path / "observations.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="without run_identity"):
        _prepare_run_identity(tmp_path, {"schema_version": 1}, resume=True)
