from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.contextbench.generate import generate, main as generate_main
from benchmarks.contextbench.generate_stress import generate as generate_stress
from benchmarks.contextbench import run_surfaces as surface_runner
from benchmarks.contextbench.run_contextbench import (
    run_case, smape, summarize, valid_disposition,
)
from benchmarks.contextbench.run_contextbench import main as run_main
from benchmarks.contextbench.run_llm import (
    _bounded_context_tool, _prepare_run_identity, compile_events, raw_case,
    safe_payload,
)
from benchmarks.contextbench.run_surfaces import surface_row
from benchmarks.contextbench.report_surfaces import aggregate
from benchmarks.contextbench.report_contextbench import (
    aggregate as aggregate_contextbench,
)
from benchmarks.contextbench.report_llm import compare as compare_llm
from benchmarks.contextbench.schema import Case, load_cases, load_oracles
from gnomon.context_model import rolling_residuals
from gnomon.workflows import normalise_context_response_containers
from gnomon.workflows import extract_explicit_schedule_context, DocumentRef


def test_generator_is_reproducible_seed_sensitive_and_balanced():
    first_cases, first_oracles = generate(41, per_family=2)
    replay_cases, replay_oracles = generate(41, per_family=2)
    changed_cases, _ = generate(42, per_family=2)
    assert first_cases == replay_cases
    assert first_oracles == replay_oracles
    assert first_cases != changed_cases
    assert {row["family"] for row in first_cases} == {
        "irrelevant", "future_covariate", "repeated_event", "prior_only"}
    assert all("actual" not in row and "counterfactual" not in row
               and "effect_magnitude" not in row for row in first_cases)


def test_naturalistic_corpus_requires_semantic_compilation_without_truth_leak():
    cases, oracles = generate(43, per_family=1,
                              narrative_style="naturalistic")
    event_case = next(row for row in cases if row["family"] == "prior_only")
    parsed = extract_explicit_schedule_context([
        DocumentRef("context.txt", event_case["narrative"],
                    source_type="narrative_assertion",
                    known_at=event_case["context_events"][0]["known_at"])
    ])

    assert parsed["events"] == []
    assert len(parsed["residual_lines"]) >= 1
    assert "narrative:naturalistic" in event_case["tags"]
    assert "effect_magnitude" not in json.dumps(event_case)
    assert "actual" not in json.dumps(event_case)
    assert len(oracles) == 4


def test_family_truth_and_cutoff_contracts_are_explicit():
    cases, oracles = generate(7, per_family=1)
    truth = {row["case_id"]: row for row in oracles}
    for raw in cases:
        case = Case.from_dict(raw)
        oracle = truth[case.case_id]
        assert len(case.history) >= 8 * case.horizon
        if case.family in {"irrelevant", "prior_only"}:
            assert oracle["should_influence"] is False
            assert oracle["actual"] == oracle["counterfactual"]
        else:
            assert oracle["should_influence"] is True
            assert oracle["actual"] != oracle["counterfactual"]
    invalid = {**cases[0], "future": [1, 2, 3]}
    with pytest.raises(ValueError, match="unknown fields"):
        Case.from_dict(invalid)


def test_generated_files_are_strict_and_hash_addressed(tmp_path, monkeypatch):
    output = tmp_path / "corpus"
    monkeypatch.setattr("sys.argv", [
        "generate", "--output-dir", str(output), "--seed", "5",
        "--per-family", "1",
    ])
    assert generate_main() == 0
    assert len(load_cases(output / "cases.jsonl")) == 4
    assert len(load_oracles(output / "oracle.jsonl")) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["generator"] == "contextbench-synthetic-v2"
    assert manifest["fresh_seed"] is False
    assert len(manifest["cases_sha256"]) == 64
    assert len(manifest["oracle_sha256"]) == 64


def test_generated_manifest_records_narrative_treatment(tmp_path, monkeypatch):
    output = tmp_path / "naturalistic"
    monkeypatch.setattr("sys.argv", [
        "generate", "--output-dir", str(output), "--seed", "6",
        "--per-family", "1", "--narrative-style", "naturalistic",
    ])
    assert generate_main() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["narrative_style"] == "naturalistic"


def test_documented_engine_runner_executes_from_clean_checkout(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "run"
    # Reuse the generator CLI for its hash-addressed manifest rather than
    # synthesising benchmark metadata in the smoke test.
    subprocess.run(
        [sys.executable, "-m", "benchmarks.contextbench.generate",
         "--output-dir", str(corpus), "--seed", "73", "--per-family", "1"],
        cwd=Path(__file__).resolve().parents[2], check=True,
        capture_output=True, text=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.contextbench.run_contextbench",
         "--corpus-dir", str(corpus), "--output-dir", str(output),
         "--limit", "1", "--allow-gate-failure"],
        cwd=Path(__file__).resolve().parents[2], check=False,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "summary.json").is_file()


def test_stress_generator_is_reproducible_and_covers_production_strata():
    cases, oracles = generate_stress(71, per_stratum=1)
    replay, replay_oracles = generate_stress(71, per_stratum=1)
    assert cases == replay and oracles == replay_oracles
    dimensions = [row["dimensions"] for row in oracles]
    assert {row["stratum"] for row in dimensions} == {
        "snr_sweep", "misleading_direction", "timing_uncertainty",
        "numeric_claim", "structural_claim", "confounded", "bitemporal",
        "entity_scope",
    }
    assert {row["snr"] for row in dimensions
            if row["stratum"] == "snr_sweep"} == {0.5, 1.0, 2.0, 4.0, 8.0}
    assert {row["frequency"] for row in dimensions} == {"15min", "h", "D"}
    for oracle in oracles:
        duration = oracle.get("duration_steps")
        if duration is None or oracle["dimensions"]["stratum"] in {
            "numeric_claim", "structural_claim",
        }:
            continue
        frequency = oracle["dimensions"]["frequency"]
        assert duration <= {"15min": 8, "h": 6, "D": 2}[frequency]
    claims = [row for row in dimensions
              if row["admission_warrant"] == "asserted"]
    assert {row["claim_truth"] for row in claims} == {"true", "false"}
    mixed = [row for row in cases if row["context_events"] and row["covariates"]]
    assert len(mixed) == 1 and mixed[0]["family"] == "confounded"


def test_stress_scope_narrative_and_oracle_preserve_named_entity():
    cases, oracles = generate_stress(8182, per_stratum=1)
    case = next(row for row in cases if row["family"] == "entity_scope")
    oracle = next(row for row in oracles
                  if row["case_id"] == case["case_id"])

    assert "affects other-series from" in case["narrative"]
    assert "affects the value series" not in case["narrative"]
    assert oracle["expected_disposition"] == "not_considered"


def test_stress_summary_separates_empirical_admission_from_asserted_truth():
    rows = []
    for case_id, warrant, truth, applied, changed in (
        ("empirical", "empirical", True, True, True),
        ("false-claim", "asserted", False, True, True),
    ):
        rows.append({
            "case_id": case_id, "family": (
                "repeated_event" if warrant == "empirical" else "numeric_claim"),
            "oracle_dimensions": {"stratum": "snr_sweep", "snr": 4.0,
                                  "admission_warrant": warrant},
            "history_smape": 2.0, "context_smape": 1.0,
            "incremental_smape": 1.0, "should_influence": truth,
            "applied": applied, "primary_changed": changed,
            "temporal_leakage": False, "publication_parity": True,
            "interval_coverage": 0.8, "effect_direction_correct": True,
            "effect_magnitude_inferred": 1.0,
            "effect_magnitude_expected": 1.0, "onset_step_inferred": 0,
            "onset_step_expected": 0, "disposition": "applied",
            "disposition_valid": True,
        })
    summary = summarize(rows, {"generator": "contextbench-stress-v1",
                               "seed": 1, "fresh_seed": True, "cases": 2})
    assert summary["metrics"]["admission_precision"] == 1.0
    assert summary["metrics"]["false_influence_rate"] == 0.0
    assert summary["metrics"]["false_asserted_claim_primary_change_rate"] == 1.0
    assert "frequency" in summary["dimensions"]


def test_smape_is_symmetric_and_zero_safe():
    assert smape([0.0, 10.0], [0.0, 12.0]) == smape([0.0, 12.0], [0.0, 10.0])
    assert smape([0.0], [0.0]) == 0.0


def test_context_residuals_respect_ets_minimum_history():
    residuals = rolling_residuals([1.0, 2.0, 3.0, 4.0, 5.0], "ets", 1)
    assert residuals[:4] == [None, None, None, None]
    assert residuals[4] is not None


def test_asserted_context_cannot_change_primary_under_default_policy(tmp_path):
    raw_cases, raw_oracles = generate_stress(73, per_stratum=1)
    raw = next(row for row in raw_cases if row["family"] == "numeric_claim")
    case = Case.from_dict(raw)
    oracle = load_oracles_from_rows(raw_oracles)[case.case_id]
    row = run_case(case, oracle, tmp_path)
    assert row["default_policy_primary_changed"] is False


def test_disposition_contract_does_not_demand_oracle_omniscience():
    assert valid_disposition("repeated_event", True, "applied")
    assert valid_disposition("repeated_event", False, "scenario_only")
    assert valid_disposition("future_covariate", False, "rejected")
    assert valid_disposition("prior_only", False, "scenario_only")
    assert not valid_disposition("future_covariate", False, "applied")


def test_summary_does_not_let_covariate_lift_hide_event_failure():
    rows = []
    for family in ("irrelevant", "prior_only", "future_covariate", "repeated_event"):
        influence = family in {"future_covariate", "repeated_event"}
        applied = family == "future_covariate"
        rows.append({
            "family": family, "history_smape": 10.0,
            "context_smape": 1.0 if applied else 10.0,
            "incremental_smape": 9.0 if applied else 0.0,
            "should_influence": influence, "applied": applied,
            "primary_changed": applied, "temporal_leakage": False,
            "publication_parity": True, "interval_coverage": 0.8,
            "effect_direction_correct": True if applied else None,
            "disposition": "applied" if applied else "scenario_only",
            "expected_disposition": "applied" if influence else "scenario_only",
        })
    summary = summarize(rows, {
        "generator": "x", "seed": 1, "fresh_seed": True, "cases": 4})
    assert summary["metrics"]["incremental_smape_influence_cases"] > 0
    assert summary["gates"]["repeated_event_smape_improves"] is False
    assert summary["gates"]["admission_recall_at_least_80pct"] is False
    assert summary["decision_ready"] is False


def test_end_to_end_smoke_runs_all_families_and_writes_scores(tmp_path,
                                                               monkeypatch):
    corpus, output = tmp_path / "corpus", tmp_path / "run"
    monkeypatch.setattr("sys.argv", [
        "generate", "--output-dir", str(corpus), "--seed", "13",
        "--per-family", "1",
    ])
    assert generate_main() == 0
    monkeypatch.setattr("sys.argv", [
        "run", "--corpus-dir", str(corpus), "--output-dir", str(output),
        "--allow-gate-failure",
    ])
    assert run_main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["cases"] == 4
    assert set(summary["families"]) == {
        "irrelevant", "future_covariate", "repeated_event", "prior_only"}
    observations = [json.loads(line) for line in
                    (output / "observations.jsonl").read_text().splitlines()]
    assert len(observations) == 4
    assert all("history_forecast" in row and "context_forecast" in row
               for row in observations)
    assert not any(row["temporal_leakage"] for row in observations)


class _ScriptedClient:
    model = "scripted"
    total_prompt_tokens = 0
    total_completion_tokens = 0

    def __init__(self, arguments):
        self.arguments = arguments
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        call = SimpleNamespace(function=SimpleNamespace(
            name=(kwargs["tools"][0]["function"]["name"]),
            arguments=json.dumps(self.arguments)))
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(tool_calls=[call]))])


def test_raw_llm_payload_cannot_contain_sealed_oracle():
    raw_cases, raw_oracles = generate(3, per_family=1)
    case = Case.from_dict(raw_cases[0])
    payload = safe_payload(case)
    rendered = json.dumps(payload)
    oracle = raw_oracles[0]
    assert "actual" not in payload and "counterfactual" not in payload
    assert "family" not in payload
    assert "case_id" not in payload
    assert json.dumps(oracle["actual"]) not in rendered
    assert json.dumps(oracle["counterfactual"]) not in rendered


def test_surface_row_is_oracle_sealed_and_family_neutral():
    raw_cases, raw_oracles = generate(31, per_family=1)
    case = Case.from_dict(next(
        row for row in raw_cases if row["family"] == "future_covariate"))
    row = surface_row(case, include_context=True)
    rendered = json.dumps(row)
    oracle = next(item for item in raw_oracles
                  if item["case_id"] == case.case_id)
    assert case.case_id not in rendered
    assert "family" not in row
    assert json.dumps(oracle["actual"]) not in rendered
    assert json.dumps(oracle["counterfactual"]) not in rendered
    assert row["tier"] == "T4"
    assert row["_require_gnomon_execution"] is True
    assert row["_host_compiled_forecast"] is True
    assert row["_require_context_explanation"] is True
    assert row["_time_origin"] == "2025-01-01T00:00:00+00:00"
    assert row["input"]["future_covariates"]
    assert row["input"]["covariate_mapping"][0]["type"] == "binary"
    assert len(row["input"]["history"]["value"]) == len(case.history)
    assert "Input (JSON)" not in row["prompt"]
    assert "dataset is bound by the host" in row["prompt"]
    assert "Preserve any interval limitation" in row["prompt"]


def test_surface_summary_reports_agent_context_explanation_contract():
    row = {
        "case_id": "case-1", "family": "repeated_event",
        "status": "answered", "history_smape": 2.0,
        "context_smape": 2.0, "incremental_smape": 0.0,
        "should_influence": True, "primary_changed": False,
        "applied": False, "oracle_dimensions": {},
        "disposition_valid": True, "temporal_leakage": False,
        "publication_parity": True, "history_calls": 0,
        "context_calls": 1, "surface_required_calls": 1,
        "context_explanation_contract": {
            "complete": True, "primary_preserved": True,
            "scenario_represented": True,
            "interval_limit_preserved": True,
                "automation_limit_preserved": True,
                "rejection_evidence_cited": True,
                "scenario_consequence_preserved": True,
        },
    }

    summary = surface_runner.summarize(
        [row], "evidence", {"seed": 1, "fresh_seed": True}, "compiled")

    assert summary["metrics"]["context_explanation_contract"] == {
        "complete": 1.0, "primary_preserved": 1.0,
        "scenario_represented": 1.0,
        "interval_limit_preserved": 1.0,
            "automation_limit_preserved": 1.0,
            "rejection_evidence_cited": 1.0,
            "scenario_consequence_preserved": 1.0,
        }
    assert summary["metrics"]["context_effect_accounting"] == {
        "answered_cases": 1,
        "admitted_cases": 0,
        "numerically_changed_cases": 0,
        "admitted_without_numeric_change": 0,
        "beneficial_changes": 0,
        "harmful_changes": 0,
        "neutral_changes": 0,
        "numeric_change_rate": 0.0,
        "mean_uplift_when_changed": None,
    }


def test_surface_summary_separates_admission_change_and_uplift() -> None:
    base = {
        "family": "repeated_event", "status": "answered",
        "history_smape": 3.0, "should_influence": True,
        "oracle_dimensions": {}, "disposition_valid": True,
        "temporal_leakage": False, "publication_parity": True,
        "history_calls": 0, "context_calls": 1,
        "surface_required_calls": 1,
    }
    rows = [
        {**base, "case_id": "admitted-unchanged", "context_smape": 3.0,
         "incremental_smape": 0.0, "applied": True,
         "primary_changed": False},
        {**base, "case_id": "helpful", "context_smape": 2.0,
         "incremental_smape": 1.0, "applied": True,
         "primary_changed": True},
        {**base, "case_id": "harmful", "context_smape": 4.0,
         "incremental_smape": -1.0, "applied": True,
         "primary_changed": True},
    ]

    summary = surface_runner.summarize(
        rows, "evidence", {"seed": 1, "fresh_seed": True}, "compiled")

    assert summary["metrics"]["context_effect_accounting"] == {
        "answered_cases": 3,
        "admitted_cases": 3,
        "numerically_changed_cases": 2,
        "admitted_without_numeric_change": 1,
        "beneficial_changes": 1,
        "harmful_changes": 1,
        "neutral_changes": 0,
        "numeric_change_rate": 2 / 3,
        "mean_uplift_when_changed": 0.0,
    }


def test_automation_limit_needs_typed_parity_and_explicit_ineligibility():
    check = surface_runner.preserves_automation_limit
    matched = {"engine": False, "supplied": False, "matched": True}

    assert check("Automation eligibility is false.", restricted=True,
                 projection=matched)
    assert check("automation_eligible=false", restricted=True,
                 projection=matched)
    assert check("This scenario cannot authorize automation.", restricted=True,
                 projection=matched)
    assert not check("Automation was not requested.", restricted=True,
                     projection=matched)
    assert not check("Automation is not eligible.", restricted=True,
                     projection={**matched, "matched": False})
    assert check("", restricted=False, projection={})


def test_history_surface_row_excludes_all_outside_context():
    raw_cases, _ = generate(32, per_family=1)
    case = Case.from_dict(next(
        row for row in raw_cases if row["family"] == "future_covariate"))
    row = surface_row(case, include_context=False)
    assert row["tier"] == "T2"
    assert row["input"] == {"history": {"value": list(case.history)}}
    assert case.narrative not in row["prompt"]


def test_surface_row_preserves_non_hourly_grid():
    raw_cases, _ = generate_stress(72, per_stratum=1)
    daily = Case.from_dict(next(row for row in raw_cases
                                if row["frequency"] == "D"))
    row = surface_row(daily, include_context=True)
    assert row["_frequency"] == "D"
    assert row["_time_step_seconds"] == 86400.0


def test_unrouted_surface_policy_preserves_execution_without_forced_routing():
    raw_cases, _ = generate(33, per_family=1)
    row = surface_row(Case.from_dict(raw_cases[0]), include_context=True,
                      routing_policy="unrouted")
    assert row["_require_gnomon_execution"] is True
    assert row["_host_compiled_forecast"] is False
    assert "Input (JSON)" in row["prompt"]


def test_surface_preflight_rejects_context_outside_the_series_grid():
    raw_cases, _ = generate(34, per_family=1)
    raw = next(row for row in raw_cases if row["family"] == "prior_only")
    raw["context_events"][0]["effective_start"] = "2030-01-01T00:00:00+00:00"
    raw["context_events"][0]["effective_end"] = "2030-01-01T01:00:00+00:00"
    with pytest.raises(ValueError, match="does not overlap the benchmark grid"):
        surface_row(Case.from_dict(raw), include_context=True)


def test_scripted_raw_arm_scores_only_after_model_submission():
    raw_cases, raw_oracles = generate(4, per_family=1)
    case = Case.from_dict(raw_cases[0])
    oracle = load_oracles_from_rows(raw_oracles)[case.case_id]
    client = _ScriptedClient({
        "forecast": list(oracle.actual), "context_used": False})
    row = raw_case(case, oracle, client)
    assert row["status"] == "answered"
    assert row["history_smape"] == 0.0
    assert row["context_smape"] == 0.0
    prompt = json.dumps(client.messages)
    assert "counterfactual" not in prompt and '"actual"' not in prompt
    assert case.case_id not in prompt and '"family"' not in prompt


def test_raw_arm_counts_wrong_length_submission_as_product_failure():
    raw_cases, raw_oracles = generate(5, per_family=1)
    case = Case.from_dict(raw_cases[0])
    oracle = load_oracles_from_rows(raw_oracles)[case.case_id]
    row = raw_case(case, oracle, _ScriptedClient({
        "forecast": [1.0], "context_used": False}))
    assert row["status"] == "product_failure"
    assert row["failure_class"] == "agent_non_submission"
    assert row["failure_stage"] == "raw_forecast_submission"
    assert row["usage_accounting_version"] == 2


def load_oracles_from_rows(rows):
    from benchmarks.contextbench.schema import Oracle
    return {row["case_id"]: Oracle.from_dict(row) for row in rows}


def test_scripted_compiler_is_quote_grounded_and_magnitude_free():
    raw_cases, _ = generate(8, per_family=1)
    repeated = Case.from_dict(next(
        row for row in raw_cases if row["family"] == "prior_only"))
    source = repeated.context_events[0]
    quote = (
        f"{source['event_type']} affects the value series from "
        f"{source['effective_start']} through {source['effective_end']}."
    )
    client = _ScriptedClient({"events": [{
        "document_index": 0, "event_type": source["event_type"],
        "entity_scope": ["*"],
        "effective_start": source["effective_start"],
        "effective_end": source["effective_end"],
        "known_at": "2099-01-01T00:00:00+00:00",
        "evidence_quote": quote, "effect_family": "temporary_pulse",
        "direction": "unknown", "duration": "temporary",
        "attributes": {"magnitude": 999999},
    }]})
    compiled = compile_events(repeated, client)
    assert len(compiled["events"]) == 1
    attributes = compiled["events"][0]["attributes"]
    assert compiled["events"][0]["known_at"] == source["known_at"]
    assert "magnitude" not in attributes
    assert attributes["evidence_quote"] == quote


def test_semantic_compiler_binds_public_target_before_runtime_alias():
    raw_cases, _ = generate(44, per_family=1,
                            narrative_style="naturalistic")
    case = Case.from_dict(next(
        row for row in raw_cases if row["family"] == "prior_only"))
    source = case.context_events[0]
    quote = next(line for line in case.narrative.splitlines()
                 if source["effective_start"] in line)
    client = _ScriptedClient({"events": [{
        "document_index": 0, "event_type": source["event_type"],
        "entity_scope": ["value"],
        "effective_start": source["effective_start"],
        "effective_end": source["effective_end"],
        "known_at": source["known_at"], "evidence_quote": quote,
    }]})

    compiled = compile_events(case, client)

    assert compiled["compiler_called"] is True
    assert compiled["events"][0]["entity_scope"] == ["*"]
    assert "from: value" in client.messages[0][0]["content"]


def test_context_llm_resume_identity_fails_closed(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    identity = {"schema_version": 1, "condition": "compiled-context"}
    _prepare_run_identity(output, identity, resume=False)
    (output / "observations.jsonl").write_text("{}\n")

    _prepare_run_identity(output, identity, resume=True)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _prepare_run_identity(
            output, {**identity, "condition": "raw-llm"}, resume=True)


def test_llm_report_requires_matched_corpus_and_preserves_pairs(tmp_path):
    raw, compiled = tmp_path / "raw", tmp_path / "compiled"
    raw.mkdir(); compiled.mkdir()
    raw_summary = {
        "condition": "raw-llm", "corpus_manifest_sha256": "same",
        "reasoning_effort": "none", "narrative_style": "naturalistic",
        "llm_usage": {"model": "test"},
        "llm_usage_observations": {"requests": 6},
    }
    compiled_summary = {
        "condition": "compiled-context", "corpus_manifest_sha256": "same",
        "compiler_calls": 2, "compiler_event_precision": 1.0,
        "compiler_event_recall": 1.0, "compiler_false_events": 0,
        "llm_usage_observations": {"requests": 2},
    }
    (raw / "summary.json").write_text(json.dumps(raw_summary))
    (compiled / "summary.json").write_text(json.dumps(compiled_summary))
    raw_rows = [
        {"case_id": "a", "family": "repeated_event", "status": "answered",
         "context_smape": 3.0, "history_smape": 2.0},
        {"case_id": "b", "family": "irrelevant", "status": "answered",
         "context_smape": 2.0, "history_smape": 1.0},
    ]
    compiled_rows = [
        {"case_id": "a", "family": "repeated_event", "status": "answered",
         "context_smape": 1.0, "history_smape": 2.0},
        {"case_id": "b", "family": "irrelevant", "status": "answered",
         "context_smape": 1.0, "history_smape": 1.0},
    ]
    for directory, rows in ((raw, raw_rows), (compiled, compiled_rows)):
        (directory / "observations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))

    report = compare_llm(raw, compiled)

    assert report["overall"]["mean_raw_minus_compiled_smape"] == 1.5
    assert report["overall"]["compiled_wins"] == 2
    assert report["families"]["irrelevant"][
        "compiled_context_harm_rate"] == 0.0
    assert len(report["paired_rows"]) == 2
    compiled_summary["corpus_manifest_sha256"] = "different"
    (compiled / "summary.json").write_text(json.dumps(compiled_summary))
    with pytest.raises(ValueError, match="same corpus"):
        compare_llm(raw, compiled)


def test_non_schedule_structural_narrative_reaches_semantic_compiler():
    raw_cases, _ = generate_stress(8183, per_stratum=1)
    structural = Case.from_dict(next(
        row for row in raw_cases
        if row["case_id"] == "stress-structural-true-0000"))
    source = structural.context_events[0]
    client = _ScriptedClient({"events": [{
        "document_index": 0,
        "event_type": "structural_break",
        "entity_scope": ["*"],
        "effective_start": source["effective_start"],
        "effective_end": None,
        "known_at": source["known_at"],
        "evidence_quote": structural.narrative,
        "effect_family": "regime_change",
        "direction": "unknown",
        "duration": "persistent",
    }]})

    compiled = compile_events(structural, client)

    assert compiled["compiler_called"] is True
    assert compiled["compiler_calls"] == 1
    assert len(compiled["events"]) == 1
    assert compiled["events"][0]["event_type"] == \
        "structural:trend_ceases"
    assert compiled["events"][0]["effective_end"] > \
        compiled["events"][0]["effective_start"]
    normalizations = compiled["events"][0]["attributes"][
        "compiler_normalizations"]
    assert {item["field"] for item in normalizations} >= {
        "event_type", "effective_end"}


def test_schedule_compiler_schema_is_bounded_and_source_grounded_only():
    from gnomon.workflows import CONTEXT_RESPONSE_SCHEMA

    tool = _bounded_context_tool(
        {"response_schema": CONTEXT_RESPONSE_SCHEMA}, 4)
    schema = tool["function"]["parameters"]
    events = schema["properties"]["events"]
    assert events["maxItems"] == 4
    assert set(events["items"]["properties"]) == set(
        events["items"]["required"])
    assert "hypotheses" not in schema["properties"]
    # The reusable product schema must not be mutated by an adapter.
    assert "effect_family" in CONTEXT_RESPONSE_SCHEMA[
        "properties"]["events"]["items"]["properties"]


def test_compiler_repairs_json_encoded_schema_containers():
    raw_cases, _ = generate(9, per_family=1)
    case = Case.from_dict(next(
        row for row in raw_cases if row["family"] == "prior_only"))
    source = case.context_events[0]
    quote = (
        f"{source['event_type']} affects the value series from "
        f"{source['effective_start']} through {source['effective_end']}."
    )
    encoded = json.dumps([{
        "document_index": 0, "event_type": source["event_type"],
        "entity_scope": ["value"], "effective_start": source["effective_start"],
        "effective_end": source["effective_end"],
        "known_at": "2025-01-01T00:00:00+00:00", "evidence_quote": quote,
    }])
    repaired, repairs = normalise_context_response_containers({
        "events": encoded, "hypotheses": "[]"})
    assert len(repaired["events"]) == 1
    assert repaired["events"][0]["entity_scope"] == ["value"]
    assert repairs == ["events", "hypotheses"]


def test_compiler_repairs_provider_trailing_fields_inside_events_string():
    raw_cases, _ = generate(10, per_family=1)
    case = Case.from_dict(next(
        row for row in raw_cases if row["family"] == "prior_only"))
    source = case.context_events[0]
    quote = (
        f"{source['event_type']} affects the value series from "
        f"{source['effective_start']} through {source['effective_end']}."
    )
    event = {
        "document_index": 0, "event_type": source["event_type"],
        "entity_scope": ["*"], "effective_start": source["effective_start"],
        "effective_end": source["effective_end"],
        "known_at": "2025-01-01T00:00:00+00:00", "evidence_quote": quote,
    }
    for suffix in (', "hypotheses": []', ', "hypotheses": []}'):
        malformed = json.dumps([event]) + suffix
        repaired, repairs = normalise_context_response_containers(
            {"events": malformed})
        assert len(repaired["events"]) == 1
        assert "events+trailing_fields" in repairs


def test_surface_report_separates_product_and_provider_failures(tmp_path):
    run = tmp_path / "v2-evidence-r1"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "profile": "evidence", "routing_policy": "compiled",
        "corpus_manifest_sha256": "same"}))
    rows = [{
        "case_id": "a", "family": "irrelevant", "status": "answered",
        "history_smape": 2.0, "context_smape": 2.0,
        "incremental_smape": 0.0, "should_influence": False,
        "primary_changed": False, "applied": False,
        "publication_parity": True, "history_calls": 1, "context_calls": 1,
        "compiler_calls": 1, "compiler_called": 1,
    }, {
        "case_id": "b", "family": "future_covariate",
        "status": "product_failure", "failure_class": "agent_non_submission",
        "error": "agent did not submit context forecast trajectory",
    }, {
        "case_id": "c", "family": "future_covariate", "status": "error",
        "error": "OpenRouterError: unavailable",
    }]
    (run / "observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    report = aggregate([run])["arms"]["evidence:compiled"]
    assert report["attempted_observations"] == 3
    assert report["successful_pairs"] == 1
    assert report["failures"] == {
        "agent_non_submission": 1, "provider_failure": 1}
    assert report["receipt_logical_model_calls"] == 1
    assert report["receipt_generation_model_calls"] == 1
    assert report["receipts_generated"] == 1
    assert report["publication_parity"]["identity_matched"] == 1
    assert report["routing_policy"] == "compiled"


def test_surface_report_pairs_compiled_profiles_and_checks_forecast_parity(tmp_path):
    runs = []
    for profile in ("core", "evidence"):
        run = tmp_path / f"{profile}-r1"
        run.mkdir()
        (run / "summary.json").write_text(json.dumps({
            "profile": profile, "routing_policy": "compiled",
            "replicate_id": "1", "corpus_manifest_sha256": "same",
        }))
        row = {
            "case_id": "a", "family": "future_covariate",
            "status": "answered", "history_smape": 2.0,
            "context_smape": 1.0, "incremental_smape": 1.0,
            "history_forecast": [1.0, 2.0],
            "context_forecast": [2.0, 3.0],
            "should_influence": True, "primary_changed": True,
            "applied": True, "publication_parity": True,
            "history_calls": 1, "context_calls": 1,
        }
        (run / "observations.jsonl").write_text(json.dumps(row) + "\n")
        runs.append(run)
    report = aggregate(runs)
    assert set(report["arms"]) == {"core:compiled", "evidence:compiled"}
    assert report["cross_surface_forecast_parity"] == {
        "comparable_case_replicates": 1, "matched": 1, "rate": 1.0}


def test_surface_runner_retries_infrastructure_and_keeps_attempt_ledger(
    tmp_path, monkeypatch,
):
    corpus = tmp_path / "corpus"
    monkeypatch.setattr("sys.argv", [
        "generate", "--output-dir", str(corpus), "--seed", "91",
        "--per-family", "1",
    ])
    assert generate_main() == 0
    calls = {"count": 0}

    class Client:
        base_url = "https://example.invalid/v1"

        def __init__(self, *args, **kwargs):
            pass

    def fake_run(case, oracle, client, profile, work, receipts,
                 routing_policy, baseline_mode, tool_timeout):
        calls["count"] += 1
        common = {
            "case_id": case.case_id, "family": case.family,
            "should_influence": oracle.should_influence,
            "routing_policy": routing_policy,
            "baseline_mode": baseline_mode,
            "tool_timeout": tool_timeout,
            "llm_usage": {scope: {"prompt_tokens": 1, "completion_tokens": 2,
                                   "requests": 1, "cost_usd": 0.0,
                                   "truncation_escalations": 0}
                          for scope in ("agent", "compiler", "total")},
            "usage_accounting_version": 2,
        }
        if calls["count"] == 1:
            return {**common, "status": "error",
                    "failure_class": "provider_failure"}
        return {**common, "status": "answered", "history_smape": 2.0,
                "context_smape": 2.0, "incremental_smape": 0.0,
                "applied": False, "primary_changed": False,
                "temporal_leakage": False, "publication_parity": True,
                "history_calls": 1, "context_calls": 1}

    monkeypatch.setattr(surface_runner, "OpenRouterClient", Client)
    monkeypatch.setattr(surface_runner, "run_case", fake_run)
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", [
        "run-surfaces", "--corpus-dir", str(corpus),
        "--output-dir", str(output), "--profile", "evidence",
        "--model", "test", "--context-receipts-dir", str(tmp_path / "r"),
        "--limit", "1", "--infrastructure-retries", "1",
    ])
    assert surface_runner.main() == 0
    assert len((output / "attempts.jsonl").read_text().splitlines()) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["execution_attempts"] == 2
    assert summary["retried_cases"] == 1
    assert summary["llm_usage_observations"]["total"]["requests"] == 2
    assert summary["run_provenance"]["baseline_mode"] == "engine"


def test_replicated_report_requires_distinct_complete_corpora(
    tmp_path, monkeypatch,
):
    runs = []
    for seed in (101, 102):
        corpus, output = tmp_path / f"corpus-{seed}", tmp_path / f"run-{seed}"
        monkeypatch.setattr("sys.argv", [
            "generate", "--output-dir", str(corpus), "--seed", str(seed),
            "--per-family", "1",
        ])
        assert generate_main() == 0
        monkeypatch.setattr("sys.argv", [
            "run", "--corpus-dir", str(corpus), "--output-dir", str(output),
            "--allow-gate-failure",
        ])
        assert run_main() == 0
        runs.append(output)
    report = aggregate_contextbench(runs, minimum_replicates=2)
    assert report["replicate_count"] == 2
    assert report["unique_corpus_manifests"] == 2
    with pytest.raises(ValueError, match="duplicate corpus"):
        aggregate_contextbench([runs[0], runs[0]], minimum_replicates=2)
    changed = json.loads((runs[1] / "summary.json").read_text())
    changed["generator"] = "contextbench-stress-v1"
    (runs[1] / "summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cannot mix clean and stress"):
        aggregate_contextbench(runs, minimum_replicates=2)


def test_oracle_and_counterfactual_scoring_are_sensitive_not_tautological():
    _, raw_oracles = generate(11, per_family=2)
    oracles = list(load_oracles_from_rows(raw_oracles).values())
    first, second = oracles[0], oracles[1]
    assert smape(first.actual, list(first.actual)) == 0.0
    assert smape(first.actual, list(second.actual)) > 0.0
    influenced = next(item for item in oracles if item.should_influence)
    assert smape(influenced.actual, list(influenced.counterfactual)) > 0.0
