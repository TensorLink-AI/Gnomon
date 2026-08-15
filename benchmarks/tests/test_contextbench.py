from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.contextbench.generate import generate, main as generate_main
from benchmarks.contextbench.run_contextbench import smape, summarize
from benchmarks.contextbench.run_contextbench import main as run_main
from benchmarks.contextbench.run_llm import (
    compile_events, raw_case, safe_payload,
)
from benchmarks.contextbench.run_surfaces import surface_row
from benchmarks.contextbench.report_surfaces import aggregate
from benchmarks.contextbench.schema import Case, load_cases, load_oracles


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
    assert manifest["generator"] == "contextbench-synthetic-v1"
    assert manifest["fresh_seed"] is False
    assert len(manifest["cases_sha256"]) == 64
    assert len(manifest["oracle_sha256"]) == 64


def test_smape_is_symmetric_and_zero_safe():
    assert smape([0.0, 10.0], [0.0, 12.0]) == smape([0.0, 12.0], [0.0, 10.0])
    assert smape([0.0], [0.0]) == 0.0


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
    assert row["input"]["future_covariates"]
    assert row["input"]["covariate_mapping"][0]["type"] == "binary"
    assert len(row["input"]["history"]["value"]) == len(case.history)


def test_history_surface_row_excludes_all_outside_context():
    raw_cases, _ = generate(32, per_family=1)
    case = Case.from_dict(next(
        row for row in raw_cases if row["family"] == "future_covariate"))
    row = surface_row(case, include_context=False)
    assert row["tier"] == "T2"
    assert row["input"] == {"history": {"value": list(case.history)}}
    assert case.narrative not in row["prompt"]


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
        "known_at": "2025-01-01T00:00:00+00:00",
        "evidence_quote": quote, "effect_family": "temporary_pulse",
        "direction": "unknown", "duration": "temporary",
        "attributes": {"magnitude": 999999},
    }]})
    compiled = compile_events(repeated, client)
    assert len(compiled["events"]) == 1
    attributes = compiled["events"][0]["attributes"]
    assert "magnitude" not in attributes
    assert attributes["evidence_quote"] == quote


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
    compiled = compile_events(case, _ScriptedClient({
        "events": encoded, "hypotheses": "[]"}))
    assert len(compiled["events"]) == 1
    assert compiled["events"][0]["entity_scope"] == ["*"]
    assert compiled["container_coercions"]


def test_surface_report_separates_product_and_provider_failures(tmp_path):
    run = tmp_path / "v2-evidence-r1"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"profile": "evidence"}))
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
    report = aggregate([run])["arms"]["evidence"]
    assert report["attempted_observations"] == 3
    assert report["successful_pairs"] == 1
    assert report["failures"] == {
        "agent_non_submission": 1, "provider_failure": 1}
    assert report["compiler_logical_calls"] == 1
    assert report["compiler_executed_calls"] == 1
    assert report["publication_parity"]["identity_matched"] == 1


def test_oracle_and_counterfactual_scoring_are_sensitive_not_tautological():
    _, raw_oracles = generate(11, per_family=2)
    oracles = list(load_oracles_from_rows(raw_oracles).values())
    first, second = oracles[0], oracles[1]
    assert smape(first.actual, list(first.actual)) == 0.0
    assert smape(first.actual, list(second.actual)) > 0.0
    influenced = next(item for item in oracles if item.should_influence)
    assert smape(influenced.actual, list(influenced.counterfactual)) > 0.0
