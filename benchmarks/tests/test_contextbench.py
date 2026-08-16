from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.contextbench.generate import generate, main as generate_main
from benchmarks.contextbench.generate_stress import generate as generate_stress
from benchmarks.contextbench import run_surfaces as surface_runner
from benchmarks.contextbench.run_contextbench import (
    smape, summarize, valid_disposition,
)
from benchmarks.contextbench.run_contextbench import main as run_main
from benchmarks.contextbench.run_llm import (
    compile_events, raw_case, safe_payload,
)
from benchmarks.contextbench.run_surfaces import surface_row
from benchmarks.contextbench.report_surfaces import aggregate
from benchmarks.contextbench.report_contextbench import (
    aggregate as aggregate_contextbench,
)
from benchmarks.contextbench.schema import Case, load_cases, load_oracles
from gnomon.context_model import rolling_residuals


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
    claims = [row for row in dimensions
              if row["admission_warrant"] == "asserted"]
    assert {row["claim_truth"] for row in claims} == {"true", "false"}
    mixed = [row for row in cases if row["context_events"] and row["covariates"]]
    assert len(mixed) == 1 and mixed[0]["family"] == "confounded"


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
    assert row["_time_origin"] == "2025-01-01T00:00:00+00:00"
    assert row["input"]["future_covariates"]
    assert row["input"]["covariate_mapping"][0]["type"] == "binary"
    assert len(row["input"]["history"]["value"]) == len(case.history)
    assert "Input (JSON)" not in row["prompt"]
    assert "dataset is bound by the host" in row["prompt"]


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
        compiled = compile_events(case, _ScriptedClient({"events": malformed}))
        assert len(compiled["events"]) == 1
        assert "events+trailing_fields" in compiled["container_coercions"]


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
