import json
from pathlib import Path

from benchmarks.gfr import SAFETY_INVARIANTS
from benchmarks.gfr_full import (
    SEEDS,
    TASK_CASE_NAMES,
    _calibration_raw,
    _selection_raw,
    _usage_raw,
    assemble,
    calibration_family_observations,
    categorical_context_observation,
    context_efficiency_observations,
    context_observations,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _identity(method: str) -> dict:
    return {
        "method": method,
        "model": "deepseek",
        "base_url": "https://example.test/v1",
        "temperature": 1.0,
        "selected_tasks": list(TASK_CASE_NAMES),
        "seed_start": 7,
        "seeds": 3,
        "n_samples": 50,
        "fail_on_invalid": True,
        "sample_parallelism": 2,
        "mcp_profile": "evidence" if method == "gnomon-mcp" else None,
        "code_revision": "revision-under-test",
    }


def _candidates() -> list[dict]:
    return [
        {
            "role": "immutable_primary",
            "selected": True,
            "human_selection_eligible": True,
            "score": 1.0,
            "nominal_coverage": 0.8,
            "empirical_coverage": 0.8,
            "wis": 1.0,
            "computed_after_forecast": True,
            "passed_to_forecaster": False,
        },
        {
            "role": "governed_categorical_state_mapping",
            "selected": False,
            "human_selection_eligible": True,
            "score": 2.0,
            "nominal_coverage": 0.8,
            "empirical_coverage": 0.75,
            "wis": 1.2,
            "computed_after_forecast": True,
            "passed_to_forecaster": False,
        },
    ]


def test_extractors_fail_closed_on_incomplete_measurements() -> None:
    assert _usage_raw(
        {"llm_usage": {"requests": 1, "prompt_tokens": 2,
                       "completion_tokens": 3}, "total_time": 1},
        {"llm_usage": {"requests": 1, "prompt_tokens": 2,
                       "completion_tokens": 0}, "total_time": 1},
    ) is None
    assert _usage_raw(
        {"llm_usage": {"requests": 1, "prompt_tokens": 2,
                       "completion_tokens": 3, "sample_cache_hits": 1,
                       "sample_cache_accounting_complete": False},
         "total_time": 1},
        {"llm_usage": {"requests": 1, "prompt_tokens": 2,
                       "completion_tokens": 3}, "total_time": 1},
    ) is None
    assert _usage_raw(
        {"llm_usage": {"requests": 2, "prompt_tokens": 4,
                       "completion_tokens": 6, "sample_cache_hits": 1,
                       "sample_cache_accounting_complete": True},
         "total_time": .1},
        {"llm_usage": {"requests": 1, "prompt_tokens": 2,
                       "completion_tokens": 3}, "total_time": .1},
        {"latency_seconds": 9.0}, {"latency_seconds": 4.0},
    )["control_latency_seconds"] == 9.0
    assert _calibration_raw({"candidates": _candidates()}, {}) == {
        "nominal_coverage": 0.8,
        "empirical_coverage": 0.75,
        "candidate_wis": 1.2,
        "reference_wis": 1.0,
        "candidate_relationship": "evaluated_candidate",
    }
    assert _selection_raw({
        "selected_score": 3.0, "candidates": _candidates(),
    }) is None


def test_context_cases_are_bound_by_semantic_identity() -> None:
    standard = [
        {"case_id": "ctx-repeated_event-0000", "family": "repeated_event",
         "should_influence": True, "applied": True},
        {"case_id": "ctx-future_covariate-0000", "family": "future_covariate",
         "should_influence": True, "applied": True},
        {"case_id": "ctx-irrelevant-0000", "family": "irrelevant",
         "should_influence": False, "applied": False},
    ]
    stress = [
        {"case_id": "stress-constraint-true-0000", "family": "numeric_claim",
         "should_influence": True, "applied": True},
        {"case_id": "stress-scope-0000", "family": "entity_scope",
         "should_influence": False, "applied": False},
        {"case_id": "stress-bitemporal-0000", "family": "bitemporal_context",
         "should_influence": True, "applied": False},
    ]
    observed = context_observations(standard, stress)

    assert set(observed) == {
        "context:useful:aperiodic-pulse",
        "context:useful:numeric-driver",
        "context:useful:bounded-event",
        "context:neutral:no-effect",
        "context:neutral:wrong-entity",
        "context:leakage:future-revision",
    }
    assert categorical_context_observation([{
        "role": "governed_categorical_state_mapping",
        "human_selection_eligible": True,
        "effect": {"validation": {"beats_baseline": True}},
    }]) == {
        "context_is_useful": True, "context_admitted": True,
    }
    assert observed["context:leakage:future-revision"] == {
        "context_is_useful": True, "context_admitted": False,
    }


def test_calibration_families_compare_current_to_prior_strict_protocol():
    observed = calibration_family_observations({
        "strict_by_family": {
            "intermittent": {"coverage": .81, "mean_wis": 1.0},
            "heteroskedastic": {"coverage": .79, "mean_wis": 2.0},
        },
        "strict_reference_by_family": {
            "intermittent": {"coverage": .6, "mean_wis": 2.0},
            "heteroskedastic": {"coverage": .7, "mean_wis": 2.5},
        },
    })

    assert observed["calibration:intermittent:seed1"] == {
        "nominal_coverage": .8,
        "empirical_coverage": .81,
        "candidate_wis": 1.0,
        "reference_wis": 2.0,
        "candidate_relationship": "evaluated_candidate",
    }


def test_context_efficiency_requires_matched_useful_and_neutral_rows():
    raw = []
    compiled = []
    for case_id, useful in (("a", True), ("b", False)):
        common = {"case_id": case_id, "family": "x", "status": "answered",
                  "should_influence": useful}
        raw.append({**common, "requests": 3, "prompt_tokens": 100,
                    "completion_tokens": 20, "latency_seconds": 2.0})
        compiled.append({**common, "compiler_calls": 1, "prompt_tokens": 30,
                         "completion_tokens": 10,
                         "latency_seconds_total": 1.0})

    observed = context_efficiency_observations(raw, compiled)

    assert observed["efficiency:context:useful"]["control_requests"] == 3.0
    assert observed["efficiency:context:neutral"]["treatment_tokens"] == 40.0


def test_full_cik_assembler_binds_all_frozen_rows_and_safety(tmp_path: Path):
    control_dir = tmp_path / "control"
    treatment_dir = tmp_path / "treatment"
    output_dir = tmp_path / "output"
    _write_json(control_dir / "run_identity.json", _identity("control"))
    _write_json(
        treatment_dir / "run_identity.json", _identity("gnomon-mcp"),
    )

    control_rows = []
    treatment_rows = []
    diagnostics = []
    for task in TASK_CASE_NAMES:
        for seed in SEEDS:
            task_id = f"{task}-seed{seed}"
            control_rows.append({
                "task_id": task_id, "success": True, "rcrps": 2.0,
            })
            treatment_rows.append({
                "task_id": task_id,
                "success": True,
                "rcrps": 1.0,
                "benchmark_input_profile": {"passed_to_forecaster": False},
            })
            diagnostics.append({
                "task": task,
                "seed": seed,
                "selected_score": 1.0,
                "primary_forecast_unchanged": True,
                "candidates": _candidates(),
            })
            _write_json(
                treatment_dir / "mcp-traces" / f"{task}-seed{seed}.json",
                {
                    "final_submission": {"automation_eligible": False},
                    "context_compilation": {
                        "future_observations_exposed": False,
                    },
                },
            )
            usage = {
                "llm_usage": {
                    "requests": 1,
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                },
                "total_time": 1.0,
            }
            for directory in (control_dir, treatment_dir):
                path = directory / "runs" / task / str(seed) / "extra_info"
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(usage)
                if (directory == treatment_dir
                        and task == "DirectNormalIrradianceFromCloudStatus"
                        and seed == 8):
                    payload["publication"] = {"candidate_portfolio": [{
                        "role": "governed_categorical_state_mapping",
                        "human_selection_eligible": True,
                        "effect": {
                            "validation": {"beats_baseline": True},
                        },
                    }]}
                path.write_text(repr(payload), encoding="utf-8")
    _write_rows(control_dir / "gnomonbench.jsonl", control_rows)
    _write_rows(treatment_dir / "gnomonbench.jsonl", treatment_rows)
    _write_rows(
        treatment_dir / "selection-diagnostics.jsonl", diagnostics,
    )
    context_standard = tmp_path / "context-standard"
    context_stress = tmp_path / "context-stress"
    for directory in (context_standard, context_stress):
        _write_json(directory / "run_identity.json", {
            "code_revision": "revision-under-test",
        })
        _write_json(directory / "summary.json", {"complete": True})
    standard_rows = [
        {"case_id": "ctx-repeated_event-0000", "family": "repeated_event",
         "should_influence": True, "applied": True,
         "canonical_primary_preserved": True, "temporal_leakage": False},
        {"case_id": "ctx-future_covariate-0000", "family": "future_covariate",
         "should_influence": True, "applied": True,
         "canonical_primary_preserved": True, "temporal_leakage": False},
        {"case_id": "ctx-irrelevant-0000", "family": "irrelevant",
         "should_influence": False, "applied": False,
         "canonical_primary_preserved": True, "temporal_leakage": False},
    ]
    stress_rows = [
        {"case_id": "stress-constraint-true-0000", "family": "numeric_claim",
         "should_influence": True, "applied": True,
         "canonical_primary_preserved": True, "temporal_leakage": False},
        {"case_id": "stress-scope-0000", "family": "entity_scope",
         "should_influence": False, "applied": False,
         "canonical_primary_preserved": True, "temporal_leakage": False},
        {"case_id": "stress-bitemporal-0000", "family": "bitemporal_context",
         "should_influence": True, "applied": False,
         "canonical_primary_preserved": True, "temporal_leakage": False},
    ]
    _write_rows(context_standard / "observations.jsonl", standard_rows)
    _write_rows(context_stress / "observations.jsonl", stress_rows)
    authority_path = tmp_path / "authority.json"
    authority_ids = json.loads(Path(
        "benchmarks/gfr_protocol.json").read_text(encoding="utf-8"))[
            "capabilities"]["future_input_authority"]["full_case_ids"]
    _write_json(authority_path, {
        "evaluated_commit": "revision-under-test",
        "rows": [{
            "case_id": case_id,
            "classification_correct": True,
            "authority_escalated": False,
        } for case_id in authority_ids],
    })
    calibration_path = tmp_path / "calibration.json"
    _write_json(calibration_path, {
        "evaluated_commit": "revision-under-test",
        "cases": 120,
        "gates": {
            "all_cases_complete": True,
            "point_forecasts_unchanged": True,
        },
        "strict_by_family": {
            "intermittent": {"coverage": .81, "mean_wis": 1.0},
            "heteroskedastic": {"coverage": .79, "mean_wis": 2.0},
        },
        "strict_reference_by_family": {
            "intermittent": {"coverage": .6, "mean_wis": 2.0},
            "heteroskedastic": {"coverage": .7, "mean_wis": 2.5},
        },
    })
    bounded_path = tmp_path / "bounded.json"
    _write_json(bounded_path, {
        "evaluated_commit": "revision-under-test",
        "cases": 8,
        "nominal_coverage": .8,
        "empirical_coverage": .79,
        "candidate_wis": 1.0,
        "reference_wis": 1.2,
        "all_gates_passed": True,
    })
    shared_trend_path = tmp_path / "shared-trend.json"
    _write_json(shared_trend_path, {
        "evaluated_commit": "revision-under-test",
        "cases": 40,
        "context_is_useful": False,
        "context_admitted": False,
        "all_gates_passed": True,
    })
    context_raw = tmp_path / "context-raw"
    context_compiled = tmp_path / "context-compiled"
    context_identity = {
        "code_revision": "revision-under-test",
        "model": "deepseek",
        "base_url": "https://example.test/v1",
        "temperature": .2,
        "reasoning_effort": None,
        "corpus_manifest_sha256": "corpus",
        "selected_case_ids_sha256": "cases",
        "selected_cases": 2,
    }
    _write_json(context_raw / "run_identity.json", {
        **context_identity, "condition": "raw-llm",
    })
    _write_json(context_compiled / "run_identity.json", {
        **context_identity, "condition": "compiled-context",
    })
    raw_efficiency_rows = []
    compiled_efficiency_rows = []
    for case_id, useful in (("useful", True), ("neutral", False)):
        common = {"case_id": case_id, "family": "fixture",
                  "status": "answered", "should_influence": useful}
        raw_efficiency_rows.append({
            **common, "requests": 3, "prompt_tokens": 100,
            "completion_tokens": 20, "latency_seconds": 2.0,
        })
        compiled_efficiency_rows.append({
            **common, "compiler_calls": 1, "prompt_tokens": 30,
            "completion_tokens": 10, "latency_seconds_total": 1.0,
        })
    _write_rows(context_raw / "observations.jsonl", raw_efficiency_rows)
    _write_rows(context_compiled / "observations.jsonl",
                compiled_efficiency_rows)
    _write_json(context_raw / "summary.json", {"complete": True})
    _write_json(context_compiled / "summary.json", {"complete": True})
    repair_efficiency_path = tmp_path / "repair-efficiency.json"
    _write_json(repair_efficiency_path, {
        "evaluated_commit": "revision-under-test",
        "case_id": "efficiency:repair:one-retry",
        "all_gates_passed": True,
        "gfr_raw": {
            "control_requests": 2,
            "control_tokens": 120,
            "control_latency_seconds": 3.0,
            "treatment_requests": 1,
            "treatment_tokens": 40,
            "treatment_latency_seconds": 1.0,
        },
    })

    base = {
        "schema_version": "0.1",
        "scope": "full",
        "evaluated_commit": "revision-under-test",
        "evidence": [],
        "observations": [],
        "safety": {
            name: {"denominator": 1, "failures": 0}
            for name in SAFETY_INVARIANTS
        },
    }
    base_path = tmp_path / "base.json"
    _write_json(base_path, base)

    _, result_path = assemble(
        root=tmp_path,
        protocol_path=Path("benchmarks/gfr_protocol.json"),
        base_result=base_path,
        control_dir=control_dir,
        treatment_dir=treatment_dir,
        output_dir=output_dir,
        context_standard_dir=context_standard,
        context_stress_dir=context_stress,
        authority_path=authority_path,
        calibration_evaluation_path=calibration_path,
        bounded_calibration_path=bounded_path,
        shared_trend_path=shared_trend_path,
        context_raw_dir=context_raw,
        context_compiled_dir=context_compiled,
        repair_efficiency_path=repair_efficiency_path,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert len(result["observations"]) == 40
    assert all(item["status"] == "answered"
               for item in result["observations"])
    for name in (
        "temporal_leakage",
        "immutable_primary_mutation",
        "unsupported_automation",
        "authority_escalation",
        "benchmark_oracle_exposure",
    ):
        expected = (
            184 if name == "temporal_leakage"
            else 53 if name == "immutable_primary_mutation"
            else 15 if name == "authority_escalation"
            else 178 if name == "benchmark_oracle_exposure"
            else 9 if name == "declared_bound_violation"
            else 8 if name == "unsupported_automation" else 7)
        assert result["safety"][name] == {
            "denominator": expected, "failures": 0,
        }
    assert not list(output_dir.glob(".*.tmp-*"))
