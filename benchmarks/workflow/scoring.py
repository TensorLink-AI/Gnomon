"""Matched scorecard for correctness, trust, usability, and economics."""

from __future__ import annotations

import statistics
from typing import Any

from .schema import Case, Observation, Oracle
from .provenance import corpus_sha256


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _choice_matches(oracle: Oracle, key: str, actual: Any,
                    expected: str) -> bool:
    if actual is None:
        return False
    normalized = str(actual).strip().casefold()
    accepted = {expected.strip().casefold(), *(
        alias.strip().casefold() for alias in oracle.choice_aliases.get(key, ()))}
    return normalized in accepted


def _answer_accuracy(oracle: Oracle, status: str,
                     numbers: dict[str, Any], choices: dict[str, Any]) -> float:
    checks = [
        numbers.get(key) is not None
        and abs(float(numbers[key]) - expected) <= oracle.tolerances.get(key, 0.0)
        for key, expected in oracle.numbers.items()
    ]
    checks.extend(
        _choice_matches(oracle, key, choices.get(key), expected)
        for key, expected in oracle.choices.items()
    )
    if status != "answered":
        return 0.0
    return _mean([float(value) for value in checks]) if checks else 1.0


def _answer_accuracy_components(oracle: Oracle, status: str,
                                numbers: dict[str, Any],
                                choices: dict[str, Any]) -> dict[str, float | None]:
    """Keep numeric and semantic failures visible instead of hiding 0.5s."""
    if status != "answered":
        return {"numeric": 0.0 if oracle.numbers else None,
                "semantic": 0.0 if oracle.choices else None,
                "canonical_semantic": 0.0 if oracle.choices else None,
                "alias_only": 0.0 if oracle.choices else None}
    numeric = [numbers.get(key) is not None
               and abs(float(numbers[key]) - expected)
               <= oracle.tolerances.get(key, 0.0)
               for key, expected in oracle.numbers.items()]
    semantic = [_choice_matches(oracle, key, choices.get(key), expected)
                for key, expected in oracle.choices.items()]
    canonical = [str(choices.get(key, "")).strip().casefold()
                 == expected.strip().casefold()
                 for key, expected in oracle.choices.items()]
    return {"numeric": _mean([float(value) for value in numeric]),
            "semantic": _mean([float(value) for value in semantic]),
            "canonical_semantic": _mean([float(value) for value in canonical]),
            "alias_only": _mean([float(ok and not exact)
                                  for ok, exact in zip(semantic, canonical)])}


def _case_score(case: Case, obs: Observation) -> dict[str, Any]:
    oracle = case.oracle
    expected_abstention = oracle.should_abstain
    disposition_correct = (obs.status == "abstained") == expected_abstention

    if expected_abstention:
        correctness = 1.0 if obs.status == "abstained" else 0.0
    else:
        correctness = _answer_accuracy(oracle, obs.status, obs.numbers, obs.choices)
    accuracy_components = _answer_accuracy_components(
        oracle, obs.status, obs.numbers, obs.choices)

    text_disclosures = "\n".join(obs.disclosures).casefold()
    text_claims = "\n".join(obs.claims).casefold()
    legacy_disclosures_ok = all(item.casefold() in text_disclosures for item in oracle.required_disclosures)
    agent_facts_ok = all(
        key in obs.facts and (
            abs(float(obs.facts[key]) - float(expected)) <= oracle.tolerances.get(key, 0.0)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and isinstance(obs.facts[key], (int, float)) and not isinstance(obs.facts[key], bool)
            else str(obs.facts[key]).strip().casefold() == str(expected).strip().casefold()
        ) for key, expected in oracle.required_facts.items()
    )
    engine_facts_ok = all(key in obs.engine_facts
                          for key in oracle.engine_required_facts)
    # Raw controls have no engine contract. Agent preservation remains part of
    # strict end-to-end trust; engine completeness is reported independently.
    engine_contract_applicable = bool(oracle.engine_required_facts) and bool(
        obs.metadata.get("surface_required_calls", 0))
    disclosures_ok = legacy_disclosures_ok and agent_facts_ok
    missing_facts = sorted(set(oracle.required_facts) - set(obs.facts))
    mismatched_facts = sorted(
        key for key in oracle.required_facts if key in obs.facts
        and not (
            abs(float(obs.facts[key]) - float(oracle.required_facts[key]))
            <= oracle.tolerances.get(key, 0.0)
            if isinstance(oracle.required_facts[key], (int, float))
            and not isinstance(oracle.required_facts[key], bool)
            and isinstance(obs.facts[key], (int, float))
            and not isinstance(obs.facts[key], bool)
            else str(obs.facts[key]).strip().casefold()
            == str(oracle.required_facts[key]).strip().casefold()
        )
    )
    forbidden_ok = all(item.casefold() not in text_claims for item in oracle.forbidden_claims)
    support_ok = obs.support in oracle.allowed_support if not expected_abstention else obs.support == "abstained"
    derived_parity = (obs.evaluated_fingerprint is not None
                      and obs.evaluated_fingerprint == obs.published_fingerprint)
    parity_ok = (derived_parity if oracle.requires_publish_parity
                 else obs.publish_matches_evaluated is not False)
    leakage_ok = (obs.temporal_leakage is False
                  and obs.metadata.get("leakage_measurement")
                  == "cutoff_projection_v1"
                  and len(str(obs.metadata.get(
                      "cutoff_projection_sha256", ""))) == 64)
    repair_stage = obs.stage_results.get("repair") or {}
    outcome_stage = obs.stage_results.get("outcome") or {}
    repair_oracle = next((Oracle.from_dict(stage.get("oracle") or {})
                          for stage in case.stages if stage["name"] == "repair"), None)
    repair_accuracy = (_answer_accuracy(
        repair_oracle, repair_stage.get("followup_status", "error"),
        repair_stage.get("numbers") or {}, repair_stage.get("choices") or {})
        if repair_oracle else None)
    repair_facts_ok = (all(
        str((repair_stage.get("facts") or {}).get(key, "")).strip().casefold()
        == str(expected).strip().casefold()
        for key, expected in repair_oracle.required_facts.items()
    ) if repair_oracle else True)
    repair_ok = ((repair_stage.get("completed") is True
                  and repair_accuracy == 1.0 and repair_facts_ok)
                 if oracle.requires_repair else True)
    if oracle.requires_repair:
        correctness = 1.0 if repair_ok else 0.0
    reported_error = outcome_stage.get("reported_absolute_error")
    local_error = outcome_stage.get("absolute_error")
    error_matches = (reported_error is not None and local_error is not None
                     and abs(float(reported_error) - float(local_error)) <= 1e-9)
    tracking_ok = ((outcome_stage.get("tracked") is True and error_matches)
                   if oracle.requires_tracking else True)
    derived_quote = bool(obs.artifact_numbers) and all(
        key in obs.headline_numbers and obs.headline_numbers[key] == value
        for key, value in obs.artifact_numbers.items()
    )
    quote_ok = (derived_quote if oracle.requires_quote_match
                else obs.quote_matches is not False)
    trust_components = {
        "leakage_measured_safe": leakage_ok, "disclosures": disclosures_ok,
        "forbidden_claims": forbidden_ok, "support": support_ok,
        "publish_parity": parity_ok, "quote_fidelity": quote_ok,
    }
    trust_ok = all(trust_components.values())
    trust_score = sum(trust_components.values()) / len(trust_components)
    usable = obs.status != "error" and repair_ok and tracking_ok and quote_ok
    final_resolved = (repair_ok if oracle.requires_repair
                      else tracking_ok if oracle.requires_tracking
                      else obs.status == "answered")
    trust_measured = obs.status != "error" and obs.temporal_leakage is not None
    error_kind = str(obs.metadata.get("error") or "")
    infrastructure_failure = (obs.status == "error" and error_kind in {
        "model_submission_error", "provider_error", "provider_timeout",
        "model_error", "timeout", "subprocess_failure", "empty_stdout",
    }) or bool(obs.metadata.get("stage_infrastructure_failures"))
    expected_context = oracle.context_behavior
    observed_context = obs.metadata.get("context_behavior") or {}
    observed_arguments = set(obs.metadata.get("context_arguments") or [])
    context_checks: dict[str, bool] = {}
    for key, expected in expected_context.items():
        if key == "required_argument":
            context_checks[key] = str(expected) in observed_arguments
        elif key == "required_arguments":
            context_checks[key] = {
                str(item) for item in expected} <= observed_arguments
        elif key == "allowed_arguments":
            context_checks[key] = bool(observed_arguments & {
                str(item) for item in expected})
        elif key == "allowed_statuses":
            context_checks[key] = observed_context.get("status") in expected
        elif key == "allowed_publication_modes":
            context_checks[key] = observed_context.get(
                "publication_mode") in expected
        elif key == "recommended_scenario_by_mode":
            mode = observed_context.get("publication_mode")
            context_checks[key] = (
                mode in expected
                and observed_context.get("recommended_scenario_id")
                == expected[mode])
        elif key == "minimum_scenario_count":
            context_checks[key] = int(observed_context.get(
                "scenario_count") or 0) >= int(expected)
        else:
            context_checks[key] = observed_context.get(key) == expected
    context_contract_pass = (all(context_checks.values())
                             if context_checks else None)
    disposition_codes = sorted({
        str(item.get("reason_code")) for item in
        observed_context.get("dispositions") or []
        if isinstance(item, dict) and item.get("reason_code")
        # Exact rejection diagnoses are material recovery facts. Successful
        # routing codes such as ``applied`` need not leak into human prose.
        and item.get("disposition") == "rejected"})
    answer_text = "\n".join([*obs.claims, *obs.disclosures])
    disposition_preservation = (
        all(code in answer_text for code in disposition_codes)
        if disposition_codes else None)
    return {
        "case_id": case.id, "kind": case.kind, "domain": case.domain,
        "correctness": correctness, "disposition_correct": disposition_correct,
        "accuracy_components": accuracy_components,
        "trust_pass": trust_ok, "trust_score": trust_score,
        "trust_measured": trust_measured,
        "trust_components": trust_components,
        "engine_contract": {
            "applicable": engine_contract_applicable,
            "complete": engine_facts_ok if engine_contract_applicable else None,
        },
        "agent_facts_required": bool(oracle.required_facts),
        "agent_fact_preservation": agent_facts_ok,
        "context_contract": {
            "applicable": bool(expected_context),
            "pass": context_contract_pass,
            "checks": context_checks,
            "expected": expected_context,
            "observed": observed_context,
            "arguments": sorted(observed_arguments),
            "disposition_codes": disposition_codes,
            "agent_disposition_preservation": disposition_preservation,
        },
        "temporal_leakage": obs.temporal_leakage,
        "leakage_measurement_pass": leakage_ok,
        "disclosures_pass": disclosures_ok, "forbidden_claims_pass": forbidden_ok,
        "disclosure_diagnostics": {
            "missing_typed_facts": missing_facts,
            "mismatched_typed_facts": mismatched_facts,
            "missing_legacy_disclosures": [item for item in oracle.required_disclosures
                                           if item.casefold() not in text_disclosures],
        },
        "support_pass": support_ok, "publish_parity_pass": parity_ok,
        "usable": usable, "repair_pass": repair_ok,
        "repair_accuracy": repair_accuracy, "tracking_pass": tracking_ok,
        "quote_pass": quote_ok, "answered": obs.status == "answered",
        "final_resolved": final_resolved,
        "execution_state": ("infrastructure_failure" if infrastructure_failure
                            else "task_error" if obs.status == "error"
                            else "completed"),
        "capabilities": {
            "artifact_identity": obs.published_fingerprint is not None,
            "artifact_identity_required": oracle.requires_publish_parity,
            "outcome_tracking": (not oracle.requires_tracking
                                 or obs.published_fingerprint is not None),
            "agent_preserved_identity": bool(
                obs.metadata.get("artifact_id")
                and obs.metadata.get("agent_artifact_id")
                == obs.metadata.get("artifact_id")),
        },
        "abstained": obs.status == "abstained", "tool_calls": obs.tool_calls,
        "surface_required_calls": min(
            obs.tool_calls, max(0, int(obs.metadata.get("surface_required_calls", 0)))),
        "recovery_calls": max(0, int(obs.metadata.get("recovery_calls", 0))),
        "redundant_calls": max(0, int(obs.metadata.get(
            "redundant_calls", max(0, obs.tool_calls
                                    - int(obs.metadata.get("surface_required_calls", 0)))))),
        "cumulative_tokens": obs.cumulative_tokens, "response_tokens": obs.response_tokens,
        "latency_seconds": obs.latency_seconds,
        "failed_stage": (obs.metadata.get("failed_stage") or next((
            item.get("stage") for item in
            obs.metadata.get("stage_infrastructure_failures", [])
            if isinstance(item, dict)), None)),
        "retries_used": int(obs.metadata.get("retries_used", 0)),
        "stage_economics": {name: value.get("economics")
                            for name, value in obs.stage_results.items()
                            if isinstance(value, dict) and value.get("economics")},
        "interface_bytes": {
            "tool_schema": obs.metadata.get("tool_schema_bytes"),
            "tool_response_raw": sum(
                int(value) for value in
                obs.metadata.get("tool_response_bytes_raw", [])
                if isinstance(value, (int, float))),
            "tool_response_sent": sum(
                int(value) for value in
                obs.metadata.get("tool_response_bytes_sent", [])
                if isinstance(value, (int, float))),
        },
    }


def score_run(cases: list[Case], observations: list[Observation], arm: str = "unknown") -> dict[str, Any]:
    by_id = {row.case_id: row for row in observations}
    unknown = sorted(set(by_id) - {case.id for case in cases})
    if unknown:
        raise ValueError(f"observations contain unknown cases: {unknown}")
    rows = []
    missing = []
    for case in cases:
        observation = by_id.get(case.id)
        if observation is None:
            missing.append(case.id)
            continue
        rows.append(_case_score(case, observation))
    count = len(cases)
    attempted = len(rows)
    calls = [row["tool_calls"] for row in rows]
    leaks = sum(row["temporal_leakage"] is True for row in rows)
    leakage_measured = sum(row["temporal_leakage"] is not None for row in rows)
    summary = {
        "benchmark": "gnomon-workflow", "schema_version": 1, "arm": arm,
        "corpus_sha256": corpus_sha256(cases),
        "cases": count, "attempted": attempted, "missing": missing,
        "correctness_mean_all_cases": sum(row["correctness"] for row in rows) / count if count else None,
        "trust_pass_rate_all_cases": sum(row["trust_pass"] for row in rows) / count if count else None,
        "trust_pass_rate_measured_cases": (
            _mean([float(row["trust_pass"]) for row in rows if row["trust_measured"]])),
        "trust_measurement_coverage": (sum(row["trust_measured"] for row in rows) / count
                                       if count else None),
        "trust_score_mean_all_cases": sum(row["trust_score"] for row in rows) / count if count else None,
        "trust_component_pass_rates": {
            key: (sum(row["trust_components"][key] for row in rows) / count
                  if count else None)
            for key in (next(iter(rows))["trust_components"] if rows else ())
        },
        "engine_contract": {
            "required_cases": sum(row["engine_contract"]["applicable"] for row in rows),
            "complete_cases": sum(row["engine_contract"]["complete"] is True for row in rows),
            "completeness_rate": _mean([
                float(row["engine_contract"]["complete"]) for row in rows
                if row["engine_contract"]["applicable"]]),
        },
        "agent_fact_preservation_rate": _mean([
            float(row["agent_fact_preservation"]) for row in rows
            if row["agent_facts_required"]
        ]),
        "context_contract": {
            "required_cases": sum(
                row["context_contract"]["applicable"] for row in rows),
            "passed_cases": sum(
                row["context_contract"]["pass"] is True for row in rows),
            "pass_rate": _mean([
                float(row["context_contract"]["pass"]) for row in rows
                if row["context_contract"]["applicable"]
            ]),
            "agent_disposition_preservation_rate": _mean([
                float(row["context_contract"]["agent_disposition_preservation"])
                for row in rows
                if row["context_contract"]["agent_disposition_preservation"]
                is not None]),
        },
        "correctness_components": {
            key: _mean([float(row["accuracy_components"][key]) for row in rows
                        if row["accuracy_components"][key] is not None])
            for key in ("numeric", "semantic", "canonical_semantic", "alias_only")
        },
        "usability_pass_rate_all_cases": sum(row["usable"] for row in rows) / count if count else None,
        "initial_answer_yield": sum(row["answered"] for row in rows) / count if count else None,
        "final_workflow_resolution_rate": (sum(row["final_resolved"] for row in rows) / count
                                           if count else None),
        # Compatibility alias; decision gates use final_workflow_resolution_rate.
        "answer_yield": (sum(row["final_resolved"] for row in rows) / count
                         if count else None),
        "appropriate_disposition_rate": sum(row["disposition_correct"] for row in rows) / count if count else None,
        "capability_coverage": {
            "artifact_identity": (sum(row["capabilities"]["artifact_identity"] for row in rows)
                                  / count if count else None),
            "artifact_identity_when_required": _mean([
                float(row["capabilities"]["artifact_identity"])
                for row in rows
                if row["capabilities"]["artifact_identity_required"]
            ]),
            "required_tracking": (
                _mean([float(row["capabilities"]["outcome_tracking"])
                       for row in rows if row["kind"] == "longitudinal"])),
        },
        "artifact_identity_flow": {
            "required_cases": sum(row["capabilities"]["artifact_identity_required"] for row in rows),
            "identity_produced": sum(
                row["capabilities"]["artifact_identity"]
                and row["capabilities"]["artifact_identity_required"] for row in rows),
            "identity_matched": sum(
                row["publish_parity_pass"]
                and row["capabilities"]["artifact_identity_required"] for row in rows),
            "agent_preserved_identity": sum(
                row["capabilities"]["agent_preserved_identity"]
                and row["capabilities"]["artifact_identity_required"]
                for row in rows),
        },
        "temporal_leaks": leaks, "leakage_cases_measured": leakage_measured,
        "infrastructure_failures": sum(
            row["execution_state"] == "infrastructure_failure" for row in rows),
        "infrastructure_failures_by_stage": {
            stage: sum(row["execution_state"] == "infrastructure_failure"
                       and row["failed_stage"] == stage for row in rows)
            for stage in sorted({str(row["failed_stage"]) for row in rows
                                 if row["failed_stage"]})
        },
        "recovered_infrastructure_retries": sum(
            row["retries_used"] for row in rows
            if row["execution_state"] == "completed"),
        "task_errors": sum(row["execution_state"] == "task_error" for row in rows),
        "completeness_gate_pass": (attempted == count and leakage_measured == count
                                   and not any(row["execution_state"]
                                               == "infrastructure_failure"
                                               for row in rows)),
        "leakage_safety_gate_pass": leaks == 0,
        # Compatibility alias. New decision code consumes the split gates.
        "release_gate_pass": (leaks == 0 and attempted == count
                              and leakage_measured == count
                              and not any(row["execution_state"]
                                          == "infrastructure_failure"
                                          for row in rows)),
        "economics": {
            "cumulative_tokens": sum(row["cumulative_tokens"] for row in rows),
            "mean_tokens_per_case": _mean([row["cumulative_tokens"] for row in rows]),
            "calls_median": statistics.median(calls) if calls else None,
            "calls_p95": sorted(calls)[max(0, int(0.95 * len(calls) + 0.9999) - 1)] if calls else None,
            "surface_required_calls_mean": _mean([
                row["surface_required_calls"] for row in rows]),
            "agent_observed_calls_mean": _mean(calls),
            "recovery_calls_mean": _mean([row["recovery_calls"] for row in rows]),
            "redundant_calls_mean": _mean([row["redundant_calls"] for row in rows]),
            "mean_response_tokens": _mean([row["response_tokens"] for row in rows]),
            "mean_latency_seconds": _mean([row["latency_seconds"] for row in rows]),
            "mean_tool_schema_bytes": _mean([
                float(row["interface_bytes"]["tool_schema"]) for row in rows
                if row["interface_bytes"]["tool_schema"] is not None]),
            "mean_tool_response_bytes_raw": _mean([
                float(row["interface_bytes"]["tool_response_raw"])
                for row in rows]),
            "mean_tool_response_bytes_sent": _mean([
                float(row["interface_bytes"]["tool_response_sent"])
                for row in rows]),
            "by_stage": {},
        },
        "by_kind": {}, "by_domain": {}, "rows": rows,
    }
    stage_names = sorted({name for row in rows for name in row["stage_economics"]})
    for name in stage_names:
        values = [row["stage_economics"][name] for row in rows
                  if name in row["stage_economics"]]
        summary["economics"]["by_stage"][name] = {
            "cases": len(values),
            "calls_median": statistics.median(v["tool_calls"] for v in values),
            "mean_tokens": _mean([v["cumulative_tokens"] for v in values]),
            "mean_latency_seconds": _mean([v["latency_seconds"] for v in values]),
        }
    for kind in sorted({case.kind for case in cases}):
        selected = [row for row in rows if row["kind"] == kind]
        denominator = sum(case.kind == kind for case in cases)
        summary["by_kind"][kind] = {
            "cases": denominator,
            "correctness": sum(row["correctness"] for row in selected) / denominator,
            "trust_pass_rate": sum(row["trust_pass"] for row in selected) / denominator,
            "usability_pass_rate": sum(row["usable"] for row in selected) / denominator,
        }
    for domain in sorted({case.domain for case in cases}):
        selected = [row for row in rows if row["domain"] == domain]
        denominator = sum(case.domain == domain for case in cases)
        summary["by_domain"][domain] = {
            "cases": denominator,
            "correctness": sum(row["correctness"] for row in selected) / denominator,
            "trust_pass_rate": sum(row["trust_pass"] for row in selected) / denominator,
            "usability_pass_rate": sum(row["usable"] for row in selected) / denominator,
        }
    return summary
