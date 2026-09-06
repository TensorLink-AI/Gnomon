"""Matched scorecard for correctness, trust, usability, and economics."""

from __future__ import annotations

import statistics
import math
from typing import Any

from .schema import Case, Observation, Oracle
from .provenance import corpus_sha256
from .accounting import receipts, summarize


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
        for key, expected in oracle.numbers.items() if key not in oracle.forecast.get("keys", ())
    ]
    if oracle.forecast:
        checks.append(forecast_metrics(oracle, status, numbers)["within_mae_limit"])
    checks.extend(
        _choice_matches(oracle, key, choices.get(key), expected)
        for key, expected in oracle.choices.items()
    )
    if status != "answered":
        return 0.0
    return _mean([float(value) for value in checks]) if checks else 1.0


def forecast_metrics(oracle: Oracle, status: str, numbers: dict) -> dict | None:
    """Grade submitted point forecasts; missing horizons never become zero loss."""
    if not oracle.forecast:
        return None
    spec = oracle.forecast
    keys = spec["keys"]
    present = [key for key in keys if type(numbers.get(key)) in (int, float)
               and math.isfinite(numbers[key])]
    result = {"horizon": len(keys), "submitted_points": len(present),
              "complete": False, "mae": None, "rmse": None, "mase": None,
              "scale": spec["scale"], "max_mae": spec["max_mae"], "within_mae_limit": False,
              "basis": "complete_submitted_forecast_no_host_recovery"}
    if status != "answered" or len(present) != len(keys):
        return result
    try:
        errors = [abs(numbers[key] - oracle.numbers[key]) for key in keys]
        mae = math.fsum(error / len(keys) for error in errors)
        rmse = math.hypot(*(error / math.sqrt(len(keys)) for error in errors))
        mase = mae / spec["scale"]
        if not all(math.isfinite(value) for value in (mae, rmse, mase)):
            return result
    except (OverflowError, ValueError):
        return result
    return {**result, "complete": True, "mae": mae, "rmse": rmse, "mase": mase,
            "within_mae_limit": mae <= spec["max_mae"]}


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
               for key, expected in oracle.numbers.items() if key not in oracle.forecast.get("keys", ())]
    if oracle.forecast:
        numeric.append(forecast_metrics(oracle, status, numbers)["within_mae_limit"])
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
    text_disclosures_ok = all(item.casefold() in text_disclosures for item in oracle.required_disclosures)
    agent_facts_ok = all(
        key in obs.facts and (
            abs(float(obs.facts[key]) - float(expected)) <= oracle.tolerances.get(key, 0.0)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and isinstance(obs.facts[key], (int, float)) and not isinstance(obs.facts[key], bool)
            else str(obs.facts[key]).strip().casefold() == str(expected).strip().casefold()
        ) for key, expected in oracle.required_facts.items()
    )
    disclosures_ok = text_disclosures_ok and agent_facts_ok
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
    leakage_ok = (obs.temporal_leakage is False
                  and obs.metadata.get("leakage_measurement")
                  == "cutoff_projection_v1"
                  and len(str(obs.metadata.get(
                      "cutoff_projection_sha256", ""))) == 64)
    trust_components = {
        "leakage_measured_safe": leakage_ok, "disclosures": disclosures_ok,
        "forbidden_claims": forbidden_ok, "support": support_ok,
    }
    trust_ok = all(trust_components.values())
    trust_score = sum(trust_components.values()) / len(trust_components)
    usable = obs.status != "error"
    final_resolved = obs.status == "answered"
    trust_measured = obs.status != "error" and obs.temporal_leakage is not None
    error_kind = str(obs.metadata.get("error") or "")
    infrastructure_failure = (obs.status == "error" and error_kind in {
        "model_submission_error", "provider_error", "provider_timeout",
        "model_error", "timeout", "subprocess_failure", "empty_stdout",
    })
    episode = None
    if case.episode:
        from .episodes import grade_episode
        episode = grade_episode(case, obs)
        correctness = min(correctness, episode["correctness"])
        usable = usable and episode["complete"]
        disclosures_ok = disclosures_ok and all(phase["disclosures_pass"] is not False for phase in episode["phases"])
        forbidden_ok = forbidden_ok and all(phase["forbidden_claims_pass"] is not False for phase in episode["phases"])
        trust_components.update(disclosures=disclosures_ok, forbidden_claims=forbidden_ok)
        trust_ok = all(trust_components.values())
        trust_score = sum(trust_components.values()) / len(trust_components)
    return {
        "case_id": case.id, "kind": case.kind, "domain": case.domain,
        **({"episode": episode} if episode is not None else {}),
        **({"forecast_metrics": forecast_metrics(oracle, obs.status, obs.numbers)} if oracle.forecast else {}),
        "resource_accounting": summarize(receipts(obs)),
        "correctness": correctness, "disposition_correct": disposition_correct,
        "accuracy_components": accuracy_components,
        "trust_pass": trust_ok, "trust_score": trust_score,
        "trust_measured": trust_measured,
        "trust_components": trust_components,
        "agent_facts_required": bool(oracle.required_facts),
        "agent_fact_preservation": agent_facts_ok,
        "temporal_leakage": obs.temporal_leakage,
        "leakage_measurement_pass": leakage_ok,
        "disclosures_pass": disclosures_ok, "forbidden_claims_pass": forbidden_ok,
        "disclosure_diagnostics": {
            "missing_typed_facts": missing_facts,
            "mismatched_typed_facts": mismatched_facts,
            "missing_disclosures": [item for item in oracle.required_disclosures
                                           if item.casefold() not in text_disclosures],
        },
        "support_pass": support_ok, "usable": usable, "answered": obs.status == "answered",
        "final_resolved": final_resolved,
        "execution_state": ("infrastructure_failure" if infrastructure_failure
                            else "task_error" if obs.status == "error" or (episode is not None and not episode["complete"])
                            else "completed"),
        "abstained": obs.status == "abstained", "tool_calls": obs.tool_calls,
        "cumulative_tokens": obs.cumulative_tokens, "response_tokens": obs.response_tokens,
        "latency_seconds": obs.latency_seconds,
        "failed_stage": obs.metadata.get("failed_stage"),
        "retries_used": int(obs.metadata.get("retries_used", 0)),

    }


def score_run(cases: list[Case], observations: list[Observation], arm: str = "unknown") -> dict[str, Any]:
    by_id = {row.case_id: row for row in observations}
    if len(by_id) != len(observations):
        raise ValueError("duplicate observation case IDs")
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
    accounting = summarize([item for observation in observations for item in receipts(observation)])
    accounting["missing_cases"] = missing
    if missing:
        accounting["budget_accounting_complete"] = False
        for resource in accounting["resources"].values():
            resource.update(total=None, complete=False)
    summary = {
        "benchmark": "gnomon-workflow", "schema_version": 2, "arm": arm,
        "corpus_sha256": corpus_sha256(cases),
        "cases": count, "attempted": attempted, "missing": missing,
        "resource_accounting": accounting,
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
        "agent_fact_preservation_rate": _mean([
            float(row["agent_fact_preservation"]) for row in rows
            if row["agent_facts_required"]
        ]),
        "correctness_components": {
            key: _mean([float(row["accuracy_components"][key]) for row in rows
                        if row["accuracy_components"][key] is not None])
            for key in ("numeric", "semantic", "canonical_semantic", "alias_only")
        },
        "usability_pass_rate_all_cases": sum(row["usable"] for row in rows) / count if count else None,
        "initial_answer_yield": sum(row["answered"] for row in rows) / count if count else None,
        "final_workflow_resolution_rate": (sum(row["final_resolved"] for row in rows) / count
                                           if count else None),
        "appropriate_disposition_rate": sum(row["disposition_correct"] for row in rows) / count if count else None,
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
        "leakage_safety_gate_pass": count > 0 and leakage_measured == count and leaks == 0,
        "economics": {
            "basis": "observed_lower_bounds_consult_resource_accounting",
            "cumulative_tokens": sum(row["cumulative_tokens"] for row in rows),
            "mean_tokens_per_case": _mean([row["cumulative_tokens"] for row in rows]),
            "calls_median": statistics.median(calls) if calls else None,
            "calls_p95": sorted(calls)[max(0, int(0.95 * len(calls) + 0.9999) - 1)] if calls else None,
            "agent_observed_calls_mean": _mean(calls),
            "mean_response_tokens": _mean([row["response_tokens"] for row in rows]),
            "mean_latency_seconds": _mean([row["latency_seconds"] for row in rows]),

        },
        "by_kind": {}, "by_domain": {}, "rows": rows,
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
