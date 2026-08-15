"""Compare matched Workflow Bench arms and apply the product decision policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any
import statistics


def _load(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json" if path.is_dir() else path
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    if value.get("benchmark") != "gnomon-workflow":
        raise ValueError(f"{summary_path}: not a Workflow Bench summary")
    return value


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def paired_interval(baseline: dict[str, Any], treatment: dict[str, Any],
                    *, seed: int = 7, samples: int = 5000) -> dict[str, float]:
    left = {row["case_id"]: row for row in baseline["rows"]}
    right = {row["case_id"]: row for row in treatment["rows"]}
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError("arms have no matched scored cases")
    deltas = [right[key]["correctness"] - left[key]["correctness"] for key in ids]
    rng = random.Random(seed)
    boot = [sum(rng.choice(deltas) for _ in ids) / len(ids) for _ in range(samples)]
    return {"matched_cases": len(ids), "delta": sum(deltas) / len(deltas),
            "ci95_low": _percentile(boot, 0.025),
            "ci95_high": _percentile(boot, 0.975)}


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Average stochastic replicates by case before comparing arms."""
    if not runs:
        raise ValueError("cannot aggregate zero runs")
    row_sets = [{row["case_id"] for row in run["rows"]} for run in runs]
    if any(ids != row_sets[0] for ids in row_sets[1:]):
        raise ValueError(f"replicates for arm {runs[0]['arm']!r} have different case ids")
    rows = []
    for case_id in sorted(row_sets[0]):
        copies = [next(row for row in run["rows"] if row["case_id"] == case_id)
                  for run in runs]
        base = dict(copies[0])
        for field in ("correctness", "tool_calls", "cumulative_tokens",
                      "response_tokens", "latency_seconds"):
            base[field] = sum(float(row[field]) for row in copies) / len(copies)
        for field in ("trust_pass", "usable", "answered", "final_resolved",
                      "disposition_correct"):
            base[field] = sum(bool(row[field]) for row in copies) / len(copies)
        stage_names = {name for row in copies
                       for name in row.get("stage_economics", {})}
        base["stage_economics"] = {}
        for stage in stage_names:
            samples = [row["stage_economics"][stage] for row in copies
                       if stage in row.get("stage_economics", {})]
            base["stage_economics"][stage] = {
                field: sum(float(sample[field]) for sample in samples) / len(samples)
                for field in ("tool_calls", "cumulative_tokens", "response_tokens",
                              "latency_seconds")
            }
        rows.append(base)
    economics = {
        "mean_tokens_per_case": sum(row["cumulative_tokens"] for row in rows) / len(rows),
        "calls_median": statistics.median(row["tool_calls"] for row in rows),
        "calls_p95": _percentile([row["tool_calls"] for row in rows], 0.95),
        "surface_required_calls_mean": statistics.mean(
            float(run.get("economics", {}).get("surface_required_calls_mean") or 0)
            for run in runs),
        "agent_observed_calls_mean": statistics.mean(
            float(run.get("economics", {}).get("agent_observed_calls_mean") or 0)
            for run in runs),
        "recovery_calls_mean": statistics.mean(
            float(run.get("economics", {}).get("recovery_calls_mean") or 0)
            for run in runs),
        "redundant_calls_mean": statistics.mean(
            float(run.get("economics", {}).get("redundant_calls_mean") or 0)
            for run in runs),
    }
    initial_calls = [row.get("stage_economics", {}).get("initial", {}).get(
        "tool_calls", row["tool_calls"]) for row in rows]
    economics["initial_calls_median"] = (
        statistics.median(initial_calls) if initial_calls else None)
    economics["initial_calls_p95"] = (
        _percentile(initial_calls, 0.95) if initial_calls else None)
    by_kind = {}
    for kind in sorted({row["kind"] for row in rows}):
        selected = [row for row in rows if row["kind"] == kind]
        by_kind[kind] = {
            "correctness": sum(row["correctness"] for row in selected) / len(selected),
            "trust": sum(row["trust_pass"] for row in selected) / len(selected),
            "usability": sum(row["usable"] for row in selected) / len(selected),
        }
    return {
        "arm": runs[0]["arm"], "replicates": len(runs), "rows": rows,
        "cases": len(rows),
        "correctness_mean_all_cases": sum(row["correctness"] for row in rows) / len(rows),
        "trust_pass_rate_all_cases": sum(row["trust_pass"] for row in rows) / len(rows),
        "usability_pass_rate_all_cases": sum(row["usable"] for row in rows) / len(rows),
        "answer_yield": sum(row["answered"] for row in rows) / len(rows),
        "final_workflow_resolution_rate": (
            sum(row.get("final_resolved", row["answered"]) for row in rows) / len(rows)),
        "release_gate_pass": all(run["release_gate_pass"] for run in runs),
        "completeness_gate_pass": all(run.get("completeness_gate_pass",
                                               run["release_gate_pass"])
                                      for run in runs),
        "leakage_safety_gate_pass": all(run.get("leakage_safety_gate_pass",
                                                 run["release_gate_pass"])
                                        for run in runs),
        "trust_component_pass_rates": {
            key: sum(float(run.get("trust_component_pass_rates", {}).get(key, 0.0))
                     for run in runs) / len(runs)
            for key in sorted({key for run in runs
                               for key in run.get("trust_component_pass_rates", {})})
        },
        "correctness_components": {
            key: statistics.mean(values) if values else None
            for key in ("numeric", "semantic", "canonical_semantic", "alias_only")
            if (values := [float(run["correctness_components"][key]) for run in runs
                           if run.get("correctness_components", {}).get(key) is not None])
        },
        "infrastructure_failure_rate": (
            sum(int(run.get("infrastructure_failures", 0)) for run in runs)
            / sum(int(run["cases"]) for run in runs)),
        "task_error_rate": (
            sum(int(run.get("task_errors", 0)) for run in runs)
            / sum(int(run["cases"]) for run in runs)),
        "engine_contract_completeness": statistics.mean([
            float(run["engine_contract"]["completeness_rate"])
            for run in runs
            if run.get("engine_contract", {}).get("completeness_rate") is not None
        ]) if any(run.get("engine_contract", {}).get("completeness_rate") is not None
                  for run in runs) else None,
        "agent_fact_preservation_rate": statistics.mean([
            float(run["agent_fact_preservation_rate"]) for run in runs
            if run.get("agent_fact_preservation_rate") is not None
        ]) if any(run.get("agent_fact_preservation_rate") is not None
                  for run in runs) else None,
        "artifact_identity_flow": {
            key: sum(int(run.get("artifact_identity_flow", {}).get(key, 0))
                     for run in runs)
            for key in ("required_cases", "identity_produced", "identity_matched",
                        "agent_preserved_identity")
        },
        "economics": economics, "by_kind": by_kind,
    }


def _accuracy_gate(comparison: dict[str, Any] | None, *, minimum_tasks: int = 20,
                   margin: float = 0.02) -> dict[str, Any]:
    """Validate a matched TemporalBench/report comparison conservatively."""
    if comparison and isinstance(comparison.get("comparisons"), list):
        rows = comparison["comparisons"]
        comparison = rows[0] if len(rows) == 1 else None
    if not comparison or comparison.get("comparable") is not True:
        return {"pass": False, "reason": "missing or incomparable accuracy comparison"}
    if comparison.get("benchmark") != "temporalbench":
        return {"pass": False, "reason": "accuracy comparison is not TemporalBench"}
    matched = int(comparison.get("matched_tasks") or 0)
    if matched < minimum_tasks:
        return {"pass": False, "reason": f"only {matched} matched tasks"}
    metrics = comparison.get("metrics") or {}
    preferred = next((name for name in ("SMAPE", "OW_sMAPE", "MAE", "RMSE")
                      if name in metrics), None)
    if preferred:
        entry = metrics[preferred]
        values = entry.get("penalized") or entry
        baseline, treatment = values.get("baseline_mean"), values.get("treatment_mean")
        if baseline is None or treatment is None:
            return {"pass": False, "reason": f"{preferred} has no paired means"}
        lower = entry.get("lower_is_better") is not False
        allowed = (float(treatment) <= float(baseline) * (1 + margin) if lower
                   else float(treatment) >= float(baseline) * (1 - margin))
        return {"pass": allowed, "matched_tasks": matched, "metric": preferred,
                "baseline": baseline, "treatment": treatment,
                "relative_margin": margin}
    rates = comparison.get("success_rate") or {}
    if rates.get("baseline") is not None and rates.get("treatment") is not None:
        allowed = float(rates["treatment"]) >= float(rates["baseline"]) - margin
        return {"pass": allowed, "matched_tasks": matched, "metric": "success_rate",
                "baseline": rates["baseline"], "treatment": rates["treatment"],
                "absolute_margin": margin}
    return {"pass": False, "reason": "no recognized matched accuracy metric"}
def compare(summaries: list[dict[str, Any]], baseline_arm: str,
            noninferiority_margin: float = 0.02,
            leaktrap: dict[str, Any] | None = None,
            accuracy_comparisons: dict[str, dict[str, Any]] | None = None,
            minimum_replicates: int = 3,
            heldout_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if minimum_replicates < 1:
        raise ValueError("minimum_replicates must be at least 1")
    hashes = {summary.get("corpus_sha256") for summary in summaries}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("arms do not share one known corpus_sha256")
    grouped: dict[str, list[dict[str, Any]]] = {}
    corpus_hash = next(iter(hashes))
    heldout_gate = bool(
        heldout_manifest
        and heldout_manifest.get("generator") == "workflow-synthetic-v2"
        and heldout_manifest.get("fresh_seed") is True
        and heldout_manifest.get("corpus_sha256") == corpus_hash
        and isinstance(heldout_manifest.get("seed"), int)
    )
    for summary in summaries:
        grouped.setdefault(summary["arm"], []).append(summary)
    by_arm = {name: _aggregate(runs) for name, runs in grouped.items()}
    if baseline_arm not in by_arm:
        raise ValueError(f"baseline arm {baseline_arm!r} not found")
    baseline = by_arm[baseline_arm]
    arms = []
    accuracy_comparisons = accuracy_comparisons or {}
    for name, summary in by_arm.items():
        economics = summary["economics"]
        interval = ({"matched_cases": summary["cases"], "delta": 0.0,
                     "ci95_low": 0.0, "ci95_high": 0.0}
                    if name == baseline_arm else paired_interval(baseline, summary))
        # Pre-committed surface objectives. Accuracy uses the conservative
        # lower paired confidence bound rather than the point estimate.
        gates = {
            "provider_complete": bool(summary["completeness_gate_pass"]),
            "workflow_leak_free": bool(summary["leakage_safety_gate_pass"]),
            "final_resolution_at_least_80pct": (
                summary["final_workflow_resolution_rate"] >= 0.8),
            "initial_median_calls_at_most_2": (
                economics["initial_calls_median"] is not None
                and economics["initial_calls_median"] <= 2),
            "workflow_median_calls_at_most_4": economics["calls_median"] <= 4,
            "mean_tokens_below_50k": economics["mean_tokens_per_case"] < 50_000,
            "accuracy_noninferior": interval["ci95_low"] >= -noninferiority_margin,
        }
        accuracy = (_accuracy_gate(accuracy_comparisons.get(name),
                                   margin=noninferiority_margin)
                    if name != baseline_arm else {"pass": True, "reason": "baseline"})
        arms.append({
            "arm": name, "correctness": summary["correctness_mean_all_cases"],
            "replicates": summary["replicates"],
            "trust": summary["trust_pass_rate_all_cases"],
            "usability": summary["usability_pass_rate_all_cases"],
            "initial_answer_yield": summary["answer_yield"],
            "final_workflow_resolution_rate": summary["final_workflow_resolution_rate"],
            "tokens_per_case": economics["mean_tokens_per_case"],
            "calls_median": economics["calls_median"], "accuracy_vs_baseline": interval,
            "calls_p95": economics["calls_p95"],
            "initial_calls_median": economics["initial_calls_median"],
            "initial_calls_p95": economics["initial_calls_p95"],
            "surface_required_calls_mean": economics["surface_required_calls_mean"],
            "agent_observed_calls_mean": economics["agent_observed_calls_mean"],
            "recovery_calls_mean": economics["recovery_calls_mean"],
            "redundant_calls_mean": economics["redundant_calls_mean"],
            "by_kind": summary["by_kind"],
            "trust_components": summary["trust_component_pass_rates"],
            "correctness_components": summary["correctness_components"],
            "engine_contract_completeness": summary["engine_contract_completeness"],
            "agent_fact_preservation_rate": summary["agent_fact_preservation_rate"],
            "artifact_identity_flow": summary["artifact_identity_flow"],
            "infrastructure_failure_rate": summary["infrastructure_failure_rate"],
            "task_error_rate": summary["task_error_rate"],
            "accuracy_benchmark": accuracy,
            "replicate_gate_pass": summary["replicates"] >= minimum_replicates,
            "gates": gates, "eligible": all(gates.values()),
        })
    safety_gate = bool(
        leaktrap and leaktrap.get("benchmark") == "leakage-trap"
        and leaktrap.get("condition") == "gnomon"
        and int(leaktrap.get("tasks") or 0) >= 40
        and leaktrap.get("tasks_flagged_as_leaking") == 0
        and leaktrap.get("tasks_transcribing_the_future") == 0
        and leaktrap.get("structural_claim_proven") == leaktrap.get("tasks")
    )
    eligible = [arm["arm"] for arm in arms if arm["eligible"]]
    arm_rows = {arm["arm"]: arm for arm in arms}
    baseline_replicated = arm_rows[baseline_arm]["replicate_gate_pass"]
    decision_ready = [name for name in eligible if safety_gate
                      and heldout_gate
                      and name != baseline_arm and baseline_replicated
                      and arm_rows[name]["replicate_gate_pass"]
                      and arm_rows[name]["accuracy_benchmark"]["pass"]]
    return {"benchmark": "gnomon-workflow-comparison", "corpus_sha256": hashes.pop(),
            "baseline_arm": baseline_arm, "noninferiority_margin": noninferiority_margin,
            "minimum_replicates": minimum_replicates,
            "arms": arms, "eligible_arms": eligible,
            "leaktrap_gate_pass": safety_gate,
            "fresh_heldout_gate_pass": heldout_gate,
            "decision_ready_arms": decision_ready}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run directories or summary.json files")
    parser.add_argument("--baseline", required=True, help="arm name used for paired accuracy")
    parser.add_argument("--noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument("--leaktrap-summary",
                        help="required for decision readiness: 40+ task Gnomon LeakTrap summary")
    parser.add_argument("--accuracy-comparison", action="append", default=[],
                        metavar="ARM=PATH", help="matched TemporalBench/report JSON for an arm")
    parser.add_argument("--minimum-replicates", type=int, default=3)
    parser.add_argument("--heldout-manifest",
                        help="fresh generator manifest required for decision readiness")
    args = parser.parse_args()
    leaktrap = (json.loads(Path(args.leaktrap_summary).read_text(encoding="utf-8"))
                if args.leaktrap_summary else None)
    accuracy = {}
    for item in args.accuracy_comparison:
        if "=" not in item:
            parser.error("--accuracy-comparison must be ARM=PATH")
        arm, path = item.split("=", 1)
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        # benchmarks.report writes a report envelope so one file can retain
        # several requested comparisons plus costs. A workflow decision arm
        # consumes exactly one matched comparison from that envelope.
        comparisons = loaded.get("comparisons") if isinstance(loaded, dict) else None
        if isinstance(comparisons, list):
            if len(comparisons) != 1:
                parser.error("--accuracy-comparison report must contain exactly one comparison")
            loaded = comparisons[0]
        accuracy[arm] = loaded
    result = compare([_load(Path(path)) for path in args.runs], args.baseline,
                     args.noninferiority_margin, leaktrap, accuracy,
                     args.minimum_replicates,
                     (json.loads(Path(args.heldout_manifest).read_text(encoding="utf-8"))
                      if args.heldout_manifest else None))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        from .provenance import atomic_write_text
        atomic_write_text(Path(args.output), text)
    print(text, end="")
    return 0 if result["decision_ready_arms"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
