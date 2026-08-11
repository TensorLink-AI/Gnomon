"""Run one DecisionBench condition and write the comparable artifacts.

Per run: ``gnomonbench.jsonl`` (one row per task, the schema
``gnomon eval compare`` consumes), ``summary.json`` (aggregates overall
and per family, with the harm cases listed individually), ``details/``
(per-task submission, grade, and transcript), ``tasks.jsonl`` (the full
task dump, truth included, for auditing), and ``manifest.json``
(provenance; ``benchmarks/report.py`` refuses cross-arm comparisons
whose manifests disagree).

Aggregation rules, stated once:

- Unanswered rows (no usable submission) count as failures in the
  success rate and are excluded from cost, regret, and calibration
  means — an answer the arm never produced is not a cheap answer.
- ``mean_cost`` and ``mean_regret_vs_naive`` are over answered rows.
- Harm cases are listed by task id, never only counted.
- The leakage flag is a heuristic (see ``grading.py``) and is reported
  under that name; LeakTrap owns the measured leakage claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.manifest import write_manifest  # noqa: E402
from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402
from benchmarks.common.records import RecordWriter, RunRecord  # noqa: E402
from benchmarks.decisionbench import arms, grading, tasks  # noqa: E402

CONDITIONS = ("llm-direct", "llm-sandbox", "gnomon-mcp")


def run_task(condition: str, task: tasks.DecisionTask, client: Any,
             work_dir: str | None = None) -> dict[str, Any]:
    if condition == "llm-direct":
        return arms.run_direct(task, client)
    if condition == "llm-sandbox":
        return arms.SandboxArm(task, client, work_dir=work_dir).drive()
    if condition == "gnomon-mcp":
        from benchmarks.decisionbench.mcp_arm import McpArm

        return McpArm(task, client, work_dir=work_dir).drive()
    raise ValueError(f"unknown condition {condition!r}")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _aggregate(grades: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [grade for grade in grades if grade["answered"]]
    committed = [grade for grade in answered if not grade["escalated"]]
    with_interval = [grade for grade in answered
                     if grade["peak_in_interval"] is not None]
    return {
        "tasks": len(grades),
        "answered": len(answered),
        "escalated": sum(grade["escalated"] for grade in answered),
        "success_rate": _mean([float(grade["success"]) for grade in grades]),
        "mean_cost": _mean([grade["cost"] for grade in answered]),
        "mean_regret_vs_naive": _mean([grade["regret_vs_naive"]
                                       for grade in answered]),
        "harm_cases": sum(grade["harm_case"] for grade in answered),
        "beat_naive_rate": _mean([
            float(grade["cost"] <= grade["cost_naive"] + 1e-9)
            for grade in answered]),
        "interval_coverage": _mean([float(grade["peak_in_interval"])
                                    for grade in with_interval]),
        "interval_count": len(with_interval),
        "appropriate_abstentions": sum(grade["appropriate_abstention"]
                                       for grade in answered),
        "leakage_heuristic_flags": sum(
            grade["temporal_leakage_heuristic"] for grade in committed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--model", required=True,
                        help="model id at the resolved endpoint")
    parser.add_argument("--count", type=int, default=60,
                        help="tasks to generate (default 60)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N generated tasks "
                             "(smoke runs; not comparable)")
    parser.add_argument("--families", default=None,
                        help="comma list filter, e.g. clean,revision "
                             "(smoke runs; not comparable)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    details_dir = output_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    suite = tasks.generate_tasks(args.count, seed=args.seed)
    families = ([name.strip() for name in args.families.split(",")]
                if args.families else None)
    if families:
        unknown = set(families) - {name for name, _ in tasks.FAMILY_WEIGHTS}
        if unknown:
            parser.error(f"unknown families: {sorted(unknown)}")
        suite = [task for task in suite if task.family in families]
    if args.limit is not None:
        suite = suite[:args.limit]
    if not suite:
        parser.error("no tasks selected")

    client = OpenRouterClient(args.model, temperature=args.temperature)
    writer = RecordWriter(output_dir / "gnomonbench.jsonl")
    grades: list[dict[str, Any]] = []

    for task in suite:
        started = time.monotonic()
        cost_before = client.total_cost_usd
        try:
            outcome = run_task(args.condition, task, client)
        except Exception as error:  # a dead row is disclosed, never invented
            outcome = {"submission": None,
                       "abstained": f"harness error: {error}",
                       "trace": [], "tool_calls": 0, "run_tokens": 0}
        latency = time.monotonic() - started
        grade = grading.grade(task, outcome["submission"])
        grades.append(grade)

        detail = {
            "task": tasks.task_public_record(task),
            "condition": args.condition,
            "submission": outcome["submission"],
            "abstained": outcome["abstained"],
            "grade": grade,
            "trace": outcome["trace"],
            "run_tokens": outcome["run_tokens"],
        }
        (details_dir / f"{task.task_id}.json").write_text(
            json.dumps(detail, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")

        writer.write(RunRecord(
            task_id=task.task_id,
            success=grade["success"],
            temporal_leakage=grade["temporal_leakage_heuristic"],
            appropriate_abstention=grade["appropriate_abstention"],
            tool_calls=outcome["tool_calls"],
            latency_seconds=round(latency, 3),
            cost_usd=round(client.total_cost_usd - cost_before, 6),
            extra={
                "family": task.family,
                "answered": grade["answered"],
                "escalated": grade["escalated"],
                "decision_cost": grade["cost"],
                "cost_naive": grade["cost_naive"],
                "cost_optimal": grade["cost_optimal"],
                "regret_vs_naive": grade["regret_vs_naive"],
                "harm_case": grade["harm_case"],
                "peak_in_interval": grade["peak_in_interval"],
                "followed_context": grade["followed_context"],
                "run_tokens": outcome["run_tokens"],
            },
        ))
        print(f"{task.task_id} [{task.family}] "
              f"{'ok' if grade['answered'] else 'UNANSWERED'} "
              f"cost={grade['cost']} naive={grade['cost_naive']} "
              f"optimal={grade['cost_optimal']}")

    # Written only after every arm conversation has ended: the sandbox
    # arm's interpreter is not path-jailed, so the truth dump must not
    # exist anywhere readable while a run is live. (The generator being
    # seeded and public means truth is always re-derivable in principle;
    # this keeps it out of trivial reach, and the README discloses the
    # rest.)
    tasks.dump_tasks(suite, output_dir / "tasks.jsonl")

    harm_ids = [
        {"task_id": task.task_id, "family": task.family,
         "cost": grade["cost"], "cost_naive": grade["cost_naive"]}
        for task, grade in zip(suite, grades) if grade["harm_case"]
    ]
    families_present = sorted({task.family for task in suite})
    summary = {
        "benchmark": "decisionbench",
        "condition": args.condition,
        "overall": _aggregate(grades),
        "per_family": {
            family: _aggregate([grade for task, grade in zip(suite, grades)
                                if task.family == family])
            for family in families_present
        },
        "harm_cases": harm_ids,
        "llm_usage": client.usage_summary,
        "leakage_note": ("temporal_leakage is the grading heuristic "
                         "described in grading.py, not LeakTrap's measured "
                         "claim"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_manifest(
        output_dir,
        benchmark="decisionbench",
        target=f"capacity-seed{args.seed}-count{args.count}"
               + (f"-families:{args.families}" if args.families else "")
               + (f"-limit:{args.limit}" if args.limit is not None else ""),
        condition=args.condition,
        model=args.model,
        base_url=client.usage_summary["base_url"],
        temperature=args.temperature,
        seed=args.seed,
        count=args.count,
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
