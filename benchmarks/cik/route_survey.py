"""Dry-run the routing decision across CiK tasks — no forecasts.

One completion per task instance, pennies for the whole grid. The
question it answers: does the routing prompt actually discriminate, or
is it a constant function? The first routing prompt measured 68/68
"gnomon" — a constant — and 58 of those runs came out bit-identical to
the agent arm, so the router arm was spend without information. This
survey is the gate: do not run `--method gnomon-router` until the
survey shows a non-degenerate route distribution, and read the sampled
rationales to check the discrimination is for sane reasons.

Usage (needs the cik_benchmark checkout and OPENROUTER_API_KEY):
    python -m benchmarks.cik.route_survey --model <id> \
        [--seeds 1] [--task-filter STR] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def tally(routes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-instance route decisions into the survey summary."""
    counts = Counter(entry["route"] for entry in routes)
    by_task: dict[str, Counter] = defaultdict(Counter)
    for entry in routes:
        by_task[entry["task"]][entry["route"]] += 1
    split_tasks = sorted(
        task for task, task_counts in by_task.items() if len(task_counts) > 1
    )
    total = len(routes) or 1
    dominant_share = max(counts.values()) / total if counts else 0.0
    examples: dict[str, list[dict[str, str]]] = {}
    for entry in routes:
        bucket = examples.setdefault(entry["route"], [])
        if len(bucket) < 5:
            bucket.append({"task": entry["task"],
                           "why": entry.get("why", "")})
    return {
        "instances": len(routes),
        "routes": dict(counts),
        "dominant_share": round(dominant_share, 4),
        "degenerate": dominant_share >= 0.95,
        "tasks_with_split_decisions": split_tasks,
        "per_task": {task: dict(c) for task, c in sorted(by_task.items())},
        "rationale_samples": examples,
        "reading": (
            "degenerate=true means the prompt is a constant function and "
            "the router arm will measure nothing the agent arm does not; "
            "fix the prompt before spending forecasts on it."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Instances per task to route (default 1)")
    parser.add_argument("--task-filter", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    from cik_benchmark import ALL_TASKS

    from benchmarks.cik.router import RoutedForecaster

    router = RoutedForecaster(args.model, temperature=args.temperature)
    routes: list[dict[str, Any]] = []
    for task_class in ALL_TASKS:
        name = task_class.__name__
        if args.task_filter and args.task_filter not in name:
            continue
        for seed in range(1, args.seeds + 1):
            instance = task_class(seed=seed)
            route, why, note = router._route(instance)
            entry = {"task": name, "seed": seed, "route": route, "why": why}
            if note:
                entry["note"] = note
            routes.append(entry)
            print(f"{name} seed={seed}: {route}  ({why[:80]})")

    summary = tally(routes)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
