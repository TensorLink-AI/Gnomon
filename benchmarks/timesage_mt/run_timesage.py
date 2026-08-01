"""Run TimeSage-MT dialogues against an agent and score them with the
dataset-embedded verify specs.

Examples
--------
Download the official dataset once::

    python -m benchmarks.timesage_mt.run_timesage --download \
        --data-dir ~/timesage-mt

Control (direct answering) vs. treatment (Aion tool loop), same model::

    python -m benchmarks.timesage_mt.run_timesage \
        --data-dir ~/timesage-mt --condition direct \
        --model openai/gpt-4o --tiers L1,L2 \
        --output-dir results/timesage-direct

    python -m benchmarks.timesage_mt.run_timesage \
        --data-dir ~/timesage-mt --condition aion-tools \
        --model openai/gpt-4o --tiers L1,L2 \
        --output-dir results/timesage-aion

    aion eval compare \
        --baseline results/timesage-direct/aionbench.jsonl \
        --treatment results/timesage-aion/aionbench.jsonl

Scoring: mechanical checks (keyword / numerical_range) are applied
exactly as the task files specify. Turns whose spec needs an embedding
or judge are counted as unscored unless ``--judge-model`` is given, and
judge-based passes are reported separately — the official platform's
judge is not public, so those numbers are not leaderboard-comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402
from benchmarks.common.records import RecordWriter, RunRecord  # noqa: E402
from benchmarks.timesage_mt import harness, scoring  # noqa: E402
from benchmarks.timesage_mt.tasks import TIERS, download, load_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", required=True,
                        help="Official dataset snapshot directory")
    parser.add_argument("--download", action="store_true",
                        help="Fetch the dataset into --data-dir first")
    parser.add_argument("--condition", choices=["direct", "aion-tools"],
                        default=None, help="Agent condition to run")
    parser.add_argument("--model", default=None, help="OpenRouter model id")
    parser.add_argument("--judge-model", default=None,
                        help="Optional OpenRouter judge for non-mechanical specs")
    parser.add_argument("--tiers", default="L1,L2,L3,L4")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max tasks (after tier filtering)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.download:
        download(data_dir)
        print(f"Dataset ready under {data_dir}")
        if not args.condition:
            return 0
    if not args.condition or not args.model or not args.output_dir:
        parser.error("--condition, --model and --output-dir are required to run")

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip() in TIERS)
    tasks = load_tasks(data_dir, tiers=tiers or TIERS, limit=args.limit)
    client = OpenRouterClient(args.model, temperature=args.temperature)
    judge = OpenRouterClient(args.judge_model, temperature=0.0) \
        if args.judge_model else None

    output_dir = Path(args.output_dir)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    records = RecordWriter(output_dir / "aionbench.jsonl")

    counts = {"mechanical_pass": 0, "mechanical_fail": 0,
              "judge_pass": 0, "judge_fail": 0, "unscored": 0}
    per_tier: dict[str, list[bool]] = {}
    rows: list[list] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task.task_id}")
        started = time.time()
        try:
            turn_records = harness.run_dialogue(
                task, client, condition=args.condition
            )
        except Exception as error:
            print(f"  task failed: {error}")
            records.write(RunRecord(
                task_id=task.task_id, success=False,
                extra={"error": str(error)[:500]},
            ))
            continue
        elapsed = time.time() - started

        task_transcript = {"task_id": task.task_id, "tier": task.tier,
                           "condition": args.condition, "turns": []}
        for record in turn_records:
            reference = task.reference_turn_after(record["user_turn_id"] or 0)
            verdict = scoring.score_turn(reference or {}, record["response"],
                                         judge_client=judge)
            record["verdict"] = verdict
            task_transcript["turns"].append(record)
            turn_key = f"{task.task_id}-turn{record['user_turn_id']}"
            if verdict["scored"]:
                bucket = ("mechanical" if verdict["basis"] == "mechanical"
                          else "judge")
                counts[f"{bucket}_{'pass' if verdict['passed'] else 'fail'}"] += 1
                per_tier.setdefault(task.tier, []).append(bool(verdict["passed"]))
                rows.append([task.task_id, task.tier, record["user_turn_id"],
                             verdict["basis"], verdict["passed"],
                             len(record["tool_calls"])])
                records.write(RunRecord(
                    task_id=turn_key, success=bool(verdict["passed"]),
                    tool_calls=len(record["tool_calls"]),
                    latency_seconds=round(elapsed / max(len(turn_records), 1), 3),
                    extra={"basis": verdict["basis"], "tier": task.tier},
                ))
            else:
                counts["unscored"] += 1
        (transcripts_dir / f"{task.task_id}.json").write_text(
            json.dumps(task_transcript, indent=2) + "\n", encoding="utf-8"
        )

    with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "tier", "turn", "basis", "passed", "tool_calls"])
        writer.writerows(rows)

    mechanical_total = counts["mechanical_pass"] + counts["mechanical_fail"]
    judge_total = counts["judge_pass"] + counts["judge_fail"]
    summary = {
        "benchmark": "timesage-mt",
        "condition": args.condition,
        "model": args.model,
        "judge_model": args.judge_model,
        "tiers": list(tiers or TIERS),
        "tasks": len(tasks),
        "mechanical_turns": mechanical_total,
        "mechanical_pass_rate": (counts["mechanical_pass"] / mechanical_total
                                 if mechanical_total else None),
        "judge_turns": judge_total,
        "judge_pass_rate": (counts["judge_pass"] / judge_total
                            if judge_total else None),
        "unscored_turns": counts["unscored"],
        "per_tier_pass_rate": {
            tier: sum(flags) / len(flags)
            for tier, flags in sorted(per_tier.items())
        },
        "llm_usage": client.usage_summary,
        "note": (
            "mechanical scores follow the official finding_verify specs; "
            "judge scores use a local judge and are not comparable to the "
            "official leaderboard"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
