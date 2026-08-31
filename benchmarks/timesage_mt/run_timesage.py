"""Run TimeSage-MT dialogues against an agent and score them with the
dataset-embedded verify specs.

Examples
--------
Download the official dataset once::

    python -m benchmarks.timesage_mt.run_timesage --download \
        --data-dir ~/timesage-mt

Control (direct answering) vs. treatment (Gnomon tool loop), same model::

    python -m benchmarks.timesage_mt.run_timesage \
        --data-dir ~/timesage-mt --condition direct \
        --model openai/gpt-4o --tiers L1,L2 \
        --output-dir results/timesage-direct

    python -m benchmarks.timesage_mt.run_timesage \
        --data-dir ~/timesage-mt --condition gnomon-tools \
        --model openai/gpt-4o --tiers L1,L2 \
        --output-dir results/timesage-gnomon

    gnomon eval compare \
        --baseline results/timesage-direct/gnomonbench.jsonl \
        --treatment results/timesage-gnomon/gnomonbench.jsonl

Scoring: any spec that lists keywords or a checkable numerical range is
graded mechanically on those parts alone (see ``scoring.py`` for the
precise rule and its disclosed leniency). Turns whose spec has neither
are counted as unscored unless ``--judge-model`` is given, and
judge-based passes are reported separately — the official platform's
judge is not public, so those numbers are not leaderboard-comparable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest  # noqa: E402
from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402
from benchmarks.common.records import RecordWriter, RunRecord  # noqa: E402
from benchmarks.timesage_mt import harness, scoring  # noqa: E402
from benchmarks.timesage_mt.tasks import (  # noqa: E402
    TIERS,
    TimeSageTask,
    download,
    load_tasks,
)


def task_request_identity(
    task: TimeSageTask, *, condition: str, model: str, base_url: str,
    temperature: float, timeout: float, revision: str,
) -> str:
    """Fingerprint every input that can change a resumed model response."""
    visible = (task.visible_csv.read_bytes() if task.visible_csv
               and task.visible_csv.is_file() else b"")
    corpus = hashlib.sha256(
        json.dumps(task.raw, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8") + b"\0" + visible
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "task_id": task.task_id,
        "tier": task.tier,
        "corpus_sha256": corpus,
        "condition": condition,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "timeout_seconds": timeout,
        "code_revision": revision,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def scorable_turns_on_failure(
    task: TimeSageTask, *, judge_available: bool
) -> list[tuple[str, str]]:
    """What a crashed dialogue still owes the score books.

    One ``(turn_key, bucket)`` per reference turn carrying a
    ``finding_verify`` spec, where ``turn_key`` is the same
    ``{task_id}-turn{n}`` a scored run would emit — so failed tasks stay
    in pass-rate denominators and join against the other arm per turn —
    and ``bucket`` is ``mechanical`` (spec has a mechanically checkable
    part), ``judge`` (needs a judge, one is supplied) or ``unscored``
    (needs a judge, none supplied).
    """
    owed: list[tuple[str, str]] = []
    for user_turn in task.user_turns:
        turn_id = user_turn.get("turn_id")
        reference = task.reference_turn_after(turn_id or 0)
        verify = (reference or {}).get("finding_verify")
        if not verify:
            continue
        mechanical = scoring.score_mechanical(verify, "") is not None
        bucket = "mechanical" if mechanical else (
            "judge" if judge_available else "unscored")
        owed.append((f"{task.task_id}-turn{turn_id}", bucket))
    return owed


def _sum_usage(summaries: list[dict]) -> dict:
    """Combine independent client counters without losing provenance."""
    if not summaries:
        return {}
    models = {item.get("model") for item in summaries}
    endpoints = {item.get("base_url") for item in summaries}
    return {
        "model": next(iter(models)) if len(models) == 1 else "mixed",
        "base_url": next(iter(endpoints)) if len(endpoints) == 1 else "mixed",
        "requests": sum(int(item.get("requests", 0)) for item in summaries),
        "transport_attempts": sum(
            int(item.get("transport_attempts", 0)) for item in summaries
        ),
        "prompt_tokens": sum(
            int(item.get("prompt_tokens", 0)) for item in summaries
        ),
        "completion_tokens": sum(
            int(item.get("completion_tokens", 0)) for item in summaries
        ),
        "cost_usd": round(sum(
            float(item.get("cost_usd", 0.0)) for item in summaries
        ), 6),
        "truncation_escalations": sum(
            int(item.get("truncation_escalations", 0)) for item in summaries
        ),
    }


class _TaskRunError(RuntimeError):
    """A failed dialogue that retains the API work already consumed."""

    def __init__(self, error: Exception, *, usage: dict, elapsed: float) -> None:
        super().__init__(str(error))
        self.usage = usage
        self.elapsed = elapsed


def _run_task(task: TimeSageTask, *, model: str, temperature: float,
              timeout: float, condition: str) -> tuple[list[dict], dict, float]:
    """Run one independent dialogue with task-local accounting."""
    client = OpenRouterClient(model, temperature=temperature, timeout=timeout)
    started = time.time()
    try:
        turns = harness.run_dialogue(task, client, condition=condition)
    except Exception as error:
        raise _TaskRunError(
            error, usage=client.usage_summary,
            elapsed=time.time() - started,
        ) from error
    return turns, client.usage_summary, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", required=True,
                        help="Official dataset snapshot directory")
    parser.add_argument("--download", action="store_true",
                        help="Fetch the dataset into --data-dir first")
    parser.add_argument("--condition", choices=["direct", "gnomon-tools"],
                        default=None, help="Agent condition to run")
    parser.add_argument("--model", default=None, help="OpenRouter model id")
    parser.add_argument("--judge-model", default=None,
                        help="Optional OpenRouter judge for non-mechanical specs")
    parser.add_argument("--tiers", default="L1,L2,L3,L4")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max tasks (after tier filtering)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Per-request timeout in seconds (default: 180)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent task dialogues to run concurrently")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse complete, provenance-bearing transcripts")
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
    # Capture before any API work. A long run must not claim a checkout that
    # happened to become HEAD while its already-imported code was executing.
    run_revision = code_revision()

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip() in TIERS)
    tasks = load_tasks(data_dir, tiers=tiers or TIERS, limit=args.limit)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    # This client performs no requests. It resolves credentials and endpoint
    # provenance once; each worker gets an independent accounting client.
    client = OpenRouterClient(args.model, temperature=args.temperature,
                              timeout=args.timeout)
    judge = OpenRouterClient(args.judge_model, temperature=0.0) \
        if args.judge_model else None

    output_dir = Path(args.output_dir)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "gnomonbench.jsonl"
    # RecordWriter appends; a rerun into the same output dir must replace
    # the previous run's rows (as summary.json is), not accumulate them.
    records_path.unlink(missing_ok=True)
    records = RecordWriter(records_path)

    counts = {"mechanical_pass": 0, "mechanical_fail": 0,
              "judge_pass": 0, "judge_fail": 0, "unscored": 0}
    robust_mechanical: list[bool] = []
    per_tier: dict[str, list[bool]] = {}
    rows: list[list] = []
    tasks_failed = 0
    tasks_csv_truncated = 0
    task_results: dict[str, tuple[list[dict], dict, float] | Exception] = {}
    pending: list[TimeSageTask] = []
    for task in tasks:
        request_identity = task_request_identity(
            task, condition=args.condition, model=args.model,
            base_url=client.base_url, temperature=args.temperature,
            timeout=args.timeout, revision=run_revision)
        transcript_path = transcripts_dir / f"{task.task_id}.json"
        if args.resume and transcript_path.exists():
            try:
                saved = json.loads(transcript_path.read_text(encoding="utf-8"))
                if (saved.get("task_id") == task.task_id
                        and saved.get("condition") == args.condition
                        and saved.get("model") == args.model
                        and saved.get("code_revision") == run_revision
                        and saved.get("request_identity_sha256")
                        == request_identity
                        and isinstance(saved.get("llm_usage"), dict)
                        and len(saved.get("turns", [])) == len(task.user_turns)):
                    task_results[task.task_id] = (
                        saved["turns"], saved["llm_usage"],
                        float(saved.get("elapsed_seconds", 0.0)),
                    )
                    print(f"[resume] {task.task_id}")
                    continue
            except (OSError, ValueError, TypeError):
                pass
        pending.append(task)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _run_task, task, model=args.model,
                temperature=args.temperature, timeout=args.timeout,
                condition=args.condition,
            ): task
            for task in pending
        }
        completed = len(task_results)
        for future in as_completed(futures):
            task = futures[future]
            completed += 1
            try:
                turn_records, usage, elapsed = future.result()
                task_results[task.task_id] = (turn_records, usage, elapsed)
                # The transcript is the durable resume unit. Write it as soon
                # as a dialogue completes so an interrupted matrix loses no
                # successful API work.
                checkpoint = {
                    "task_id": task.task_id, "tier": task.tier,
                    "condition": args.condition, "model": args.model,
                    "code_revision": run_revision,
                    "request_identity_sha256": task_request_identity(
                        task, condition=args.condition, model=args.model,
                        base_url=client.base_url,
                        temperature=args.temperature, timeout=args.timeout,
                        revision=run_revision),
                    "elapsed_seconds": elapsed, "llm_usage": usage,
                    "turns": turn_records,
                }
                (transcripts_dir / f"{task.task_id}.json").write_text(
                    json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
                )
                print(f"[{completed}/{len(tasks)}] {task.task_id} complete")
            except Exception as error:
                task_results[task.task_id] = error
                print(f"[{completed}/{len(tasks)}] {task.task_id} failed: {error}")

    usage_summaries: list[dict] = []
    for task in tasks:
        result = task_results[task.task_id]
        if isinstance(result, Exception):
            error = result
            if isinstance(error, _TaskRunError):
                usage_summaries.append(error.usage)
            print(f"  task failed: {error}")
            message = str(error)[:500]
            tasks_failed += 1
            # One task-level error line, keyed off the turn-id namespace
            # so it can never collide with a scored turn.
            records.write(RunRecord(
                task_id=f"{task.task_id}-error", success=False,
                extra={"error": message, "tier": task.tier},
            ))
            # Every turn the task would have scored still counts: a
            # failed record per reference-scored turn, keyed exactly
            # like a scored one, in the pass-rate denominator.
            for turn_key, bucket in scorable_turns_on_failure(
                task, judge_available=judge is not None
            ):
                if bucket == "unscored":
                    counts["unscored"] += 1
                    continue
                counts[f"{bucket}_fail"] += 1
                per_tier.setdefault(task.tier, []).append(False)
                rows.append([task.task_id, task.tier,
                             turn_key.rsplit("-turn", 1)[1],
                             "task_error", False, 0])
                records.write(RunRecord(
                    task_id=turn_key, success=False,
                    extra={"basis": "task_error", "tier": task.tier,
                           "error": message},
                ))
            continue
        turn_records, usage, elapsed = result
        usage_summaries.append(usage)
        if turn_records and turn_records[0].get("csv_truncated"):
            tasks_csv_truncated += 1

        task_transcript = {
            "task_id": task.task_id, "tier": task.tier,
            "condition": args.condition, "model": args.model,
            "code_revision": run_revision,
            "request_identity_sha256": task_request_identity(
                task, condition=args.condition, model=args.model,
                base_url=client.base_url, temperature=args.temperature,
                timeout=args.timeout, revision=run_revision),
            "elapsed_seconds": elapsed, "llm_usage": usage, "turns": [],
        }
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
                if verdict["basis"] == "mechanical" \
                        and verdict.get("numerosity_robust"):
                    robust_mechanical.append(bool(verdict["passed"]))
                rows.append([task.task_id, task.tier, record["user_turn_id"],
                             verdict["basis"], verdict["passed"],
                             len(record["tool_calls"])])
                records.write(RunRecord(
                    task_id=turn_key, success=bool(verdict["passed"]),
                    tool_calls=len(record["tool_calls"]),
                    latency_seconds=round(elapsed / max(len(turn_records), 1), 3),
                    extra={"basis": verdict["basis"], "tier": task.tier,
                           "csv_truncated": bool(record.get("csv_truncated"))},
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
        "code_revision": run_revision,
        "judge_model": args.judge_model,
        "tiers": list(tiers or TIERS),
        "tasks": len(tasks),
        "tasks_failed": tasks_failed,
        "tasks_csv_truncated": tasks_csv_truncated,
        "mechanical_turns": mechanical_total,
        "mechanical_pass_rate": (counts["mechanical_pass"] / mechanical_total
                                 if mechanical_total else None),
        "numerosity_robust_mechanical_turns": len(robust_mechanical),
        "numerosity_robust_mechanical_pass_rate": (
            sum(robust_mechanical) / len(robust_mechanical)
            if robust_mechanical else None),
        "judge_turns": judge_total,
        "judge_pass_rate": (counts["judge_pass"] / judge_total
                            if judge_total else None),
        "unscored_turns": counts["unscored"],
        "per_tier_pass_rate": {
            tier: sum(flags) / len(flags)
            for tier, flags in sorted(per_tier.items())
        },
        "llm_usage": _sum_usage(usage_summaries),
        "note": (
            "mechanical scores apply the checkable parts of the official "
            "finding_verify specs (see scoring.py for the precise rule); "
            "a failed task counts every reference-scored turn as a "
            "failure; judge scores use a local judge and are not "
            "comparable to the official leaderboard"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # Provenance beside the results on direct CLI runs too: without it a
    # comparison of two timesage arms could only ever be "assumed
    # comparable" — report.py had no manifest to check.
    write_manifest(
        output_dir,
        benchmark="timesage-mt",
        condition=args.condition,
        model=args.model,
        target="tiers=" + ",".join(tiers or TIERS),
        command=" ".join(sys.argv),
        limit=args.limit,
        judge_model=args.judge_model,
        base_url=client.base_url,
        code_revision=run_revision,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
