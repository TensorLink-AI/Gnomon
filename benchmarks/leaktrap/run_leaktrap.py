"""Run the temporal-leakage trap family.

Conditions
----------
``oracle-leak``  Deliberately forecasts from the fully revised series,
                 including rows published after the cutoff. It exists to
                 **validate the trap**: if the oracle does not beat the
                 no-leak ceiling, then leaking does not help on these tasks
                 and every other number in the family is meaningless. Run it
                 first; it is free and needs no model.
``aion``         Ingests the same CSV with its publication dates and
                 forecasts through the snapshot at ``--as-of <cutoff>``.
                 Post-cutoff reads are structurally impossible, and the
                 grader asserts it over the run's own ``snapshot_access``
                 log rather than trusting the condition's name.
``control``      An OpenRouter model receives the same CSV — publication
                 dates and all — plus the cutoff, and is asked for the
                 forecast. Measures whether a frontier model actually
                 respects the dates when reading past them would score
                 better.

Examples
--------
::

    python -m benchmarks.leaktrap.run_leaktrap oracle-leak --limit 40 \
        --output-dir results/leaktrap-oracle

    python -m benchmarks.leaktrap.run_leaktrap aion --limit 40 \
        --output-dir results/leaktrap-aion

    python -m benchmarks.leaktrap.run_leaktrap control --limit 40 \
        --model z-ai/glm-5.2 --output-dir results/leaktrap-control

    python -m benchmarks.report --baseline results/leaktrap-control \
        --treatment results/leaktrap-aion --metric score
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.manifest import write_manifest  # noqa: E402
from benchmarks.leaktrap.grade import (  # noqa: E402
    leak_verdict,
    no_leak_ceiling,
    structural_assertion,
    wape,
)
from benchmarks.leaktrap.tasks import TrapTask, generate_tasks  # noqa: E402

CONDITIONS = ("oracle-leak", "aion", "control")

CONTROL_PROMPT = """You are forecasting a daily time series.

The data below is an extract from a system that publishes revisions. Each row
has three fields: the date the value applies to (`timestamp`), the value, and
the date that value was published (`published`). A value is only knowable on
or after its publication date.

You are standing at {cutoff}. Forecast the next {horizon} daily values,
starting {first_forecast_date}.

Data:
{data}

Reply with JSON only, in the form:
{{"forecast": [v1, v2, ..., v{horizon}]}}
"""


def _oracle_forecast(task: TrapTask) -> list[float]:
    """Forecast from everything in the file, publication dates ignored.

    Deliberately the cheating condition. It uses the *revised* pre-cutoff
    history and, in the same spirit, the post-cutoff rows that reveal the
    shock — which is exactly the read a careless consumer of a revision-
    carrying extract makes.
    """
    from aion.models import MODELS, predict

    revised = task.revised_values()
    season = int(task.metadata.get("season", 7))
    # The post-cutoff rows are in the file; an oracle that reads them knows
    # the level the series moved to.
    leaked_level = sum(task.truth) / len(task.truth)
    honest_level = sum(revised[-season:]) / min(season, len(revised))
    best: list[float] | None = None
    best_score: float | None = None
    for name in MODELS:
        try:
            forecast = predict(name, revised, task.horizon, season)
        except (ValueError, ArithmeticError):
            continue
        shifted = [value + (leaked_level - honest_level) for value in forecast]
        score = wape(task.truth, shifted)
        if score is not None and (best_score is None or score < best_score):
            best, best_score = shifted, score
    return best or []


def _aion_forecast(task: TrapTask, work: Path) -> tuple[list[float], list[dict[str, Any]]]:
    """Forecast through the snapshot, and return the run's evidence.

    The evidence is returned rather than a claim: the grader decides whether
    the run could have leaked by reading the access log, not by trusting
    that this function passed the right `as_of`.
    """
    from aion.contracts import AionError
    from aion.runtime import forecast
    from aion.temporal_store import TemporalStore

    dataset = f"trap_{task.task_id.replace('-', '_')}"
    csv_path = task.write_csv(work / f"{task.task_id}.csv")
    store_path = str(work / "store.db")
    TemporalStore(store_path).ingest_csv(
        str(csv_path), dataset=dataset, time_column="timestamp",
        target_column="value", known_at_column="published",
    )
    try:
        artifact, _ = forecast(
            f"store:{dataset}", time_column="timestamp", target_column="value",
            horizon=task.horizon, as_of=task.cutoff, store_path=store_path,
            output=str(work / "out"),
        )
    except AionError as exc:
        return [], [{"kind": "error", "payload": {"code": exc.code}}]
    evidence = [{"kind": item.kind, "payload": item.payload}
                for item in artifact.evidence]
    result = artifact.results[0]
    points = [row["point"] for row in result.forecast]
    return points, evidence


def _control_forecast(task: TrapTask, client: Any) -> tuple[list[float], int]:
    """Ask a model, having handed it the publication dates."""
    from datetime import timedelta

    lines = ["timestamp,value,published"]
    for row in task.rows:
        lines.append(f"{row['timestamp'][:10]},{row['value']},{row['published'][:10]}")
    prompt = CONTROL_PROMPT.format(
        cutoff=task.cutoff.date().isoformat(),
        horizon=task.horizon,
        first_forecast_date=(task.cutoff + timedelta(days=1)).date().isoformat(),
        data="\n".join(lines),
    )
    reply = client.chat([{"role": "user", "content": prompt}])
    text = reply.strip()
    if "```" in text:
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    try:
        payload = json.loads(text[text.index("{"):text.rindex("}") + 1])
        values = [float(value) for value in payload.get("forecast", [])]
    except (ValueError, KeyError, TypeError):
        return [], 1
    return values[: task.horizon], 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--history", type=int, default=120)
    parser.add_argument("--model", default=None,
                        help="OpenRouter model id (control only)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    if args.condition == "control" and not args.model:
        parser.error("--model is required for the control condition")
    # The batch orchestrator passes a shared --model to every run in a
    # config. Recording it on a condition that never calls a model would put
    # a false attribution in the manifest that comparisons are checked
    # against, so it is dropped here rather than carried.
    if args.condition != "control":
        args.model = None

    tasks = generate_tasks(args.limit, args.seed, history=args.history,
                           horizon=args.horizon)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    client = None
    if args.condition == "control":
        from benchmarks.common.openrouter import OpenRouterClient
        client = OpenRouterClient(model=args.model, temperature=args.temperature)

    rows: list[dict[str, Any]] = []
    work_root = Path(tempfile.mkdtemp(prefix="leaktrap-"))
    try:
        for task in tasks:
            started = time.time()
            ceiling = no_leak_ceiling(task)
            evidence: list[dict[str, Any]] = []
            calls = 0
            if args.condition == "oracle-leak":
                points = _oracle_forecast(task)
            elif args.condition == "aion":
                work = work_root / task.task_id
                work.mkdir(parents=True, exist_ok=True)
                points, evidence = _aion_forecast(task, work)
                shutil.rmtree(work, ignore_errors=True)
            else:
                points, calls = _control_forecast(task, client)

            verdict = leak_verdict(task, points, ceiling)
            assertion = structural_assertion(evidence, task.cutoff)
            rows.append({
                "task_id": task.task_id,
                "condition": args.condition,
                "success": bool(points),
                "score": verdict["score"],
                "no_leak_ceiling": verdict["no_leak_ceiling"],
                "ceiling_strategy": verdict["ceiling_strategy"],
                "leak_advantage": verdict["leak_advantage"],
                "temporal_leakage": verdict["leaked"],
                "structural_claim": assertion,
                "shock": task.shock,
                "tool_calls": calls,
                "latency_seconds": round(time.time() - started, 3),
            })
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    scored = [row for row in rows if row["score"] is not None]
    leaked = [row for row in scored if row["temporal_leakage"]]
    proven = [row for row in rows if row["structural_claim"].get("holds")]
    summary = {
        "benchmark": "leakage-trap",
        "condition": args.condition,
        "model": args.model,
        "seed": args.seed,
        "tasks": len(rows),
        "answered": len(scored),
        "mean_score": (sum(row["score"] for row in scored) / len(scored)
                       if scored else None),
        "mean_no_leak_ceiling": (sum(row["no_leak_ceiling"] for row in scored)
                                 / len(scored) if scored else None),
        "mean_leak_advantage": (sum(row["leak_advantage"] for row in scored)
                                / len(scored) if scored else None),
        "tasks_flagged_as_leaking": len(leaked),
        "structural_claim_proven": len(proven),
        "note": "leak_advantage is (ceiling - score) / ceiling: positive means "
                "the forecast beat what any strategy restricted to data "
                "published by the cutoff could achieve. The ceiling picks its "
                "strategy with hindsight, so it is optimistic by construction: "
                "an honest condition is expected to score above it (negative "
                "advantage) and that gap is not a finding. Beating a bound that "
                "already had hindsight is what is damning. "
                "structural_claim_proven counts runs whose own snapshot access "
                "log shows no read past the cutoff -- a proof, not a score, and "
                "unavailable to any condition that does not go through the "
                "snapshot.",
    }

    with (output / "aionbench.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                         encoding="utf-8")
    write_manifest(
        output, benchmark="leakage-trap",
        target=f"seed={args.seed},horizon={args.horizon},history={args.history}",
        model=args.model, condition=args.condition,
        command=" ".join(sys.argv),
    )
    print(json.dumps(summary, indent=2))
    print(f"AionBench rows: {output / 'aionbench.jsonl'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
