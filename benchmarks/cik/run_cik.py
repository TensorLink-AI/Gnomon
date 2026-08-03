"""Run the Context-is-Key benchmark against Gnomon or an LLM control.

The tasks, seeds, sampling protocol, and RCRPS metric are the official
ones from ``cik_benchmark``; this script only chooses which forecaster
answers them and collects the results.

Conditions:

- ``control``     official DirectPrompt LLM baseline via OpenRouter
- ``gnomon-pure``   Gnomon alone, context text ignored
- ``gnomon-agent``  OpenRouter LLM proposes typed context events; Gnomon
                  validates, computes, or abstains

Examples
--------
Full official run (5 seeds, official sample count), control vs treatment::

    python -m benchmarks.cik.run_cik --method control \
        --model openai/gpt-4o --output-dir results/cik-control
    python -m benchmarks.cik.run_cik --method gnomon-agent \
        --model openai/gpt-4o --output-dir results/cik-gnomon

Then compare the completion/safety view::

    gnomon eval compare \
        --baseline results/cik-control/gnomonbench.jsonl \
        --treatment results/cik-gnomon/gnomonbench.jsonl

The headline CiK number is the mean RCRPS written to ``summary.json``;
the official per-run scores are in ``scores.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import RecordWriter, RunRecord  # noqa: E402

ABSTAIN_MARKER = "GNOMON_ABSTAINED"


def build_method(args):
    if args.method == "control":
        if not args.model:
            raise SystemExit("--model is required for the control condition")
        from benchmarks.cik.openrouter_direct_prompt import OpenRouterDirectPrompt

        return OpenRouterDirectPrompt(
            openrouter_model=args.model,
            temperature=args.temperature,
            fail_on_invalid=False,
        )
    from benchmarks.cik.gnomon_forecaster import GnomonForecaster

    mode = "agent" if args.method == "gnomon-agent" else "pure"
    if args.future_context and mode != "agent":
        raise SystemExit("--future-context requires --method gnomon-agent")
    return GnomonForecaster(
        mode=mode, openrouter_model=args.model, temperature=args.temperature,
        future_context=args.future_context,
    )


def run(args) -> int:
    from cik_benchmark import ALL_TASKS
    from cik_benchmark.config import DEFAULT_N_SAMPLES
    from cik_benchmark.evaluation import evaluate_all_tasks, evaluate_task

    n_samples = args.n_samples or DEFAULT_N_SAMPLES
    method = build_method(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_filter:
        selected = [
            task for task in ALL_TASKS
            if args.task_filter.lower() in task.__name__.lower()
        ]
        if not selected:
            raise SystemExit(f"No CiK task matches {args.task_filter!r}")
        print(f"Running {len(selected)} task(s) matching {args.task_filter!r}")
        results = {}
        for task_cls in selected:
            for seed in range(1, args.seeds + 1):
                name, row = evaluate_task(
                    task_cls, seed, method, n_samples,
                    output_folder=output_dir / "runs",
                )
                results.setdefault(name, []).append(row)
    else:
        results = evaluate_all_tasks(
            method,
            seeds=args.seeds,
            n_samples=n_samples,
            output_folder=output_dir / "runs",
            use_cache=not args.no_cache,
            cache_name=method.cache_name,
            max_parallel=args.max_parallel,
        )

    write_outputs(results, method, args, output_dir)
    return 0


def write_outputs(results: dict, method, args, output_dir: Path) -> None:
    scores_path = output_dir / "scores.csv"
    jsonl = RecordWriter(output_dir / "gnomonbench.jsonl")
    is_gnomon = args.method != "control"

    scored: list[float] = []
    abstentions = 0
    errors = 0
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "seed", "rcrps", "error"])
        for task_name in sorted(results):
            for row in results[task_name]:
                seed = row.get("seed")
                score = row.get("score")
                error = row.get("error", "")
                finite = isinstance(score, (int, float)) and math.isfinite(score)
                abstained = ABSTAIN_MARKER in str(error)
                if finite:
                    scored.append(float(score))
                elif abstained:
                    abstentions += 1
                else:
                    errors += 1
                writer.writerow(
                    [task_name, seed, score if finite else "", error]
                )
                jsonl.write(RunRecord(
                    task_id=f"{task_name}-seed{seed}",
                    success=finite,
                    appropriate_abstention=abstained,
                    tool_calls=1 if is_gnomon else 0,
                    extra={"rcrps": float(score) if finite else None,
                           "method": method.cache_name},
                ))

    summary = {
        "benchmark": "context-is-key",
        "method": method.cache_name,
        "condition": args.method,
        "model": args.model,
        "seeds": args.seeds,
        "runs_scored": len(scored),
        "runs_abstained": abstentions,
        "runs_errored": errors,
        "mean_rcrps": sum(scored) / len(scored) if scored else None,
        "note": (
            "mean_rcrps averages scored runs only; abstentions and errors "
            "are reported separately and must be disclosed next to it"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Official per-run scores: {scores_path}")
    print(f"GnomonBench rows: {jsonl.path} ({jsonl.count} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["control", "gnomon-pure", "gnomon-agent"],
    )
    parser.add_argument(
        "--model",
        help="OpenRouter model id (control and gnomon-agent), e.g. openai/gpt-4o",
    )
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds per task (official: 5)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Forecast samples per run (default: official)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--task-filter", default=None,
                        help="Only run tasks whose class name contains this")
    parser.add_argument(
        "--future-context", action="store_true",
        help="Enable Gnomon's context.future_events lane (gnomon-agent only): "
             "the proposer may quote constraint/override spans, admitted by "
             "textual verification instead of fold ablation",
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the official result cache")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.method == "gnomon-agent" and not args.model:
        parser.error("--model is required for gnomon-agent")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
