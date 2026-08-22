"""Run the Context-is-Key benchmark against Gnomon or an LLM control.

The tasks, seeds, sampling protocol, and RCRPS metric are the official
ones from ``cik_benchmark``; this script only chooses which forecaster
answers them and collects the results.

Conditions:

- ``control``     official DirectPrompt LLM baseline via OpenRouter
- ``gnomon-pure``   Gnomon alone, context text ignored
- ``gnomon-agent``  OpenRouter LLM proposes typed context events; Gnomon
                  validates, computes, or abstains
- ``gnomon-conditional``  the same proposer with Gnomon's explicitly labelled
                  prospective context lane enabled; the immutable primary is
                  retained beside every conditional forecast
- ``gnomon-mcp``    OpenRouter model may call Gnomon's real MCP surface or
                  submit its own labeled forecast

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

The headline CiK number is ``mean_rcrps_capped_imputed`` in
``summary.json``: per-run RCRPS capped at 5.0 and every abstained or
errored run imputed at 5.0, matching the official aggregation's
cap-and-impute rule (``compile_roi_results.py`` upstream), so an
abstention can never improve it. ``mean_rcrps_scored_only`` (the
uncapped mean over scored runs only) is reported beside it and is NOT
comparable to published numbers. Both local aggregates are unweighted
over runs — the official per-task weighting is not reproduced here; the
official per-run scores are in ``scores.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.common.manifest import code_revision, write_manifest  # noqa: E402
from benchmarks.common.records import RecordWriter, RunRecord  # noqa: E402

ABSTAIN_MARKER = "GNOMON_ABSTAINED"

#: The official aggregation cap (``CAP = 5`` in the upstream
#: ``compile_roi_results.py``): per-run RCRPS above it is clipped to it,
#: and runs with no score are imputed at it.
RCRPS_CAP = 5.0


def capped_imputed_mean(
    scores: list[float], unscored_runs: int, cap: float = RCRPS_CAP
) -> float | None:
    """Official-style aggregate: mean over ALL runs, capped and imputed.

    Every score is clipped to ``cap`` and every unscored run (abstained
    or errored) counts as ``cap``, so a missing forecast can never
    improve the aggregate. The upstream script also treats NaN and
    negative entries as failures at the cap; RCRPS is non-negative by
    construction, so that branch only matters for corrupt score files.
    Unweighted over runs: the official per-task weights are not
    reproduced here.
    """
    values = [min(s, cap) if s >= 0 and math.isfinite(s) else cap
              for s in scores]
    values.extend([cap] * unscored_runs)
    if not values:
        return None
    return sum(values) / len(values)


def load_run_extra_info(runs_dir: Path, task_name: str, seed) -> dict:
    """Parse the official per-run ``extra_info`` dump, if present.

    ``cik_benchmark.evaluation.evaluate_task`` writes each run's
    ``extra_info`` with ``pprint``; both adapters put only literals in
    it, so ``ast.literal_eval`` reads it back. Abstained and errored
    runs never produce the file; anything unparseable yields ``{}``
    rather than failing result collection.
    """
    import ast

    path = runs_dir / task_name / str(seed) / "extra_info"
    if not path.exists():
        return {}
    try:
        parsed = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_method(args):
    if args.method == "control":
        if not args.model:
            raise SystemExit("--model is required for the control condition")
        from benchmarks.cik.openrouter_direct_prompt import OpenRouterDirectPrompt

        return OpenRouterDirectPrompt(
            openrouter_model=args.model,
            temperature=args.temperature,
            fail_on_invalid=args.fail_on_invalid,
        )
    if args.method == "gnomon-mcp":
        if not args.model:
            raise SystemExit("--method gnomon-mcp requires --model")
        if args.future_context or args.structural_context:
            raise SystemExit(
                "gnomon-mcp takes no lane flags: the model chooses its own "
                "tool arguments (future_events, structural_events, ...) "
                "per call"
            )
        from benchmarks.cik.mcp_agent import McpAgentForecaster

        return McpAgentForecaster(
            args.model, temperature=args.temperature,
            trace_dir=Path(args.output_dir) / "mcp-traces",
        )
    conditional_arm = args.method == "gnomon-conditional"
    if conditional_arm:
        # This is deliberately a named condition rather than an easy-to-miss
        # flag combination. It keeps the benchmark's primary and conditional
        # arms identifiable in manifests and cache keys.
        args.future_context = True
    if args.structural_context and not args.future_context:
        raise SystemExit("--structural-context requires --future-context")
    from benchmarks.cik.gnomon_forecaster import GnomonForecaster

    mode = "agent" if args.method in {
        "gnomon-agent", "gnomon-conditional"
    } else "pure"
    if args.future_context and mode != "agent":
        raise SystemExit("--future-context requires --method gnomon-agent")
    return GnomonForecaster(
        mode=mode, openrouter_model=args.model, temperature=args.temperature,
        future_context=args.future_context,
        structural_context=args.structural_context,
    )


def run(args) -> int:
    run_revision = code_revision()
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
    write_manifest(
        output_dir,
        benchmark="cik",
        condition=args.method,
        model=args.model,
        command=" ".join(sys.argv),
        seeds=args.seeds,
        n_samples=n_samples,
        task_filter=args.task_filter,
        fail_on_invalid=args.fail_on_invalid if args.method == "control" else None,
        status="ok",
        code_revision=run_revision,
    )
    return 0


def write_outputs(results: dict, method, args, output_dir: Path) -> None:
    scores_path = output_dir / "scores.csv"
    records_path = output_dir / "gnomonbench.jsonl"
    # RecordWriter appends; a rerun into the same output dir must replace
    # the previous run's rows (as scores.csv does), not accumulate them.
    records_path.unlink(missing_ok=True)
    jsonl = RecordWriter(records_path)
    runs_dir = output_dir / "runs"
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
                extra_info = load_run_extra_info(runs_dir, task_name, seed)
                latency = extra_info.get("total_time")
                jsonl.write(RunRecord(
                    task_id=f"{task_name}-seed{seed}",
                    success=finite,
                    appropriate_abstention=abstained,
                    # One gnomon.forecast call per gnomon run; the control
                    # calls no tools, and a routed run that chose (or fell
                    # back to) the direct path made none either. Per-run
                    # cost stays 0: the adapters only report cost
                    # accumulated across the whole client lifetime, and
                    # faking a per-run split would be worse than the zero.
                    tool_calls=(
                        int(extra_info["mcp_calls"])
                        if "mcp_calls" in extra_info
                        else 1 if is_gnomon
                        and extra_info.get("route", "gnomon") == "gnomon"
                        else 0
                    ),
                    latency_seconds=(
                        float(latency)
                        if isinstance(latency, (int, float)) else 0.0
                    ),
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
        "rcrps_cap": RCRPS_CAP,
        "mean_rcrps_capped_imputed": capped_imputed_mean(
            scored, abstentions + errors
        ),
        "mean_rcrps_scored_only": (
            sum(scored) / len(scored) if scored else None
        ),
        "note": (
            "mean_rcrps_capped_imputed follows the official aggregation "
            "rule (cap per-run RCRPS at 5.0, impute every abstained or "
            "errored run at 5.0), so abstention can never improve it; it "
            "is the number to put next to published means. "
            "mean_rcrps_scored_only averages scored runs only and is "
            "flattered by abstention — never quote it without the "
            "abstention and error counts beside it. Both are unweighted "
            "over runs; the official per-task weighting is not "
            "reproduced here."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Official per-run scores: {scores_path}")
    print(f"GnomonBench rows: {jsonl.path} ({jsonl.count} rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["control", "gnomon-pure", "gnomon-agent",
                 "gnomon-conditional", "gnomon-mcp"],
        help="gnomon-mcp: the model holds Gnomon's real MCP tools and "
             "chooses per task whether to use them; the route is "
             "classified from the transcript "
             "(docs/design/cik-mcp-tool-arm.md)",
    )
    parser.add_argument(
        "--model",
        help="OpenRouter model id (control and agent conditions), e.g. openai/gpt-4o",
    )
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds per task (official: 5)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Forecast samples per run (default: official)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--fail-on-invalid", action=argparse.BooleanOptionalAction,
        default=True,
        help="Control condition only: error the run when rejection sampling "
             "cannot collect n_samples valid forecasts (the official "
             "DirectPrompt default). --no-fail-on-invalid scores the run on "
             "however many valid forecasts were collected — a protocol "
             "deviation that must be disclosed",
    )
    parser.add_argument("--task-filter", default=None,
                        help="Only run tasks whose class name contains this")
    parser.add_argument(
        "--future-context", action="store_true",
        help="Enable Gnomon's context.future_events lane (gnomon-agent only): "
             "the proposer may quote constraint/override spans, admitted by "
             "textual verification instead of fold ablation",
    )
    parser.add_argument(
        "--structural-context", action="store_true",
        help="Additionally enable context.structural_events (requires "
             "--future-context): the proposer may classify stated cessations "
             "into the closed structural-effect menu (trend_ceases, "
             "level_matches_seasonal_high/_low); every "
             "applied quantity is derived from Gnomon's own emitted path. "
             "Experimental: results/structural-effects/HYPOTHESIS.md",
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the official result cache")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.method in {"gnomon-agent", "gnomon-conditional"} and not args.model:
        parser.error(f"--model is required for {args.method}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
