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
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

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


def _available_memory_mb() -> int | None:
    """Return Linux's immediately available memory without extra deps."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _process_tree_rss_mb(root_pid: int) -> int:
    """Return resident RAM for a process and its descendants on Linux."""
    processes: dict[int, tuple[int, int]] = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            pid = int(status.parent.name)
            ppid = 0
            rss_kb = 0
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
            processes[pid] = (ppid, rss_kb)
        except (OSError, ValueError, IndexError):
            continue
    family = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in processes.items():
            if ppid in family and pid not in family:
                family.add(pid)
                changed = True
    return sum(processes.get(pid, (0, 0))[1] for pid in family) // 1024


def _task_information_profile(task) -> dict:
    """Describe whether a scored case contains forecast information.

    This is computed after the forecasting method returns and is never passed
    into a model prompt. It does not alter the official score. Its purpose is
    to stop a constant-history/identical-constant-future case from being
    mistaken for evidence that a forecaster learned anything.
    """
    frames = {"past": task.past_time, "future": task.future_time}
    columns = list(getattr(task.past_time, "columns", []))
    profiles: dict[str, dict] = {}
    for name, frame in frames.items():
        per_column = {}
        for column in columns:
            values = [float(value) for value in frame[column].tolist()]
            finite = [value for value in values if math.isfinite(value)]
            unique = set(finite)
            per_column[str(column)] = {
                "points": len(values),
                "finite_points": len(finite),
                "unique_finite_values": len(unique),
                "constant_value": (
                    finite[0] if finite and len(unique) == 1 else None),
            }
        profiles[name] = per_column
    same_constant_channels = []
    for column in columns:
        past = profiles["past"][str(column)]
        future = profiles["future"][str(column)]
        if (past["finite_points"] == past["points"]
                and future["finite_points"] == future["points"]
                and past["constant_value"] is not None
                and past["constant_value"] == future["constant_value"]):
            same_constant_channels.append(str(column))
    return {
        "version": "0.1",
        "computed_after_forecast": True,
        "passed_to_forecaster": False,
        "channels": len(columns),
        "past": profiles["past"],
        "future": profiles["future"],
        "same_constant_past_and_future_channels": same_constant_channels,
        "degenerate_same_constant_case": bool(
            columns and len(same_constant_channels) == len(columns)),
        "interpretation": (
            "benchmark_information_diagnostic_not_forecast_evidence"),
    }


def _candidate_interval_diagnostics(task, rows: list[dict],
                                    nominal_coverage: float = .8) -> dict:
    """Measure one candidate's retained intervals against sealed outcomes.

    This is post-forecast benchmark telemetry.  It deliberately consumes the
    task's future target only after publication and returns no value that can
    flow back into model or scenario selection.
    """
    future = task.future_time
    columns = list(future.columns)
    if not columns:
        raise ValueError("future target has no columns")
    series = future[columns[-1]]
    values = series.tolist() if hasattr(series, "tolist") else list(series)
    if len(values) != len(rows) or not values:
        raise ValueError("future target and forecast horizon differ")
    alpha = 1.0 - nominal_coverage
    if not 0.0 < alpha < 1.0:
        raise ValueError("nominal coverage must be strictly between zero and one")
    covered = 0
    weighted_interval_scores = []
    for truth_raw, row in zip(values, rows):
        truth = float(truth_raw)
        lower = float(row["q10"])
        median = float(row["q50"])
        upper = float(row["q90"])
        if not all(math.isfinite(value) for value in
                   (truth, lower, median, upper)):
            raise ValueError("interval diagnostic contains non-finite values")
        if not lower <= median <= upper:
            raise ValueError("interval quantiles are crossed")
        covered += int(lower <= truth <= upper)
        interval_score = upper - lower
        if truth < lower:
            interval_score += (2.0 / alpha) * (lower - truth)
        elif truth > upper:
            interval_score += (2.0 / alpha) * (truth - upper)
        # Proper WIS with the median and one central (1-alpha) interval.
        weighted_interval_scores.append(
            (.5 * abs(truth - median) + (alpha / 2.0) * interval_score)
            / (.5 + alpha / 2.0))
    return {
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": covered / len(values),
        "wis": sum(weighted_interval_scores) / len(weighted_interval_scores),
        "interval_points": len(values),
        "computed_after_forecast": True,
        "passed_to_forecaster": False,
    }


def _counterfactual_candidate_scores(task, extra_info: dict,
                                     n_samples: int) -> list[dict]:
    """Score sealed publication candidates after forecasting, for diagnosis.

    These outcomes are benchmark-only telemetry. They are computed after the
    model and Gnomon have returned and are never passed back into selection.
    Quantile rows are reconstructed with the same deterministic sampler used
    by the MCP adapter, so the report can separate candidate quality from
    selector quality without another stochastic model call.
    """
    publication = extra_info.get("publication") or {}
    portfolio = publication.get("candidate_portfolio") or []
    if not isinstance(portfolio, list) or not portfolio:
        return []
    from benchmarks.cik.mcp_agent import samples_from_quantile_rows
    import numpy as np

    output = []
    selected = str(publication.get("recommended_scenario_id") or "")
    for item in portfolio:
        if not isinstance(item, dict) or not isinstance(item.get("forecast"), list):
            continue
        scenario_id = str(item.get("scenario_id") or "")
        human_selection_eligible = bool(item.get(
            "human_selection_eligible",
            item.get("selection_eligible", scenario_id == "primary")))
        try:
            paths = samples_from_quantile_rows(item["forecast"], n_samples)
            evaluation = task.evaluate(np.asarray(paths, dtype=float)[:, :, None])
            score = evaluation.get("metric") if isinstance(evaluation, dict) \
                else evaluation
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("non-finite counterfactual score")
            record = {
                "scenario_id": scenario_id,
                "role": str(item.get("role") or "unknown"),
                "selected": scenario_id == selected,
                "human_selection_eligible": human_selection_eligible,
                "score": score,
                "scoring_representation": "deterministic_quantile_reconstruction",
                "computed_after_forecast": True,
                "passed_to_forecaster": False,
            }
            try:
                record.update(_candidate_interval_diagnostics(
                    task, item["forecast"]))
            except Exception as error:
                record["calibration_error"] = (
                    f"{type(error).__name__}: {error}"[:300])
            output.append(record)
        except Exception as error:
            output.append({
                "scenario_id": scenario_id,
                "role": str(item.get("role") or "unknown"),
                "selected": scenario_id == selected,
                "human_selection_eligible": human_selection_eligible,
                "score": None,
                "error": f"{type(error).__name__}: {error}"[:300],
                "computed_after_forecast": True,
                "passed_to_forecaster": False,
            })
    return output


def _isolated_case_worker(conn, task_name: str, seed: int, args_dict: dict,
                          n_samples: int, runs_dir: str) -> None:
    """Evaluate exactly one task/seed and return serializable state."""
    try:
        from cik_benchmark import ALL_TASKS
        from cik_benchmark.evaluation import evaluate_task

        classes = {task.__name__: task for task in ALL_TASKS}
        method = build_method(SimpleNamespace(**args_dict))
        # Upstream task instances do not retain the seed. Each disposable
        # worker owns exactly one case, so bind the runner's authoritative
        # identity to the method for trace/receipt naming.
        setattr(method, "benchmark_seed", seed)

        def profiled_method(*, task_instance, n_samples):
            result = method(task_instance=task_instance, n_samples=n_samples)
            if isinstance(result, tuple):
                samples, extra_info = result
                extra_info = dict(extra_info or {})
            else:
                samples, extra_info = result, {}
            extra_info["benchmark_input_profile"] = (
                _task_information_profile(task_instance))
            extra_info["benchmark_counterfactual_candidate_scores"] = (
                _counterfactual_candidate_scores(
                    task_instance, extra_info, n_samples))
            return samples, extra_info

        name, row = evaluate_task(
            classes[task_name], seed, profiled_method, n_samples,
            output_folder=Path(runs_dir),
        )
        conn.send({"ok": True, "name": name, "row": row})
    except BaseException as exc:  # child must report failures, then disappear
        conn.send({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        })
    finally:
        conn.close()


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "case-checkpoint.json"


def _load_checkpoint(output_dir: Path) -> dict[str, dict]:
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    # Provider/network failures are missing observations, not model scores.
    # Keep valid work and retry only those cases on resume.
    retryable = (
        "HTTP 403", "HTTP 408", "HTTP 409", "HTTP 429", "HTTP 500",
        "HTTP 502", "HTTP 503", "HTTP 504", "daily limit",
        "timed out", "timeout", "connection reset", "temporary failure",
        "case_timeout_after_", "case_process_exit_", "system_memory_guard",
        "simplenamespace' object has no attribute",
    )
    return {
        key: item for key, item in payload.items()
        if not any(marker.casefold() in str(
            (item.get("row") or {}).get("error") or "").casefold()
                   for marker in retryable)
    }


def _write_checkpoint(output_dir: Path, completed: dict[str, dict]) -> None:
    path = _checkpoint_path(output_dir)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(completed, indent=2, default=str) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_identity(args, selected, n_samples: int,
                         revision: str) -> dict[str, object]:
    """Describe every input that can change a resumed CiK case result."""
    return {
        "schema_version": 1,
        "benchmark": "context-is-key",
        "code_revision": revision,
        "method": args.method,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "temperature": args.temperature,
        "selected_tasks": [task.__name__ for task in selected],
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "n_samples": n_samples,
        "fail_on_invalid": args.fail_on_invalid,
        "future_context": args.future_context,
        "structural_context": args.structural_context,
        "mcp_profile": args.mcp_profile,
        "mcp_output_role": args.mcp_output_role,
        "mcp_candidate_paths": args.mcp_candidate_paths,
        "mcp_candidate_history_rows": args.mcp_candidate_history_rows,
        "mcp_candidate_temporal_facts": args.mcp_candidate_temporal_facts,
        "no_cache": args.no_cache,
        "case_timeout_seconds": args.case_timeout_seconds,
        "case_memory_mb": args.case_memory_mb,
        "min_free_memory_mb": args.min_free_memory_mb,
        "max_parallel": args.max_parallel,
    }


def _prepare_checkpoint_identity(output_dir: Path,
                                 identity: dict[str, object],
                                 *, fresh: bool) -> None:
    """Fail closed instead of mixing checkpoints from different runs."""
    path = output_dir / "run_identity.json"
    checkpoint = _checkpoint_path(output_dir)
    if fresh:
        # Clear the only resumable state before adopting the new identity.
        # Otherwise an interrupt between writing identity and the first case
        # could make old rows look as if they belonged to the fresh run.
        _write_checkpoint(output_dir, {})
    if not fresh and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            raise SystemExit(
                "resume identity mismatch; use a new output directory or "
                "pass --no-resume")
        return
    if not fresh and checkpoint.is_file():
        raise SystemExit(
            "cannot resume CiK checkpoint without run_identity.json; use a "
            "new output directory or pass --no-resume")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _run_isolated_cases(selected, args, n_samples: int,
                        output_dir: Path) -> dict:
    """Run sequential disposable cases with resume, timeout, and RAM caps."""
    if args.max_parallel != 1:
        raise SystemExit(
            "resource-safe CiK execution requires --max-parallel 1; shard "
            "independent output directories across separate machines")
    available = _available_memory_mb()
    if available is not None and available < args.min_free_memory_mb:
        raise SystemExit(
            f"CiK preflight refused to start: {available} MiB available, "
            f"but --min-free-memory-mb={args.min_free_memory_mb}")

    completed = {} if args.no_resume else _load_checkpoint(output_dir)
    args_dict = vars(args).copy()
    ctx = mp.get_context("spawn")
    seed_values = range(args.seed_start, args.seed_start + args.seeds)
    total = len(selected) * len(seed_values)
    ordinal = 0
    for task_cls in selected:
        for seed in seed_values:
            ordinal += 1
            key = f"{task_cls.__name__}::seed={seed}"
            if key in completed:
                print(f"[{ordinal}/{total}] resume {key}", flush=True)
                continue
            available = _available_memory_mb()
            if available is not None and available < args.min_free_memory_mb:
                raise SystemExit(
                    f"CiK stopped safely before {key}: only {available} MiB "
                    "available; completed cases remain resumable")
            print(f"[{ordinal}/{total}] start {key}", flush=True)
            parent, child = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_isolated_case_worker,
                args=(child, task_cls.__name__, seed, args_dict, n_samples,
                      str(output_dir / "runs")),
            )
            process.start()
            child.close()
            deadline = time.monotonic() + args.case_timeout_seconds
            forced_error = None
            peak_rss_mb = 0
            while process.is_alive() and time.monotonic() < deadline:
                process.join(0.25)
                rss_mb = _process_tree_rss_mb(process.pid)
                peak_rss_mb = max(peak_rss_mb, rss_mb)
                if args.case_memory_mb > 0 and rss_mb > args.case_memory_mb:
                    forced_error = (
                        f"case_rss_limit_exceeded: {rss_mb} MiB > "
                        f"{args.case_memory_mb} MiB")
                    break
                available = _available_memory_mb()
                if (available is not None
                        and available < args.min_free_memory_mb):
                    forced_error = (
                        f"system_memory_guard: only {available} MiB available")
                    break
            if process.is_alive():
                process.terminate()
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join()
                payload = {"ok": False, "error": forced_error or (
                    f"case_timeout_after_{args.case_timeout_seconds}s")}
            elif parent.poll():
                payload = parent.recv()
            else:
                payload = {"ok": False, "error": (
                    f"case_process_exit_{process.exitcode}; possible memory "
                    "limit or native-library failure")}
            parent.close()
            if payload.get("ok"):
                name = payload["name"]
                row = payload["row"]
            else:
                name = task_cls.__name__
                row = {"seed": seed, "score": None,
                       "error": payload.get("error", "isolated_case_failed")}
                trace = payload.get("traceback")
                if trace:
                    error_dir = output_dir / "case-errors"
                    error_dir.mkdir(parents=True, exist_ok=True)
                    (error_dir / f"{task_cls.__name__}-seed{seed}.txt").write_text(
                        trace, encoding="utf-8")
            completed[key] = {"name": name, "row": row}
            _write_checkpoint(output_dir, completed)
            print(f"[{ordinal}/{total}] finish {key}: "
                  f"{row.get('score', row.get('error'))}; "
                  f"peak_rss={peak_rss_mb}MiB", flush=True)

    results: dict[str, list[dict]] = {}
    for item in completed.values():
        results.setdefault(item["name"], []).append(item["row"])
    return results


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
    def provider():
        import os
        key_env = getattr(args, "api_key_env", "ENGY_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            from benchmarks.common.envfile import load_env_file
            load_env_file()
            api_key = os.environ.get(key_env)
        if not api_key:
            raise SystemExit(f"{key_env} is not set")
        return getattr(args, "base_url", "https://api.engy.ai/v1"), api_key
    if args.method == "control":
        if not args.model:
            raise SystemExit("--model is required for the control condition")
        from benchmarks.cik.openrouter_direct_prompt import OpenRouterDirectPrompt

        base_url, api_key = provider()
        return OpenRouterDirectPrompt(
            openrouter_model=args.model,
            temperature=args.temperature,
            fail_on_invalid=args.fail_on_invalid,
            base_url=base_url, api_key=api_key,
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
        base_url, api_key = provider()

        return McpAgentForecaster(
            args.model, temperature=args.temperature,
            trace_dir=Path(args.output_dir) / "mcp-traces",
            profile=args.mcp_profile,
            output_role=args.mcp_output_role,
            candidate_sample_budget=args.mcp_candidate_paths,
            candidate_history_budget=args.mcp_candidate_history_rows,
            candidate_temporal_facts_enabled=args.mcp_candidate_temporal_facts,
            base_url=base_url, api_key=api_key,
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


def select_tasks(all_tasks, *, task_names: list[str] | None = None,
                 task_filter: str | None = None):
    """Select an exact crash-safe shard without changing task semantics."""
    names = list(task_names or [])
    if names and task_filter:
        raise SystemExit("--task-name and --task-filter are mutually exclusive")
    if len(names) != len(set(names)):
        raise SystemExit("--task-name may not repeat a task")
    if names:
        by_name = {task.__name__: task for task in all_tasks}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise SystemExit("Unknown CiK task name(s): " + ", ".join(missing))
        return [by_name[name] for name in names]
    if task_filter:
        selected = [
            task for task in all_tasks
            if task_filter.lower() in task.__name__.lower()
        ]
        if not selected:
            raise SystemExit(f"No CiK task matches {task_filter!r}")
        return selected
    return list(all_tasks)


def run(args) -> int:
    run_revision = code_revision()
    from cik_benchmark import ALL_TASKS
    from cik_benchmark.config import DEFAULT_N_SAMPLES

    n_samples = args.n_samples or DEFAULT_N_SAMPLES
    method = build_method(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_tasks(
        ALL_TASKS, task_names=args.task_name, task_filter=args.task_filter)
    run_identity = _checkpoint_identity(
        args, selected, n_samples, run_revision)
    _prepare_checkpoint_identity(
        output_dir, run_identity, fresh=args.no_resume)
    if args.task_name:
        print(f"Running {len(selected)} exact task(s): " +
              ", ".join(args.task_name), flush=True)
        results = _run_isolated_cases(selected, args, n_samples, output_dir)
    elif args.task_filter:
        print(f"Running {len(selected)} task(s) matching {args.task_filter!r}")
        results = _run_isolated_cases(selected, args, n_samples, output_dir)
    else:
        # The upstream all-task runner keeps a long-lived process pool and
        # cannot enforce per-case memory ceilings. Every case therefore uses
        # the same disposable-process safety contract.
        print(f"Running all {len(selected)} CiK tasks safely", flush=True)
        results = _run_isolated_cases(selected, args, n_samples, output_dir)

    write_outputs(results, method, args, output_dir,
                  run_identity=run_identity)
    write_manifest(
        output_dir,
        benchmark="cik",
        condition=args.method,
        model=args.model,
        command=" ".join(sys.argv),
        seeds=args.seeds,
        seed_start=args.seed_start,
        n_samples=n_samples,
        task_filter=args.task_filter,
        task_names=args.task_name or None,
        fail_on_invalid=args.fail_on_invalid if args.method == "control" else None,
        status="ok",
        code_revision=run_revision,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    return 0


def write_outputs(results: dict, method, args, output_dir: Path,
                  *, run_identity: dict[str, object] | None = None) -> None:
    scores_path = output_dir / "scores.csv"
    records_path = output_dir / "gnomonbench.jsonl"
    selection_records_path = output_dir / "selection-diagnostics.jsonl"
    # RecordWriter appends; a rerun into the same output dir must replace
    # the previous run's rows (as scores.csv does), not accumulate them.
    records_path.unlink(missing_ok=True)
    selection_records_path.unlink(missing_ok=True)
    jsonl = RecordWriter(records_path)
    runs_dir = output_dir / "runs"
    is_gnomon = args.method != "control"

    scored: list[float] = []
    abstentions = 0
    errors = 0
    degenerate_same_constant_runs = 0
    perfect_scores_on_degenerate_runs = 0
    selection_diagnostics: list[dict] = []
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
                candidate_scores = extra_info.get(
                    "benchmark_counterfactual_candidate_scores") or []
                finite_candidates = [
                    item for item in candidate_scores
                    if isinstance(item.get("score"), (int, float))
                    and math.isfinite(item["score"])
                ]
                selected_candidate = next(
                    (item for item in finite_candidates
                     if item.get("selected") is True), None)
                primary_candidate = next(
                    (item for item in finite_candidates
                     if item.get("role") == "immutable_primary"), None)
                if selected_candidate is not None and primary_candidate is not None:
                    best_candidate = min(
                        finite_candidates, key=lambda item: item["score"])
                    eligible_candidates = [
                        item for item in finite_candidates
                        if item.get("human_selection_eligible") is True
                    ]
                    best_eligible = (
                        min(eligible_candidates,
                            key=lambda item: item["score"])
                        if eligible_candidates else selected_candidate)
                    selection_record = {
                        "benchmark_method": method.cache_name,
                        "diagnostic_schema_version": "1",
                        "task": task_name,
                        "seed": seed,
                        "selected_score": float(selected_candidate["score"]),
                        "selected_role": str(selected_candidate.get(
                            "role") or "unknown"),
                        "primary_score": float(primary_candidate["score"]),
                        "best_candidate_score": float(best_candidate["score"]),
                        "best_eligible_candidate_score": float(
                            best_eligible["score"]),
                        "selected_primary": (
                            selected_candidate.get("scenario_id") ==
                            primary_candidate.get("scenario_id")),
                        "selected_hindsight_best": (
                            selected_candidate.get("scenario_id") ==
                            best_candidate.get("scenario_id")),
                        "selected_best_eligible": (
                            selected_candidate.get("scenario_id") ==
                            best_eligible.get("scenario_id")),
                        "primary_forecast_unchanged": extra_info.get(
                            "primary_forecast_unchanged"),
                        "automation_eligible": extra_info.get(
                            "automation_eligible"),
                        "computed_after_forecast": True,
                        "passed_to_forecaster": False,
                        "candidates": candidate_scores,
                    }
                    selection_diagnostics.append(selection_record)
                    with selection_records_path.open(
                            "a", encoding="utf-8") as selection_handle:
                        selection_handle.write(json.dumps(
                            selection_record, sort_keys=True) + "\n")
                input_profile = extra_info.get("benchmark_input_profile") or {}
                degenerate = bool(
                    input_profile.get("degenerate_same_constant_case"))
                if degenerate:
                    degenerate_same_constant_runs += 1
                    if finite and float(score) == 0.0:
                        perfect_scores_on_degenerate_runs += 1
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
                    extra={
                        "rcrps": float(score) if finite else None,
                        "method": method.cache_name,
                        "benchmark_input_profile": input_profile or None,
                        "score_information": (
                            "uninformative_same_constant_past_and_future"
                            if degenerate else "ordinary_scored_case"),
                    },
                ))

    summary = {
        "benchmark": "context-is-key",
        "method": method.cache_name,
        "condition": args.method,
        "model": args.model,
        "seeds": args.seeds,
        "seed_start": getattr(args, "seed_start", 1),
        "runs_scored": len(scored),
        "runs_abstained": abstentions,
        "runs_errored": errors,
        "degenerate_same_constant_runs": degenerate_same_constant_runs,
        "perfect_scores_on_degenerate_runs": (
            perfect_scores_on_degenerate_runs),
        "rcrps_cap": RCRPS_CAP,
        "mean_rcrps_capped_imputed": capped_imputed_mean(
            scored, abstentions + errors
        ),
        "mean_rcrps_scored_only": (
            sum(scored) / len(scored) if scored else None
        ),
        "selection_diagnostics": _summarize_selection_diagnostics(
            selection_diagnostics),
        "run_identity": run_identity,
        "note": (
            "mean_rcrps_capped_imputed follows the official aggregation "
            "rule (cap per-run RCRPS at 5.0, impute every abstained or "
            "errored run at 5.0), so abstention can never improve it; it "
            "is the number to put next to published means. "
            "mean_rcrps_scored_only averages scored runs only and is "
            "flattered by abstention — never quote it without the "
            "abstention and error counts beside it. Both are unweighted "
            "over runs; the official per-task weighting is not "
            "reproduced here. Degenerate same-constant past/future cases "
            "remain in the official aggregate but are counted separately; "
            "a perfect score on such a case is not forecasting uplift."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Official per-run scores: {scores_path}")
    print(f"GnomonBench rows: {jsonl.path} ({jsonl.count} rows)")


def _summarize_selection_diagnostics(rows: list[dict]) -> dict:
    """Summarize post-outcome diagnostics without creating forecast evidence.

    These values measure the publication policy after the run. They must never
    be passed to the forecaster or used to select a candidate in the same run.
    """
    if not rows:
        return {"cases": 0, "available": False}

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    selected = mean("selected_score")
    primary = mean("primary_score")
    best = mean("best_candidate_score")
    best_eligible = mean("best_eligible_candidate_score")
    return {
        "cases": len(rows),
        "available": True,
        "computed_after_forecast": True,
        "passed_to_forecaster": False,
        "mean_selected_rcrps": selected,
        "mean_primary_rcrps": primary,
        "mean_hindsight_best_candidate_rcrps": best,
        "mean_hindsight_best_eligible_candidate_rcrps": best_eligible,
        "mean_uplift_vs_primary_rcrps": primary - selected,
        "mean_oracle_headroom_rcrps": selected - best,
        "mean_selector_regret_among_eligible_rcrps": selected - best_eligible,
        "selected_primary_cases": sum(
            bool(row["selected_primary"]) for row in rows),
        "selected_hindsight_best_cases": sum(
            bool(row["selected_hindsight_best"]) for row in rows),
        "selected_best_eligible_cases": sum(
            bool(row["selected_best_eligible"]) for row in rows),
        "selected_role_counts": {
            role: sum(str(row.get("selected_role") or "unknown") == role
                      for row in rows)
            for role in sorted({str(row.get("selected_role") or "unknown")
                                for row in rows})
        },
        "primary_immutability_failures": sum(
            row["primary_forecast_unchanged"] is not True for row in rows),
        "automation_eligible_cases": sum(
            row["automation_eligible"] is True for row in rows),
    }


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
    parser.add_argument("--base-url", default="https://api.engy.ai/v1",
                        help="OpenAI-compatible endpoint (recorded in usage).")
    parser.add_argument("--api-key-env", default="ENGY_API_KEY",
                        choices=["ENGY_API_KEY", "OPENROUTER_API_KEY",
                                 "CHUTES_API_KEY"])
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds per task (official: 5)")
    parser.add_argument(
        "--seed-start", type=int, default=1,
        help="First official task seed (default: 1). Use a preregistered "
             "untouched seed range for held-out evaluation.",
    )
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
        "--task-name", action="append", default=[],
        help="Run this exact task class; repeat for a small explicit shard. "
             "Mutually exclusive with --task-filter.",
    )
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
    parser.add_argument(
        "--case-memory-mb", type=int, default=4096,
        help="Per-case process-tree resident-memory ceiling in MiB "
             "(default: 4096; 0 disables)",
    )
    parser.add_argument(
        "--case-timeout-seconds", type=int, default=900,
        help="Hard wall-clock limit for one disposable task/seed process",
    )
    parser.add_argument(
        "--min-free-memory-mb", type=int, default=2048,
        help="Refuse or stop safely before a case below this available RAM",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore the atomic per-case checkpoint and evaluate every case",
    )
    parser.add_argument(
        "--mcp-profile", default=None,
        choices=["core", "describe", "evidence", "mega", "full"],
        help="MCP surface for gnomon-mcp; defaults to GNOMON_MCP_PROFILE "
             "or full. Evidence host-binds the first valid forecast artifact.",
    )
    parser.add_argument(
        "--mcp-output-role", default="canonical",
        choices=["canonical", "immutable_primary", "llm_candidate_shadow",
                 "publication_best_effort", "prior_compromise_shadow"],
        help="gnomon-mcp Evidence only. canonical scores Gnomon's public "
             "artifact trajectory, which may be context-conditioned. "
             "immutable_primary is a diagnostic that ignores the public "
             "context recommendation and scores the preserved primary. "
             "publication_best_effort uses the product "
             "publication contract; llm_candidate_shadow scores the separately "
             "sealed, prior_assisted LLM candidate for evaluation; it is "
             "never an automation-eligible product publication. "
             "prior_compromise_shadow scores a fixed, equal-weight midpoint "
             "between the immutable primary and one eligible sealed prior; "
             "it is a preregisterable diagnostic, never product authority.",
    )
    parser.add_argument(
        "--mcp-candidate-paths", type=int, default=None,
        help="Evidence host candidate inference budget (3-16 paths). The "
             "default remains horizon-aware (4-5). This changes compute and "
             "candidate stability, never support or automation authority.",
    )
    parser.add_argument(
        "--mcp-candidate-history-rows", type=int, default=None,
        help="Evidence host history budget (8-256 rows) for controlled "
             "ablations. Default is horizon-aware and capped at 256.",
    )
    parser.add_argument(
        "--mcp-candidate-temporal-facts",
        action=argparse.BooleanOptionalAction, default=True,
        help="Include compact deterministic full-history temporal facts in "
             "the Evidence candidate prompt (default: enabled).",
    )
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
