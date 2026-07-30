from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .contracts import AionError


def _common_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input")
    parser.add_argument("--time", required=True, dest="time_column")
    parser.add_argument("--target", required=True, dest="target_column")
    parser.add_argument("--series", dest="series_column")
    parser.add_argument("--frequency")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aion", description="Evidence-backed local forecasting")
    parser.add_argument("--version", action="version", version="aion 0.2.0")
    subcommands = parser.add_subparsers(dest="command", required=True)

    capability_parser = subcommands.add_parser("capabilities", help="Report implemented capabilities")
    capability_parser.add_argument("--output", choices=["json"], default="json")

    inspect_parser = subcommands.add_parser("inspect", help="Validate a temporal dataset")
    _common_input(inspect_parser)
    inspect_parser.add_argument("--seasonal-period", type=int)

    forecast_parser = subcommands.add_parser("forecast", help="Run an evaluated forecast")
    _common_input(forecast_parser)
    forecast_parser.add_argument("--horizon", required=True, type=int)
    forecast_parser.add_argument("--output", default="aion-output")
    forecast_parser.add_argument("--minimum-baseline-improvement", type=float, default=0.02)
    forecast_parser.add_argument(
        "--threshold", type=float,
        help="Report when and how likely the forecast crosses this value",
    )
    forecast_parser.add_argument(
        "--config", default=None,
        help="Path to aion.yaml config file (models, ensemble, meta-model, backends)",
    )
    forecast_parser.add_argument(
        "--context", dest="context_file",
        help="Validated context-events JSON file (output of `aion context validate`)",
    )
    forecast_parser.add_argument(
        "--project", default=None,
        help="Register this forecast in a project for ongoing tracking and scoring",
    )
    forecast_parser.add_argument("--covariates", help="Point-in-time covariates CSV")
    forecast_parser.add_argument(
        "--covariate-mapping",
        help="Comma-separated name:type:future_known entries",
    )
    forecast_parser.add_argument("--covariate-time", default="timestamp")
    forecast_parser.add_argument("--covariate-known-at", default="known_at")
    forecast_parser.add_argument("--covariate-series")
    forecast_parser.add_argument("--seasonal-period", type=int)
    forecast_parser.add_argument("--strict-abstention", action="store_true")
    forecast_parser.add_argument("--selection-strategy", choices=("best", "ensemble"), default="best")
    forecast_parser.add_argument("--ensemble", action="store_true", help=argparse.SUPPRESS)
    forecast_parser.add_argument("--multivariate", action="store_true")

    covariate_parser = subcommands.add_parser(
        "covariates", help="Guide and validate point-in-time covariate data"
    )
    covariate_commands = covariate_parser.add_subparsers(
        dest="covariate_command", required=True
    )
    covariate_guide = covariate_commands.add_parser(
        "guide", help="Report required format and temporal constraints"
    )
    _common_input(covariate_guide)
    covariate_guide.add_argument("--horizon", required=True, type=int)
    covariate_validate = covariate_commands.add_parser(
        "validate", help="Validate covariate coverage at every backtest cutoff"
    )
    _common_input(covariate_validate)
    covariate_validate.add_argument("--horizon", required=True, type=int)
    covariate_validate.add_argument("--covariates", required=True)
    covariate_validate.add_argument("--covariate-mapping", required=True)
    covariate_validate.add_argument("--covariate-time", default="timestamp")
    covariate_validate.add_argument("--covariate-known-at", default="known_at")
    covariate_validate.add_argument("--covariate-series")

    context_parser = subcommands.add_parser(
        "context", help="LLM context-investigation workflow (prompt out, validation in)"
    )
    context_commands = context_parser.add_subparsers(dest="context_command", required=True)
    prompt_parser = context_commands.add_parser(
        "prompt", help="Emit the Aion-owned extraction prompt for permitted documents"
    )
    prompt_parser.add_argument("--file", action="append", required=True, dest="files")
    prompt_parser.add_argument("--series", action="append", default=[], dest="series_names")
    prompt_parser.add_argument("--timezone", default="+00:00")
    validate_parser = context_commands.add_parser(
        "validate", help="Ground and validate an LLM context response into typed events"
    )
    validate_parser.add_argument("--response", required=True)
    validate_parser.add_argument("--file", action="append", required=True, dest="files")

    mcp_parser = subcommands.add_parser("mcp", help="Model Context Protocol server")
    mcp_commands = mcp_parser.add_subparsers(dest="mcp_command", required=True)
    mcp_commands.add_parser("serve", help="Serve Aion tools over stdio MCP")

    tsfm_parser = subcommands.add_parser(
        "tsfm", help="Manage TSFM sandbox environments (isolated venvs per model)"
    )
    tsfm_commands = tsfm_parser.add_subparsers(dest="tsfm_command", required=True)

    tsfm_list = tsfm_commands.add_parser(
        "list", help="List available, installed (in-process), and sandboxed TSFMs"
    )

    tsfm_install = tsfm_commands.add_parser(
        "install", help="Create a sandboxed venv for a TSFM (isolated deps)"
    )
    tsfm_install.add_argument("name", help="TSFM adapter name (e.g. chronos_bolt_mini)")

    tsfm_remove = tsfm_commands.add_parser(
        "remove", help="Remove a TSFM sandbox venv"
    )
    tsfm_remove.add_argument("name", help="TSFM adapter name")

    tsfm_install_all = tsfm_commands.add_parser(
        "install-all", help="Create sandboxed venvs for all known TSFMs"
    )

    # --- Tracking ---
    track_parser = subcommands.add_parser(
        "track", help="Manage forecast projects, submit actuals, and track model performance"
    )
    track_commands = track_parser.add_subparsers(dest="track_command", required=True)

    track_list = track_commands.add_parser(
        "list", help="List forecasts in a project (or all projects)"
    )
    track_list.add_argument("--project", default=None, help="Filter by project name")
    track_list.add_argument("--limit", type=int, default=50)

    track_actuals = track_commands.add_parser(
        "actuals", help="Submit actual values and score all unscored forecasts in a project"
    )
    track_actuals.add_argument("--project", required=True)
    track_actuals.add_argument("--file", required=True, help="CSV file with actual values")

    track_score = track_commands.add_parser(
        "score", help="Score a single forecast against actuals"
    )
    track_score.add_argument("--forecast-id", required=True)
    track_score.add_argument("--file", required=True, help="CSV file with actual values")

    track_compare = track_commands.add_parser(
        "compare", help="Compare two scored forecasts"
    )
    track_compare.add_argument("--a", required=True, dest="forecast_a")
    track_compare.add_argument("--b", required=True, dest="forecast_b")

    track_perf = track_commands.add_parser(
        "performance", help="Show model performance history for a project"
    )
    track_perf.add_argument("--project", required=True)
    track_perf.add_argument("--model", default=None, help="Filter by model name")

    track_leaderboard = track_commands.add_parser(
        "leaderboard", help="Show ranked model performance for a project"
    )
    track_leaderboard.add_argument("--project", required=True)

    track_due = track_commands.add_parser(
        "due", help="List open forecasts whose horizon has completed"
    )
    track_due.add_argument("--project", default=None)

    track_decision = track_commands.add_parser(
        "decision", help="Record and resolve decisions supported by forecasts"
    )
    decision_commands = track_decision.add_subparsers(dest="decision_command", required=True)
    decision_record = decision_commands.add_parser("record")
    decision_record.add_argument("--decision-id", required=True)
    decision_record.add_argument("--project", required=True)
    decision_record.add_argument("--forecast-id", required=True)
    decision_record.add_argument("--action", required=True)
    decision_record.add_argument("--expected-outcome", required=True)
    decision_resolve = decision_commands.add_parser("resolve")
    decision_resolve.add_argument("--decision-id", required=True)
    decision_resolve.add_argument("--actual-outcome", required=True)
    decision_resolve.add_argument("--correct", required=True, choices=["true", "false"])
    decision_list = decision_commands.add_parser("list")
    decision_list.add_argument("--project", default=None)

    track_export = track_commands.add_parser("export", help="Export registry metadata as JSON")
    track_export.add_argument("--project", default=None)
    track_export.add_argument("--output", required=True)
    track_relocate = track_commands.add_parser("relocate", help="Update a moved artifact path")
    track_relocate.add_argument("--forecast-id", required=True)
    track_relocate.add_argument("--artifact-path", required=True)

    eval_parser = subcommands.add_parser(
        "eval", help="Compare agent runs with and without Aion"
    )
    eval_commands = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_compare = eval_commands.add_parser("compare")
    eval_compare.add_argument("--baseline", required=True, help="Control JSONL runs")
    eval_compare.add_argument("--treatment", required=True, help="Aion-enabled JSONL runs")

    return parser


def _read_documents(paths: list[str]):
    from .workflows import DocumentRef

    documents = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise AionError("DOCUMENT_NOT_FOUND", f"Document does not exist: {path}")
        documents.append(DocumentRef(
            name=path.name,
            content=path.read_text(encoding="utf-8", errors="replace"),
            source_type="planning_file",
            reference=str(path.resolve()),
        ))
    return documents


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "mcp":
            from .mcp_server import serve

            return serve()
        if args.command == "tsfm":
            from .tsfm import available_tsfms, installed_tsfms
            from .tsfm_sandbox import (
                ensure_sandbox, remove_sandbox, list_sandboxes,
                sandbox_exists, TSFM_PIP_SPECS,
            )
            from .tsfm import TSFMUnavailable, TSFMError

            if args.tsfm_command == "list":
                payload = {
                    "available": available_tsfms(),
                    "installed_in_process": installed_tsfms(),
                    "sandboxed": list_sandboxes(),
                    "pip_specs": {
                        name: specs for name, specs in TSFM_PIP_SPECS.items()
                    },
                }
                print(json.dumps(payload, indent=2))
                return 0
            elif args.tsfm_command == "install":
                name = args.name
                if name not in TSFM_PIP_SPECS:
                    print(json.dumps({
                        "status": "error",
                        "error": {
                            "code": "UNKNOWN_TSFM",
                            "message": f"Unknown TSFM: {name}. Available: {available_tsfms()}",
                        },
                    }, indent=2), file=sys.stderr)
                    return 2
                print(f"Creating sandbox venv for {name}...", file=sys.stderr)
                try:
                    venv_dir = ensure_sandbox(name)
                    print(json.dumps({
                        "status": "ok",
                        "tsfm": name,
                        "sandbox_path": str(venv_dir),
                        "pip_specs": TSFM_PIP_SPECS[name],
                    }, indent=2))
                    return 0
                except (TSFMUnavailable, TSFMError) as exc:
                    print(json.dumps({
                        "status": "error",
                        "error": {"code": "SANDBOX_FAILED", "message": str(exc)},
                    }, indent=2), file=sys.stderr)
                    return 2
            elif args.tsfm_command == "remove":
                name = args.name
                remove_sandbox(name)
                print(json.dumps({"status": "ok", "removed": name}, indent=2))
                return 0
            elif args.tsfm_command == "install-all":
                results = {}
                for name in TSFM_PIP_SPECS:
                    print(f"Creating sandbox for {name}...", file=sys.stderr)
                    try:
                        ensure_sandbox(name)
                        results[name] = "ok"
                    except (TSFMUnavailable, TSFMError) as exc:
                        results[name] = f"failed: {exc}"
                print(json.dumps({"status": "complete", "results": results}, indent=2))
                return 0
        if args.command == "track":
            from .tracking import TrackingStore
            store = TrackingStore()

            if args.track_command == "list":
                forecasts = store.list_forecasts(
                    project=args.project, limit=args.limit,
                )
                payload = [
                    {
                        "forecast_id": f.forecast_id,
                        "project": f.project,
                        "series": f.series,
                        "model": f.selected_model,
                        "support": f.support,
                        "horizon": f.horizon,
                        "threshold": f.threshold,
                        "scored": f.scored,
                        "mase": f.mase,
                        "drift": f.drift_flag,
                        "created_at": f.created_at,
                    }
                    for f in forecasts
                ]
                print(json.dumps(payload, indent=2))
                return 0

            elif args.track_command == "actuals":
                results = store.submit_actuals_csv(args.project, args.file)
                payload = [
                    {
                        "forecast_id": r.forecast_id,
                        "mase": r.mase,
                        "mape": r.mape,
                        "bias": r.bias,
                        "coverage": r.coverage,
                        "threshold_accuracy": r.threshold_accuracy,
                        "drift": r.drift_flag,
                    }
                    for r in results
                ]
                print(json.dumps({
                    "status": "ok",
                    "project": args.project,
                    "scored": len(results),
                    "results": payload,
                }, indent=2))
                return 0

            elif args.track_command == "score":
                record = store.get_forecast(args.forecast_id)
                if record is None:
                    print(json.dumps({
                        "status": "error",
                        "error": {"code": "NOT_FOUND",
                                  "message": f"Forecast {args.forecast_id} not found"},
                    }, indent=2), file=sys.stderr)
                    return 2
                # Load actuals and score
                import csv as _csv
                from pathlib import Path as _Path
                actuals: list[tuple[str, float]] = []
                with open(_Path(args.file), encoding="utf-8-sig", newline="") as f:
                    reader = _csv.DictReader(f)
                    cols = reader.fieldnames or []
                    ts_col = cols[0] if cols else "timestamp"
                    val_col = cols[1] if len(cols) > 1 else "value"
                    for row in reader:
                        try:
                            actuals.append((row[ts_col], float(row[val_col])))
                        except (ValueError, TypeError):
                            continue
                # Load forecast
                fc_path = _Path(record.artifact_path) / "forecast.csv"
                fc_data = [
                    row for row in store._load_forecast_csv(fc_path)
                    if row.get("series", "__default__") == record.series
                ]
                actual_map = {
                    store._normalise_timestamp(ts): val for ts, val in actuals
                }
                matched_a, matched_p, matched_q10, matched_q90 = [], [], [], []
                for entry in fc_data:
                    ts = store._normalise_timestamp(entry["timestamp"])
                    if ts in actual_map:
                        matched_a.append(actual_map[ts])
                        matched_p.append(entry["point"])
                        if "q10" in entry: matched_q10.append(entry["q10"])
                        if "q90" in entry: matched_q90.append(entry["q90"])
                if not matched_a:
                    print(json.dumps({
                        "status": "error",
                        "error": {"code": "NO_MATCH",
                                  "message": "No matching actuals found for forecast timestamps"},
                    }, indent=2), file=sys.stderr)
                    return 2
                if len(matched_a) != len(fc_data):
                    print(json.dumps({
                        "status": "error",
                        "error": {"code": "INCOMPLETE_ACTUALS",
                                  "message": (
                                      f"Matched {len(matched_a)} of {len(fc_data)} "
                                      "forecast timestamps"
                                  )},
                    }, indent=2), file=sys.stderr)
                    return 2
                result = store.score_forecast(
                    args.forecast_id, matched_a, matched_p,
                    q10=matched_q10 or None, q90=matched_q90 or None,
                    threshold=record.threshold,
                    predicted_above=(
                        [point > record.threshold for point in matched_p]
                        if record.threshold is not None else None
                    ),
                )
                print(json.dumps({
                    "forecast_id": result.forecast_id,
                    "mase": result.mase,
                    "mape": result.mape,
                    "bias": result.bias,
                    "coverage": result.coverage,
                    "threshold_accuracy": result.threshold_accuracy,
                    "drift": result.drift_flag,
                }, indent=2))
                return 0

            elif args.track_command == "compare":
                result = store.compare(args.forecast_a, args.forecast_b)
                print(json.dumps(result, indent=2))
                return 0

            elif args.track_command == "performance":
                if args.model:
                    history = store.model_performance(args.project, args.model)
                    payload = history
                else:
                    lb = store.leaderboard(args.project)
                    payload = [
                        {
                            "model": m.model,
                            "count": m.count,
                            "avg_mase": m.avg_mase,
                            "avg_mape": m.avg_mape,
                            "avg_bias": m.avg_bias,
                            "avg_coverage": m.avg_coverage,
                            "avg_threshold_accuracy": m.avg_threshold_accuracy,
                            "last_mase": m.last_mase,
                            "last_scored": m.last_scored,
                        }
                        for m in lb
                    ]
                print(json.dumps(payload, indent=2))
                return 0

            elif args.track_command == "leaderboard":
                lb = store.leaderboard(args.project)
                print(f"\n  Model Leaderboard: {args.project}")
                print(f"  {'Model':25s} {'Count':>5s} {'MASE':>7s} {'MAPE':>7s} {'Bias':>8s} {'Coverage':>9s} {'Last':>7s}")
                print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*7}")
                for m in lb:
                    mase_s = f"{m.avg_mase:.3f}" if m.avg_mase is not None else "N/A"
                    mape_s = f"{m.avg_mape:.1f}%" if m.avg_mape is not None else "N/A"
                    bias_s = f"{m.avg_bias:+.2f}" if m.avg_bias is not None else "N/A"
                    cov_s = f"{m.avg_coverage:.0%}" if m.avg_coverage is not None else "N/A"
                    last_s = f"{m.last_mase:.3f}" if m.last_mase is not None else "N/A"
                    print(f"  {m.model:25s} {m.count:>5d} {mase_s:>7s} {mape_s:>7s} {bias_s:>8s} {cov_s:>9s} {last_s:>7s}")
                print()
                return 0

            elif args.track_command == "due":
                print(json.dumps(store.due_forecasts(args.project), indent=2))
                return 0

            elif args.track_command == "decision":
                if args.decision_command == "record":
                    decision = store.record_decision(
                        args.decision_id, args.project, args.forecast_id,
                        args.action, args.expected_outcome,
                    )
                    print(json.dumps(decision.__dict__, indent=2))
                    return 0
                if args.decision_command == "resolve":
                    decision = store.resolve_decision(
                        args.decision_id, args.actual_outcome, args.correct == "true",
                    )
                    print(json.dumps(decision.__dict__, indent=2))
                    return 0
                decisions = store.list_decisions(args.project)
                print(json.dumps([item.__dict__ for item in decisions], indent=2))
                return 0

            elif args.track_command == "export":
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(store.export_snapshot(args.project), indent=2) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps({"status": "ok", "output": str(output.resolve())}, indent=2))
                return 0

            elif args.track_command == "relocate":
                record = store.relocate_artifact(args.forecast_id, args.artifact_path)
                print(json.dumps(record.__dict__, indent=2))
                return 0

        if args.command == "eval":
            from .agent_eval import compare_runs
            print(json.dumps(compare_runs(args.baseline, args.treatment), indent=2))
            return 0

        if args.command == "capabilities":
            from .runtime import capabilities

            payload = capabilities()
        elif args.command == "inspect":
            from .runtime import inspect_dataset

            payload = inspect_dataset(
                args.input, time_column=args.time_column, target_column=args.target_column,
                series_column=args.series_column, frequency=args.frequency,
                seasonal_period=args.seasonal_period,
            )
        elif args.command == "covariates":
            from .covariates import covariate_guide, validate_covariate_file
            common = {
                "time_column": args.time_column, "target_column": args.target_column,
                "series_column": args.series_column, "frequency": args.frequency,
                "horizon": args.horizon,
            }
            if args.covariate_command == "guide":
                payload = covariate_guide(args.input, **common)
            else:
                payload = validate_covariate_file(
                    args.input, args.covariates, args.covariate_mapping, **common,
                    covariate_time_column=args.covariate_time,
                    covariate_known_at_column=args.covariate_known_at,
                    covariate_series_column=args.covariate_series,
                )
        elif args.command == "context":
            from .workflows import build_context_investigation_prompt, parse_context_response

            documents = _read_documents(args.files)
            if args.context_command == "prompt":
                payload = build_context_investigation_prompt(
                    documents, args.series_names, args.timezone
                )
            else:
                response_path = Path(args.response).expanduser()
                if not response_path.is_file():
                    raise AionError("RESPONSE_NOT_FOUND", f"Response file does not exist: {response_path}")
                try:
                    raw = json.loads(response_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise AionError("INVALID_RESPONSE", f"LLM response is not valid JSON: {exc}") from exc
                payload = parse_context_response(raw, documents)
        else:
            from .context import load_events_file
            from .runtime import forecast
            from .toolspec import forecast_summary
            from .config import load_config

            events = load_events_file(args.context_file) if args.context_file else None
            from .covariates import load_covariates
            covariates = None
            if args.covariates:
                if not args.covariate_mapping:
                    raise AionError(
                        "MISSING_COVARIATE_MAPPING",
                        "--covariate-mapping is required with --covariates.",
                    )
                covariates = load_covariates(
                    args.covariates, args.covariate_mapping,
                    time_column=args.covariate_time,
                    known_at_column=args.covariate_known_at,
                    series_column=args.covariate_series,
                )
            config = load_config(getattr(args, "config", None))
            artifact, path = forecast(
                args.input, time_column=args.time_column, target_column=args.target_column,
                series_column=args.series_column, frequency=args.frequency, horizon=args.horizon,
                output=args.output, minimum_baseline_improvement=args.minimum_baseline_improvement,
                context_events=events, threshold=args.threshold,
                covariates=covariates,
                config=config, strict_abstention=args.strict_abstention,
                seasonal_period=args.seasonal_period,
                selection_strategy="ensemble" if args.ensemble else args.selection_strategy,
                multivariate=args.multivariate,
            )
            payload = forecast_summary(artifact, path)

            # Auto-register in tracking store if --project is set
            if getattr(args, "project", None):
                from .tracking import register_artifact
                register_artifact(artifact, args.project, str(path))
                print(f"Registered forecast {artifact.forecast_id} in project '{args.project}'", file=sys.stderr)

        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except AionError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        error = AionError("TRACKING_ERROR", str(exc))
        print(json.dumps(error.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
