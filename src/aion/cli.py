from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .contracts import AionError


def _common_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="Path to a CSV/TSV/JSON/JSONL/Parquet/Excel file, or "
             "store:<dataset> to read the bitemporal store",
    )
    # Not required: inferred from the file when the schema leaves no
    # choice, and disclosed as an assumption when it is. `aion forecast
    # data.csv` on an obvious two-column file used to fail with a bare
    # argparse error, which was the most common first-run failure there is.
    parser.add_argument(
        "--time", dest="time_column",
        help="Timestamp column. Inferred when exactly one column parses as "
             "timestamps.",
    )
    parser.add_argument(
        "--target", dest="target_column",
        help="Numeric column to model. Inferred when exactly one non-time "
             "column parses as numbers.",
    )
    parser.add_argument(
        "--series", dest="series_column",
        help="Column identifying the series in a panel file; omit for a "
             "single series",
    )
    parser.add_argument(
        "--frequency",
        help="Explicit time grid (h, D, W, MS, …). Inferred from the "
             "observed step when omitted.",
    )


def _resolve_schema(args) -> list[str]:
    """Fill in ``--time``/``--target`` from the file, or say why it cannot.

    Returns the assumptions made, so the caller can disclose them. An
    inference nobody is told about is a guess.
    """
    if args.time_column and args.target_column:
        return []
    source = str(getattr(args, "input", ""))
    if source.startswith("store:"):
        raise AionError(
            "INVALID_ARGUMENTS",
            "--time and --target are required for store:<dataset> inputs; a "
            "stored dataset has no file header to infer from.",
            {"input": source},
            repair_options=[{
                "action": "supply_arguments",
                "description": "Pass --time and --target explicitly, or run "
                               "`aion store list` to see a dataset's variables.",
                "arguments": ["--time", "--target"],
            }],
        )
    from .data import infer_schema_columns

    inferred = infer_schema_columns(source)
    assumptions: list[str] = []
    for flag, key, candidates_key in (
        ("--time", "time", "time_candidates"),
        ("--target", "target", "target_candidates"),
    ):
        attribute = "time_column" if key == "time" else "target_column"
        if getattr(args, attribute):
            continue
        chosen = inferred[key]
        if chosen is None:
            candidates = inferred[candidates_key]
            raise AionError(
                "AMBIGUOUS_SCHEMA",
                f"{flag} was not supplied and cannot be inferred: "
                + (f"{len(candidates)} columns qualify ({', '.join(candidates)})."
                   if candidates else "no column qualifies."),
                {"argument": flag, "candidates": candidates,
                 "columns_examined": inferred["time_candidates"]
                 + inferred["target_candidates"]},
                repair_options=[{
                    "action": "supply_arguments",
                    "description": f"Pass {flag} explicitly."
                                   + (f" Candidates: {', '.join(candidates)}."
                                      if candidates else ""),
                    "arguments": [flag],
                }],
            )
        setattr(args, attribute, chosen)
        assumptions.append(
            f"{flag} was not supplied; inferred as {chosen!r}, the only "
            f"column in the input that qualifies."
        )
    return assumptions


def build_parser() -> argparse.ArgumentParser:
    parser = _StructuredArgumentParser(prog="aion", description="Evidence-backed local forecasting")
    parser.add_argument("--version", action="version", version="aion 0.4.0")
    subcommands = parser.add_subparsers(dest="command", required=True)

    capability_parser = subcommands.add_parser("capabilities", help="Report implemented capabilities")
    capability_parser.add_argument("--output", choices=["json"], default="json")

    inspect_parser = subcommands.add_parser("inspect", help="Validate a temporal dataset")
    _common_input(inspect_parser)
    inspect_parser.add_argument("--seasonal-period", type=int)

    forecast_parser = subcommands.add_parser("forecast", help="Run an evaluated forecast")
    _common_input(forecast_parser)
    forecast_parser.add_argument(
        "--horizon", type=int,
        help="Periods to forecast. Defaults to one seasonal period of the "
             "inferred grid, disclosed as an assumption.",
    )
    forecast_parser.add_argument("--output", default="aion-output")
    forecast_parser.add_argument(
        "--minimum-baseline-improvement", type=float, default=0.02,
        help="Relative margin a candidate must beat the strongest baseline "
             "by to be selected (default 0.02). Must be >= 0: a negative "
             "margin inverts the mandated-baseline gate.",
    )
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
    forecast_parser.add_argument(
        "--as-of", dest="as_of",
        help="Replay instant: use only data known at or before this ISO timestamp",
    )
    forecast_parser.add_argument(
        "--store-path", dest="store_path",
        help="Override the temporal-store path (for store:<dataset> inputs)",
    )
    forecast_parser.add_argument(
        "--repair", choices=("off", "safe", "aggressive"), default="safe",
        help="Messy-data handling: off rejects anything non-strict; safe "
             "(default) normalises cell text with disclosure; aggressive "
             "additionally fills gaps, snaps jittered timestamps, and "
             "resolves conflicts — capped and reported as warnings",
    )
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

    investigate_parser = subcommands.add_parser(
        "investigate", help="What changed? Changepoints, regimes, anomalies, ranked explanations"
    )
    _common_input(investigate_parser)
    investigate_parser.add_argument("--context", dest="context_file",
                                    help="Validated context-events JSON to rank as concurrent events")
    investigate_parser.add_argument("--as-of", dest="as_of")
    investigate_parser.add_argument("--output", default="aion-output")
    investigate_parser.add_argument("--store-path", dest="store_path")

    route_parser = subcommands.add_parser(
        "route", help="Which method for this task on this data? Disclosed, advisory, recorded"
    )
    _common_input(route_parser)
    route_parser.add_argument("--task", default="forecast",
                              choices=["forecast", "detect_anomalies"])
    route_parser.add_argument("--horizon", type=int, default=1)
    route_parser.add_argument("--project",
                              help="Tracking project: consults the realised-performance "
                                   "prior and records the decision")
    route_parser.add_argument("--as-of", dest="as_of")
    route_parser.add_argument("--store-path", dest="store_path")

    detect_parser = subcommands.add_parser(
        "detect", help="What is abnormal? Graded detectors, disclosed selection, flagged anomalies"
    )
    _common_input(detect_parser)
    detect_parser.add_argument("--threshold", type=float,
                               help="Detection threshold on standardised scores (default 3.5)")
    detect_parser.add_argument("--labels",
                               help="Comma-separated ISO timestamps of known anomalies "
                                    "(selection then uses label F1)")
    detect_parser.add_argument("--as-of", dest="as_of")
    detect_parser.add_argument("--output", default="aion-output")
    detect_parser.add_argument("--store-path", dest="store_path")

    decide_parser = subcommands.add_parser(
        "decide", help="What should we do? Scenario risk, feasible actions, expected utility or abstention"
    )
    _common_input(decide_parser)
    decide_parser.add_argument("--horizon", required=True, type=int)
    decide_parser.add_argument("--threshold", required=True, type=float)
    decide_parser.add_argument(
        "--actions", required=True,
        help=f"JSON list of action objects, or @path/to/actions.json. Each "
             f"object needs a 'name' and may carry 'feasible' (bool) and "
             f"'residual_risk' (number). Example: {ACTIONS_EXAMPLE}",
    )
    decide_parser.add_argument("--utilities",
                               help="JSON {action: {exceed: x, no_exceed: y}}, or @file")
    decide_parser.add_argument("--max-acceptable-risk", type=float, dest="max_acceptable_risk")
    decide_parser.add_argument("--series-name", dest="series_name")
    decide_parser.add_argument("--project", help="Record a DecisionArtifact in this tracking project")
    decide_parser.add_argument("--as-of", dest="as_of")
    decide_parser.add_argument("--output", default="aion-output")
    decide_parser.add_argument("--store-path", dest="store_path")

    monitor_parser = subcommands.add_parser(
        "monitor", help="When should we intervene? Sequential risk and a cost-aware alert rule"
    )
    _common_input(monitor_parser)
    monitor_parser.add_argument("--horizon", required=True, type=int)
    monitor_parser.add_argument("--threshold", required=True, type=float)
    monitor_parser.add_argument("--alert-cost", type=float, dest="alert_cost")
    monitor_parser.add_argument("--miss-cost", type=float, dest="miss_cost")
    monitor_parser.add_argument("--project")
    monitor_parser.add_argument("--as-of", dest="as_of")
    monitor_parser.add_argument("--output", default="aion-output")
    monitor_parser.add_argument("--store-path", dest="store_path")

    plan_parser = subcommands.add_parser(
        "plan", help="Experimental: compile, validate, and execute TemporalPlans"
    )
    plan_commands = plan_parser.add_subparsers(dest="plan_command", required=True)
    plan_compile = plan_commands.add_parser("compile")
    plan_compile.add_argument("--task-type", required=True,
                              choices=["forecast", "investigate_change", "decide", "monitor"])
    plan_compile.add_argument("--params", required=True,
                              help="JSON object of macro parameters, or @file")
    plan_validate = plan_commands.add_parser("validate")
    plan_validate.add_argument("--plan", required=True, help="Plan JSON, or @file")
    plan_execute = plan_commands.add_parser("execute")
    plan_execute.add_argument("--plan", required=True, help="Plan JSON, or @file")
    plan_execute.add_argument("--output", default="aion-output")
    plan_execute.add_argument("--as-of", dest="as_of")
    plan_execute.add_argument("--store-path", dest="store_path")

    ingest_parser = subcommands.add_parser(
        "ingest", help="Append observations (as vintages) to the bitemporal store"
    )
    ingest_parser.add_argument("input")
    ingest_parser.add_argument("--dataset", required=True)
    ingest_parser.add_argument("--time", required=True, dest="time_column")
    ingest_parser.add_argument("--target", required=True, dest="target_column")
    ingest_parser.add_argument("--series", dest="series_column")
    ingest_parser.add_argument(
        "--known-at", dest="known_at_column",
        help="Column stating when each value became known; omitted = assumed known at valid time",
    )
    ingest_parser.add_argument("--variable", help="Variable name (default: target column name)")
    ingest_parser.add_argument("--store-path", dest="store_path")

    store_parser = subcommands.add_parser("store", help="Inspect the bitemporal store")
    store_commands = store_parser.add_subparsers(dest="store_command", required=True)
    store_list = store_commands.add_parser("list", help="List ingested datasets")
    store_list.add_argument("--store-path", dest="store_path")

    status_parser = subcommands.add_parser(
        "status", help="Open forecasts, due horizons, unresolved decisions, realised performance"
    )
    status_parser.add_argument("--project", default=None)

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
    # Explicit, like every other input path. Guessing positionally turned a
    # perfectly good `requests,timestamp,host` export into a silent
    # `{"scored": 0}` with no error and no diagnosis.
    track_actuals.add_argument(
        "--time", dest="actuals_time",
        help="Timestamp column in the actuals file (inferred when obvious)",
    )
    track_actuals.add_argument(
        "--target", dest="actuals_target",
        help="Value column in the actuals file (inferred when obvious)",
    )
    track_actuals.add_argument(
        "--series", dest="actuals_series",
        help="Series column, for multi-series projects",
    )

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
    track_leaderboard.add_argument("--task",
                                   help="Restrict to one task (e.g. forecast, detect_anomalies)")

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

    track_outcome = track_commands.add_parser(
        "outcome", help="Resolve a DecisionArtifact with realised results (regret scoring)"
    )
    track_outcome.add_argument("--decision-id", required=True)
    track_outcome.add_argument("--realised-scenario")
    track_outcome.add_argument("--realised-utilities", help="JSON {action: payoff}, or @file")
    track_outcome.add_argument("--violations", help="JSON list of violated constraints, or @file")
    track_outcome.add_argument("--note")

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
    eval_episodes = eval_commands.add_parser(
        "episodes", help="Run the built-in episode suite with the honest reference policy"
    )
    eval_episodes.add_argument("--workdir", required=True,
                               help="Directory for generated worlds and artifacts")
    eval_episodes.add_argument("--trials", type=int, default=1)
    eval_episodes.add_argument("--jsonl", help="Also write graded rows for `aion eval compare`")

    return parser


def _json_argument(raw: str | None, *, argument: str | None = None):
    """Parse a JSON CLI argument, naming which one failed.

    ``INVALID_JSON_ARGUMENT`` used to say only that "an argument" was not
    valid JSON, with empty repair options — on a command taking four of
    them, that is not a diagnosis.
    """
    if raw is None:
        return None
    label = f"--{argument}" if argument else "argument"
    if raw.startswith("@"):
        path = Path(raw[1:]).expanduser()
        if not path.is_file():
            raise AionError(
                "ARGUMENT_FILE_NOT_FOUND", f"File does not exist: {path}",
                {"argument": label, "path": str(path)},
            )
        raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AionError(
            "INVALID_JSON_ARGUMENT",
            f"{label} is not valid JSON: {exc}",
            {"argument": label, "supplied": raw[:200],
             "parse_error": str(exc)},
        ) from exc


#: The shape `--actions` takes, matching the MCP schema exactly. The CLI
#: help used to describe "a JSON list of actions", which read as
#: `["scale_up", "do_nothing"]` and crashed with an unhandled TypeError
#: deep in the operator.
ACTIONS_EXAMPLE = (
    '[{"name": "scale_up", "feasible": true, "residual_risk": 0.1}, '
    '{"name": "do_nothing"}]'
)


def _validate_actions(actions: Any) -> list[dict[str, Any]]:
    """Check `--actions` against the MCP schema before the operator sees it."""
    if not isinstance(actions, list):
        raise AionError(
            "INVALID_ACTIONS",
            f"--actions must be a JSON list of objects; got "
            f"{type(actions).__name__}.",
            {"example": ACTIONS_EXAMPLE, "supplied_type": type(actions).__name__},
        )
    problems: list[str] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            problems.append(
                f"item {index} is {type(action).__name__}, not an object with a "
                f"'name'"
            )
            continue
        if "name" not in action:
            problems.append(f"item {index} has no 'name'")
        if "feasible" in action and not isinstance(action["feasible"], bool):
            problems.append(f"item {index}: 'feasible' must be true or false")
        if "residual_risk" in action:
            try:
                float(action["residual_risk"])
            except (TypeError, ValueError):
                problems.append(f"item {index}: 'residual_risk' must be a number")
    if problems:
        raise AionError(
            "INVALID_ACTIONS",
            "--actions does not match the expected shape: "
            + "; ".join(problems) + ".",
            {"example": ACTIONS_EXAMPLE, "problems": problems},
        )
    return actions


def _parse_as_of(raw: str | None):
    if not raw:
        return None
    from .data import _parse_timestamp
    return _parse_timestamp(raw, 0)


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


def _disclose_assumptions(payload, assumptions: list[str]):
    """Attach CLI-level inferences to every result's support assessment.

    An inference the caller is not told about is a guess. These ride in the
    same `assumptions` list as `known_time_assumed`, so an agent reading the
    envelope finds them where it already looks.
    """
    if not assumptions or not isinstance(payload, dict):
        return payload
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return {**payload, "assumptions": assumptions}
    decorated = []
    for result in results:
        if not isinstance(result, dict):
            decorated.append(result)
            continue
        assessment = dict(result.get("support_assessment") or {})
        assessment["assumptions"] = list(assessment.get("assumptions", [])) + assumptions
        decorated.append({**result, "support_assessment": assessment})
    return {**payload, "results": decorated}


def _default_horizon(args) -> int:
    """One seasonal period of the input's own grid."""
    from .pipeline import load_stage
    from .temporal import detect_season

    loaded = load_stage(
        args.input, time_column=args.time_column,
        target_column=args.target_column,
        series_column=getattr(args, "series_column", None),
        frequency=getattr(args, "frequency", None),
        as_of=_parse_as_of(getattr(args, "as_of", None)),
        store_path=getattr(args, "store_path", None),
    )
    longest = max(loaded.groups.values(), key=len, default=[])
    season, _, _ = detect_season([item.value for item in longest], loaded.frequency)
    return max(1, int(season))


class _StructuredArgumentParser(argparse.ArgumentParser):
    """An argparse parser whose failures obey the error contract.

    A usage error used to exit(2) with plain text on stderr — no JSON
    envelope, no `code`, no `repair_options` — which made the single most
    common first-run failure the one error in the product that bypassed the
    contract entirely. Any agent wrapping the CLI had to special-case it.
    """

    def error(self, message: str):  # noqa: D102 - argparse override
        raise _ArgumentError(message, self.prog)


class _ArgumentError(Exception):
    def __init__(self, message: str, prog: str):
        super().__init__(message)
        self.message = message
        self.prog = prog


def _argument_error(exc: _ArgumentError) -> AionError:
    missing = _missing_arguments(exc.message)
    repairs = []
    if missing:
        repairs.append({
            "action": "supply_arguments",
            "description": (
                "Supply the missing argument(s): " + ", ".join(missing) + "."
            ),
            "arguments": missing,
        })
    repairs.append({
        "action": "show_usage",
        "description": f"Run `{exc.prog} --help` for the full argument list.",
    })
    return AionError(
        "INVALID_ARGUMENTS", exc.message,
        {"command": exc.prog, "missing_arguments": missing},
        repair_options=repairs,
    )


def _missing_arguments(message: str) -> list[str]:
    marker = "the following arguments are required:"
    if marker not in message:
        return []
    tail = message.split(marker, 1)[1]
    return [item.strip() for item in tail.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = None
    try:
        args = build_parser().parse_args(argv)
    except _ArgumentError as exc:
        print(json.dumps(_argument_error(exc).to_dict(), indent=2), file=sys.stderr)
        return 2
    schema_assumptions: list[str] = []
    try:
        if hasattr(args, "time_column"):
            schema_assumptions = _resolve_schema(args)
        if getattr(args, "horizon", "absent") is None:
            # One season ahead is the smallest horizon that can show a
            # seasonal pattern, and it is derivable from the data. Disclosed
            # like any other inference.
            args.horizon = _default_horizon(args)
            schema_assumptions.append(
                f"--horizon was not supplied; defaulted to {args.horizon}, "
                f"one seasonal period of the inferred grid."
            )
    except AionError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2
    try:
        if args.command == "investigate":
            from .context import load_events_file
            from .macros import investigate_change

            events = load_events_file(args.context_file) if args.context_file else None
            payload, path = investigate_change(
                args.input, time_column=args.time_column,
                target_column=args.target_column, series_column=args.series_column,
                frequency=args.frequency, as_of=_parse_as_of(args.as_of),
                context_events=events, output=args.output,
                store_path=args.store_path,
            )
            print(json.dumps(_disclose_assumptions(
                {**payload, "artifact_path": str(path)}, schema_assumptions,
            ), indent=2, allow_nan=False))
            return 0
        if args.command == "route":
            from .pipeline import load_stage
            from .router import route
            from .tracking import TrackingStore

            loaded = load_stage(
                args.input, time_column=args.time_column,
                target_column=args.target_column, series_column=args.series_column,
                frequency=args.frequency, as_of=_parse_as_of(args.as_of),
                store_path=args.store_path,
            )
            store = TrackingStore() if args.project else None
            decisions = [
                route(args.task, [item.value for item in items], loaded.frequency,
                      horizon=args.horizon, series=name, project=args.project,
                      store=store)
                for name, items in sorted(loaded.groups.items())
            ]
            print(json.dumps({"schema_version": "0.1", "decisions": decisions},
                             indent=2, allow_nan=False))
            return 0
        if args.command == "detect":
            from .macros import detect_anomalies

            labels = [item.strip() for item in args.labels.split(",")] if args.labels else None
            payload, path = detect_anomalies(
                args.input, time_column=args.time_column,
                target_column=args.target_column, series_column=args.series_column,
                frequency=args.frequency, as_of=_parse_as_of(args.as_of),
                threshold=args.threshold, labels=labels, output=args.output,
                store_path=args.store_path,
            )
            print(json.dumps(_disclose_assumptions(
                {**payload, "artifact_path": str(path)}, schema_assumptions,
            ), indent=2, allow_nan=False))
            return 0
        if args.command == "decide":
            from .macros import decide

            payload, path = decide(
                args.input, time_column=args.time_column,
                target_column=args.target_column, horizon=args.horizon,
                threshold=args.threshold,
                actions=_validate_actions(_json_argument(args.actions, argument="actions") or []),
                utilities=_json_argument(args.utilities, argument="utilities"),
                max_acceptable_risk=args.max_acceptable_risk,
                series_column=args.series_column, series_name=args.series_name,
                frequency=args.frequency, as_of=_parse_as_of(args.as_of),
                project=args.project,
                output=args.output, store_path=args.store_path,
            )
            print(json.dumps(_disclose_assumptions(
                {**payload, "artifact_path": str(path)}, schema_assumptions,
            ), indent=2, allow_nan=False))
            return 0
        if args.command == "status":
            from .tracking import TrackingStore
            print(json.dumps(TrackingStore().status(args.project), indent=2, allow_nan=False))
            return 0
        if args.command == "monitor":
            from .macros import monitor

            payload, path = monitor(
                args.input, time_column=args.time_column,
                target_column=args.target_column, horizon=args.horizon,
                threshold=args.threshold, alert_cost=args.alert_cost,
                miss_cost=args.miss_cost, series_column=args.series_column,
                frequency=args.frequency, as_of=_parse_as_of(args.as_of),
                project=args.project, output=args.output,
                store_path=args.store_path,
            )
            print(json.dumps(_disclose_assumptions(
                {**payload, "artifact_path": str(path)}, schema_assumptions,
            ), indent=2, allow_nan=False))
            return 0
        if args.command == "plan":
            from .toolspec import (
                _run_compile_task, _run_execute_plan, _run_validate_plan,
            )
            if args.plan_command == "compile":
                payload = _run_compile_task({
                    "task_type": args.task_type,
                    "params": _json_argument(args.params, argument="params"),
                })
            elif args.plan_command == "validate":
                payload = _run_validate_plan({"plan": _json_argument(args.plan, argument="plan")})
            else:
                payload = _run_execute_plan({
                    "plan": _json_argument(args.plan, argument="plan"),
                    "output_dir": args.output,
                    "as_of": args.as_of,
                    "store_path": args.store_path,
                })
            print(json.dumps(payload, indent=2, allow_nan=False))
            return 0
        if args.command == "ingest":
            from .temporal_store import TemporalStore

            report = TemporalStore(args.store_path).ingest_csv(
                args.input, dataset=args.dataset,
                time_column=args.time_column, target_column=args.target_column,
                series_column=args.series_column,
                known_at_column=args.known_at_column,
                variable=args.variable,
            )
            print(json.dumps(report.to_dict(), indent=2))
            return 0
        if args.command == "store":
            from .temporal_store import TemporalStore

            print(json.dumps({
                "schema_version": "0.1",
                "datasets": TemporalStore(args.store_path).list_datasets(),
            }, indent=2))
            return 0
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
                results = store.submit_actuals_csv(
                    args.project, args.file,
                    time_column=args.actuals_time,
                    target_column=args.actuals_target,
                    series_column=args.actuals_series,
                )
                if not results:
                    # `scored: 0` on its own is indistinguishable from
                    # "nothing was due". Say which it is.
                    import csv as csv_module
                    with open(args.file, encoding="utf-8-sig", newline="") as handle:
                        reader = csv_module.DictReader(handle)
                        columns = reader.fieldnames or []
                        rows = list(reader)
                    time_column, _, _ = store._resolve_actuals_columns(
                        columns, args.actuals_time, args.actuals_target,
                        args.actuals_series,
                    )
                    print(json.dumps({
                        "schema_version": "0.1",
                        "status": "ok",
                        "project": args.project,
                        **store.explain_unscored(
                            args.project, [row[time_column] for row in rows],
                        ),
                    }, indent=2))
                    return 0
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
                            "avg_wape": m.avg_wape,
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
                lb = store.leaderboard(args.project, task=args.task)
                task_label = f" [{args.task}]" if args.task else ""
                print(f"\n  Model Leaderboard: {args.project}{task_label}")
                print(f"  {'Model':25s} {'Count':>5s} {'MASE':>7s} {'WAPE':>7s} {'MAPE':>7s} {'Bias':>8s} {'Coverage':>9s} {'Last':>7s}")
                print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*7}")
                for m in lb:
                    mase_s = f"{m.avg_mase:.3f}" if m.avg_mase is not None else "N/A"
                    # WAPE is the metric selection was decided on; showing it
                    # next to MASE is what lets a contradiction between the
                    # leaderboard and the selection be read as a real
                    # regression rather than a change of units.
                    wape_s = f"{m.avg_wape:.3f}" if m.avg_wape is not None else "N/A"
                    mape_s = f"{m.avg_mape:.1f}%" if m.avg_mape is not None else "N/A"
                    bias_s = f"{m.avg_bias:+.2f}" if m.avg_bias is not None else "N/A"
                    cov_s = f"{m.avg_coverage:.0%}" if m.avg_coverage is not None else "N/A"
                    last_s = f"{m.last_mase:.3f}" if m.last_mase is not None else "N/A"
                    print(f"  {m.model:25s} {m.count:>5d} {mase_s:>7s} {wape_s:>7s} {mape_s:>7s} {bias_s:>8s} {cov_s:>9s} {last_s:>7s}")
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

            elif args.track_command == "outcome":
                artifact = store.resolve_decision_outcome(
                    args.decision_id,
                    realised_scenario=args.realised_scenario,
                    realised_utilities=_json_argument(args.realised_utilities, argument="realised-utilities"),
                    constraint_violations=_json_argument(args.violations, argument="violations"),
                    note=args.note,
                )
                print(json.dumps(artifact.to_dict(), indent=2, allow_nan=False))
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
            if args.eval_command == "episodes":
                from .episodes import (
                    evaluate_policy, honest_aion_policy, standard_suite,
                    write_runs_jsonl,
                )
                workdir = Path(args.workdir).expanduser()
                workdir.mkdir(parents=True, exist_ok=True)
                (workdir / "worlds").mkdir(parents=True, exist_ok=True)
                episodes = standard_suite(workdir / "worlds")
                report = evaluate_policy(
                    episodes, honest_aion_policy, workdir / "runs", trials=args.trials,
                )
                if args.jsonl:
                    write_runs_jsonl(report["rows"], Path(args.jsonl).expanduser())
                    report["jsonl"] = str(Path(args.jsonl).expanduser())
                print(json.dumps(report, indent=2, allow_nan=False))
                return 0
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
            as_of = None
            if getattr(args, "as_of", None):
                from .data import _parse_timestamp
                as_of = _parse_timestamp(args.as_of, 0)
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
                as_of=as_of,
                store_path=getattr(args, "store_path", None),
                repair=args.repair,
            )
            payload = forecast_summary(artifact, path)

            # Auto-register in tracking store if --project is set
            if getattr(args, "project", None):
                from .tracking import register_artifact
                register_artifact(artifact, args.project, str(path))
                print(f"Registered forecast {artifact.forecast_id} in project '{args.project}'", file=sys.stderr)

        print(json.dumps(_disclose_assumptions(payload, schema_assumptions),
                         indent=2, allow_nan=False))
        return 0
    except AionError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        error = AionError("TRACKING_ERROR", str(exc))
        print(json.dumps(error.to_dict(), indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        # No `aion` invocation prints a Python traceback. An unexpected
        # failure is still a failure of the product, and it reaches the
        # caller in the same envelope as every other one so a wrapper never
        # has to special-case it.
        error = AionError(
            "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}",
            {"command": getattr(args, "command", None)},
        )
        print(json.dumps(error.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
