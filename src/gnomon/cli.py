"""Small CLI over the same operator-owned session used by Python and MCP."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Sequence

from .contracts import GnomonError
from .product_contract import __version__
from .session import EVALUATE_SCHEMA, INSPECT_SCHEMA, ROUTE_SCHEMA, GnomonSession, read_json_argument


_CONFIG_HELP = 'Path to operator TOML (not JSON); e.g. ledger_path = "ledger.db"'
_EXAMPLES = {
    "evaluate": '{"data":{"input":"data.csv"},"candidates":["historical_mean"],'
                '"baseline":"last_value","horizon":2,"folds":4,"budget":{"max_calls":8}}',
    "route": '{"data":{"input":"data.csv"},"study_id":"STUDY_ID",'
             '"candidates":["historical_mean"],"baseline":"last_value","horizon":2,'
             '"source_as_of":"2026-01-31T00:00:00Z","recorded_as_of":"2099-01-01T00:00:00Z"}',
}


class _UsageError(GnomonError):
    """Keep JSON errors, without treating a CLI usage mistake as evidence."""

    def __init__(self, message, prog="gnomon"):
        super().__init__("INVALID_ARGUMENTS", message, repair_options=[{
            "action": "show_usage", "description": f"Run {prog} --help.",
        }])

    def to_dict(self):
        result = super().to_dict()
        result.pop("rejection")
        return result


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message, self.prog)


def _input_options(parser):
    parser.add_argument("--time-column", default="timestamp",
                        help="Time column name (default: timestamp, for every provider); use --time-column ts for ts,value CSV")
    parser.add_argument("--target-column", default="value", help="Target column name (default: value)")
    parser.add_argument("--timezone", help="Declare input timezone, e.g. UTC or Australia/Brisbane; required for routing date-only data")
    parser.add_argument("--window", choices=("latest_contiguous",),
                        help="Use the latest uninterrupted observed segment per series; requires --frequency and discloses excluded rows")
    for name in ("series-column", "frequency", "as-of", "recorded-as-of", "store-path", "unit"):
        parser.add_argument("--" + name)
    parser.add_argument("--repair", choices=("off", "safe", "aggressive"), default="off")
    parser.add_argument("--regrid", choices=("business_daily", "month_start"))


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="gnomon", description="Run your time-series models and keep explicit evidence.")
    parser.add_argument("--version", action="version", version=f"gnomon {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    caps = commands.add_parser("capabilities", help="List registered providers and enabled tools")
    caps.add_argument("--output", choices=("json",), default="json")
    caps.add_argument("--providers-config", help=_CONFIG_HELP)
    infer = commands.add_parser("infer", help="Run a named provider without implicit backtesting")
    infer.add_argument("--provider", required=True)
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", help="Forecast request JSON or @file.json")
    source.add_argument("--input", help="Data file or store:<dataset>")
    infer.add_argument("--horizon", type=int)
    infer.add_argument("--season", type=int, help="Seasonal period in observations for --input (default: 1); weekly daily data uses 7")
    infer.add_argument("--quantiles", type=float, nargs="+", help="Requested quantiles for --input, e.g. 0.1 0.5 0.9; provider must support them")
    infer.add_argument("--series-id")
    infer.add_argument("--providers-config", help=_CONFIG_HELP)
    infer.add_argument("--no-cache", action="store_true", help=(
        "Bypass cache lookup. Cache is off by default and session-only; each CLI invocation "
        "starts a fresh session, so separate infer runs cannot reuse results."))
    _input_options(infer)
    inspect = commands.add_parser("inspect", help="Validate data and show its snapshot provenance")
    inspect.add_argument("input", nargs="?", help="Data file, store:<dataset>, or - for stdin")
    inspect.add_argument("--input", dest="input_option", help="Alias for the positional input")
    inspect.add_argument("--for", dest="purpose", choices=("infer", "evaluate", "route"), default="infer",
                         help="Check suitability for the next operation before proceeding (default: infer)")
    inspect.add_argument("--save-snapshot", metavar="FILE.gnomon", help="Save frozen data for reuse via --input FILE.gnomon in later processes")
    _input_options(inspect)
    describe = commands.add_parser("describe", help="Calculate an exact statistic over observed data")
    describe.add_argument("input", nargs="?", help="Data file, store:<dataset>, or - for stdin")
    describe.add_argument("--input", dest="input_option", help="Alias for the positional input")
    describe.add_argument("--statistic", required=True,
                          choices=("mean", "median", "latest", "minimum", "maximum", "sum"))
    for name in ("series-id", "start", "end"):
        describe.add_argument("--" + name)
    _input_options(describe)
    for name in ("evaluate", "route", "ledger"):
        epilog = None
        if name in _EXAMPLES:
            direct = ("gnomon evaluate --input data.gnomon --candidates historical_mean --baseline last_value "
                      "--horizon 2 --ledger-path evidence.db --save-result study.json" if name == "evaluate" else
                      "gnomon route --input data.gnomon --study @study.json --ledger-path evidence.db "
                      "--source-as-of 2026-01-31T00:00:00Z --recorded-as-of 2099-01-01T00:00:00Z")
            epilog = (
                "Prepare a reusable input (declare your source's actual timezone):\n"
                "  gnomon inspect --input data.csv --timezone UTC --for route --save-snapshot data.gnomon\n\n"
                f"Direct flags:\n  {direct}\n\n"
                f"JSON example:\n  gnomon {name} --providers-config providers.toml --arguments '{_EXAMPLES[name]}'\n\n"
                "providers.toml (TOML, not JSON):\n  schema_version = 1\n  ledger_path = \"ledger.db\"\n\n"
                "Pass a JSON object directly, not a tools/call or arguments wrapper.\n"
                'data must be an inspection object, e.g. {"input":"data.csv","time_column":"ts"}.\n'
                f"Run gnomon {name} --schema for all fields and types.\n"
                "Data references expire when the CLI process exits; a saved .gnomon snapshot preserves frozen data.\n"
                "Evaluate with this same config first, then use its study_id in route.\n"
                "For route, data timestamps and cutoffs must include a timezone. Set source_as_of\n"
                "to the last observation, or inspect data with as_of to select an earlier cutoff.\n"
                "Replace example cutoffs with your analysis cutoffs; recorded_as_of must be at\n"
                "or after the study's recording time to use it (2099 illustrates all recorded evidence).\n"
                "Evaluation exits 0 when complete, 3 when partial, and 2 when unscored; inspect issues for recovery."
            )
        sub = commands.add_parser(name, help=f"Run the session's {name} operation",
                                  epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument("--arguments", help="JSON object or @file.json" + (
            "; see example below" if name in _EXAMPLES else ""))
        if name in _EXAMPLES:
            source.add_argument("--schema", action="store_true", help="Print the CLI arguments JSON Schema and exit")
            source.add_argument("--input", help="Data file, saved .gnomon snapshot, store:<dataset> or -; use task flags instead of JSON")
            sub.add_argument("--candidates", nargs="+", help="Candidate provider names; excludes the baseline")
            sub.add_argument("--baseline", help="Explicit baseline provider")
            sub.add_argument("--horizon", type=int, help="Forecast steps per fold")
            sub.add_argument("--season", type=int, help="Seasonal period in observations (default: 1)")
            sub.add_argument("--series-id")
            if name == "evaluate":
                for option in ("folds", "min-history", "stride", "max-calls"):
                    sub.add_argument("--" + option, type=int)
            else:
                sub.add_argument("--study", help="Study ID or @study.json saved by evaluate; a report supplies candidates, baseline, horizon and season")
                sub.add_argument("--source-as-of", help="Required source cutoff, with explicit timezone")
                sub.add_argument("--min-folds", type=int)
                sub.add_argument("--min-improvement", type=float)
            _input_options(sub)
        sub.add_argument("--providers-config", help=_CONFIG_HELP)
        sub.add_argument("--ledger-path", help="SQLite ledger path; route/ledger require this or ledger_path in provider TOML")
        sub.add_argument("--save-result", help="Save the result JSON atomically; stdout still returns the full result")
    temporal = commands.add_parser("temporal", help="Calculate explicit dates, intervals and event order")
    temporal.add_argument("--arguments", required=True, help="JSON object or @file.json")
    check = commands.add_parser("self-check", help="Verify installed snapshot cutoff behavior")
    check.add_argument("check", choices=("leakage",))
    check.add_argument("--cases", type=int, default=8)
    check.add_argument("--seed", type=int, default=7)
    mcp = commands.add_parser("mcp", help="Serve the execution session to an agent")
    serve = mcp.add_subparsers(dest="mcp_command", required=True).add_parser("serve")
    serve.add_argument("--providers-config", help=_CONFIG_HELP)
    infer.add_argument("--ledger-path", help="Record forecasts in this SQLite ledger without a TOML file")
    infer.add_argument("--save-result", help="Save the result JSON atomically")
    return parser


def _arguments_schema(command):
    """Extend the shared tool schema with the CLI's inline inspection form."""
    schema = deepcopy(EVALUATE_SCHEMA if command == "evaluate" else ROUTE_SCHEMA)
    variants = schema["oneOf"] if "oneOf" in schema else [schema]
    inline = deepcopy(variants[0])
    inline["required"] = ["data" if key == "data_ref" else key for key in inline["required"]]
    inline["properties"].pop("data_ref")
    inline["properties"]["data"] = deepcopy(INSPECT_SCHEMA)
    return {"type": "object", "oneOf": [inline, *variants]}


def _validate_cli_args(args):
    prog = f"gnomon {args.command}"
    if args.command in ("inspect", "describe"):
        if args.input is not None and args.input_option is not None:
            raise _UsageError("Supply input either positionally or with --input, not both.", prog)
        args.input = args.input if args.input is not None else args.input_option
        if args.input is None:
            raise _UsageError("An input is required: supply a positional path or --input PATH.", prog)
    if args.command in {"route", "ledger"} and not getattr(args, "schema", False) \
            and args.providers_config is None and args.ledger_path is None:
        raise _UsageError("Supply --ledger-path evidence.db or --providers-config providers.toml with ledger_path set.", prog)
    if args.command in {"evaluate", "route"} and not args.input:
        keys = ["candidates", "baseline", "horizon", "season", "series_id", "timezone", "series_column",
                "frequency", "as_of", "recorded_as_of", "store_path", "unit", "regrid", "window"]
        keys += (["folds", "min_history", "stride", "max_calls"] if args.command == "evaluate" else
                 ["study", "source_as_of", "min_folds", "min_improvement"])
        if any(getattr(args, key) is not None for key in keys) or args.repair != "off" \
                or args.time_column != "timestamp" or args.target_column != "value":
            raise _UsageError("Task and data flags require --input; put them inside JSON when using --arguments.", prog)


_INPUT_KEYS = ("input", "time_column", "target_column", "series_column", "frequency",
               "as_of", "recorded_as_of", "store_path", "unit", "repair", "regrid", "timezone", "purpose", "window")
MAX_STDIN_BYTES = 8 * 1024 * 1024


def _inspect(session, args):
    arguments = {key: getattr(args, key) for key in _INPUT_KEYS if getattr(args, key, None) is not None}
    if args.command in {"evaluate", "route"}:
        arguments["purpose"] = args.command
    if args.command == "route":
        # The route recording cutoff governs evidence, not the input file's
        # observation recording history. Store replay stays explicit in JSON.
        if not args.input.startswith("store:"):
            arguments.pop("recorded_as_of", None)
        if (args.input.startswith("store:") or Path(args.input).suffix != ".gnomon") and args.as_of is None:
            arguments["as_of"] = args.source_as_of
    if args.input != "-":
        return session.call("gnomon_inspect", arguments, compact=False)
    raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
    if len(raw) > MAX_STDIN_BYTES:
        raise GnomonError("INPUT_TOO_LARGE", "Piped CSV input exceeds the 8 MiB limit; use a file.")
    # Inspection freezes the rows before this temporary source is removed.
    with TemporaryDirectory(prefix="gnomon-stdin-") as directory:
        source = Path(directory) / "stdin.csv"
        source.write_bytes(raw)
        return session.call("gnomon_inspect", {**arguments, "input": str(source)}, compact=False)


def _execute(session, args):
    if args.command == "capabilities":
        return session.capabilities()
    if args.command in ("inspect", "describe"):
        data = _inspect(session, args)
        if args.command == "inspect":
            if args.save_snapshot:
                saved = session.data.save(data["data_ref"], args.save_snapshot)
                data.update(snapshot_file=saved, reuse={"input": saved})
            else:
                data["reuse_guidance"] = "data_ref expires when this CLI exits. Use --save-snapshot data.gnomon, then --input data.gnomon."
            return data
        result = session.call("gnomon_describe", {
            "data_ref": data["data_ref"], "statistic": args.statistic,
            **{key: getattr(args, key) for key in ("series_id", "start", "end") if getattr(args, key) is not None},
        }, compact=False)
        return {**result, "input": data}
    if args.command == "infer":
        if args.input:
            if args.horizon is None:
                raise _UsageError("--input requires --horizon.", "gnomon infer")
            data = _inspect(session, args)
            task = {"data_ref": data["data_ref"], "horizon": args.horizon}
            for key in ("series_id", "season", "quantiles"):
                if getattr(args, key) is not None:
                    task[key] = getattr(args, key)
        else:
            if any(getattr(args, key) is not None for key in (
                    "horizon", "series_column", "series_id", "frequency", "as_of",
                    "recorded_as_of", "store_path", "unit", "regrid", "timezone", "season", "quantiles", "window")) or args.repair != "off" \
                    or args.time_column != "timestamp" or args.target_column != "value":
                raise _UsageError("File/schema flags apply only with --input, not --request.", "gnomon infer")
            task = {"request": read_json_argument(args.request)}
        result = session.call("gnomon_forecast", {"provider": args.provider, **task,
                              "use_cache": not args.no_cache}, compact=False)
        return {**result, "input": data} if args.input else result
    if args.command in {"evaluate", "route"} and args.input:
        arguments = _task_flags(args)
        data = _inspect(session, args)
        arguments["data_ref"] = data["data_ref"]
    else:
        arguments = read_json_argument(args.arguments)
    if args.command in ("evaluate", "route") and "data" in arguments:
        if not isinstance(arguments["data"], dict):
            raise _UsageError('data must be an inspection object, e.g. {"data":{"input":"data.csv"}}.',
                              f"gnomon {args.command}")
        if "data_ref" in arguments or (args.command == "evaluate" and "study_id" in arguments):
            raise _UsageError("data cannot be combined with data_ref/study_id.", f"gnomon {args.command}")
        data = session.call("gnomon_inspect", {**arguments.pop("data"), "purpose": args.command}, compact=False)
        arguments["data_ref"] = data["data_ref"]
    result = session.call("gnomon_" + args.command, arguments, compact=False)
    if args.command == "evaluate":
        result["persistence"] = {"durable": session.ledger is not None,
                                 "next_step": ("Reuse study_id with the same ledger, or pass --study @study.json after --save-result study.json."
                                               if session.ledger is not None else
                                               "Routing needs recorded executions. Evaluate with --ledger-path evidence.db; saving a report alone does not record executions.")}
    return result


def _task_flags(args):
    arguments = {}
    if args.command == "route" and args.study:
        if args.study.startswith("@"):
            report = read_json_argument(args.study)
            if (report.get("evidence") != "rolling_origin_backtest"
                    or not isinstance(report.get("study_id"), str) or not report["study_id"]
                    or not isinstance(report.get("baseline"), str)
                    or not isinstance(report.get("providers"), dict)
                    or report["baseline"] not in report["providers"]
                    or type(report.get("horizon")) is not int):
                raise _UsageError("--study @file must contain an evaluate result; stored predictions are verified against the ledger.", "gnomon route")
            arguments = {key: report[key] for key in ("study_id", "baseline", "horizon", "season", "series_id") if key in report}
            arguments["candidates"] = [key for key in report["providers"] if key != report["baseline"]]
        else:
            arguments["study_id"] = args.study
    keys = ["candidates", "baseline", "horizon", "season", "series_id"]
    keys += (["folds", "min_history", "stride"] if args.command == "evaluate" else
             ["source_as_of", "recorded_as_of", "min_folds", "min_improvement"])
    arguments.update({key: getattr(args, key) for key in keys if getattr(args, key) is not None})
    if args.command == "evaluate" and args.max_calls is not None:
        arguments["budget"] = {"max_calls": args.max_calls}
    required = ["candidates", "baseline", "horizon"]
    if args.command == "route":
        required += ["study_id", "source_as_of", "recorded_as_of"]
    missing = ["--" + ("study" if key == "study_id" else key.replace("_", "-")) for key in required if key not in arguments]
    if missing:
        raise _UsageError("--input requires " + ", ".join(missing) + ".", f"gnomon {args.command}")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = None
    try:
        args = build_parser().parse_args(argv)
        _validate_cli_args(args)
        if getattr(args, "schema", False):
            print(json.dumps(_arguments_schema(args.command), indent=2))
            return 0
        if args.command == "self-check":
            from .selfcheck import leakage_self_check
            result = leakage_self_check(args.cases, args.seed)
        elif args.command == "temporal":
            with GnomonSession(enable_temporal=True) as session:
                result = _execute(session, args)
        else:
            with GnomonSession.from_config(getattr(args, "providers_config", None),
                                           ledger_path=getattr(args, "ledger_path", None)) as session:
                if args.command == "mcp":
                    from .mcp_server import serve
                    return serve(session=session)
                result = _execute(session, args)
        if getattr(args, "save_result", None):
            from .snapshot_files import write_json
            result["saved_result"] = write_json(args.save_result, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("status") == "unscored" or not result.get("structural_claim_proven", True):
            return 2
        return 3 if result.get("status") == "partial" else 0
    except GnomonError as exc:
        error = exc
    except FileNotFoundError as exc:
        error = GnomonError("INPUT_NOT_FOUND", str(exc))
    except ValueError as exc:
        error = _UsageError(str(exc), f"gnomon {args.command}" if args else "gnomon")
    except KeyboardInterrupt:
        print(json.dumps(GnomonError("INTERRUPTED", "Operation interrupted.", retryable=True).to_dict()))
        return 130
    except Exception:
        # Provider failures must not expose credentials or private endpoints.
        details = {"command": getattr(args, "command", "unknown")}
        if details["command"] == "infer" and args is not None:
            details.update({"stage": "provider_execution", "provider": args.provider})
        error = GnomonError("EXECUTION_FAILED", "Provider or ledger execution failed.", details=details)
    # The CLI is a JSON interface: stdout always carries its one structured
    # response. The exit status distinguishes success from failure; stderr is
    # reserved for unstructured diagnostics emitted outside this boundary.
    payload = error.to_dict()
    if error.code == "INVALID_ARGUMENTS" and args is not None and args.command in _EXAMPLES:
        payload["error"]["details"].update({"example_arguments": json.loads(_EXAMPLES[args.command]),
                                           "schema_command": f"gnomon {args.command} --schema"})
    print(json.dumps(payload, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
