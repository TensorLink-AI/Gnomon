"""Small CLI over the same operator-owned session used by Python and MCP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Sequence

from .contracts import GnomonError
from .product_contract import __version__, resolve_mcp_profile
from .session import GnomonSession, read_json_argument


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise GnomonError("INVALID_ARGUMENTS", message,
                          repair_options=[{"action": "show_usage", "description": f"Run {self.prog} --help."}])


def _input_options(parser):
    parser.add_argument("--time-column", default="timestamp")
    parser.add_argument("--target-column", default="value")
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
    caps.add_argument("--providers-config")
    infer = commands.add_parser("infer", help="Run a named provider without implicit backtesting")
    infer.add_argument("--provider", required=True)
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", help="Forecast request JSON or @file.json")
    source.add_argument("--input", help="Data file or store:<dataset>")
    infer.add_argument("--horizon", type=int)
    infer.add_argument("--series-id")
    infer.add_argument("--providers-config")
    infer.add_argument("--no-cache", action="store_true")
    _input_options(infer)
    inspect = commands.add_parser("inspect", help="Validate data and show its snapshot provenance")
    inspect.add_argument("input")
    _input_options(inspect)
    describe = commands.add_parser("describe", help="Calculate an exact statistic over observed data")
    describe.add_argument("input")
    describe.add_argument("--statistic", required=True,
                          choices=("mean", "median", "latest", "minimum", "maximum", "sum"))
    for name in ("series-id", "start", "end"):
        describe.add_argument("--" + name)
    _input_options(describe)
    for name in ("evaluate", "route", "ledger"):
        sub = commands.add_parser(name, help=f"Run the session's {name} operation")
        sub.add_argument("--arguments", required=True, help="JSON object or @file.json")
        sub.add_argument("--providers-config", required=name in ("ledger", "route"))
    temporal = commands.add_parser("temporal", help="Calculate explicit dates, intervals and event order")
    temporal.add_argument("--arguments", required=True, help="JSON object or @file.json")
    check = commands.add_parser("self-check", help="Verify installed snapshot cutoff behavior")
    check.add_argument("check", choices=("leakage",))
    check.add_argument("--cases", type=int, default=8)
    check.add_argument("--seed", type=int, default=7)
    mcp = commands.add_parser("mcp", help="Serve the execution session to an agent")
    serve = mcp.add_subparsers(dest="mcp_command", required=True).add_parser("serve")
    serve.add_argument("--profile", help="Only execution is supported; legacy profiles are retired")
    serve.add_argument("--providers-config")
    return parser


_INPUT_KEYS = ("input", "time_column", "target_column", "series_column", "frequency",
               "as_of", "recorded_as_of", "store_path", "unit", "repair", "regrid")
MAX_STDIN_BYTES = 8 * 1024 * 1024


def _inspect(session, args):
    arguments = {key: getattr(args, key) for key in _INPUT_KEYS if getattr(args, key) is not None}
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
            return data
        result = session.call("gnomon_describe", {
            "data_ref": data["data_ref"], "statistic": args.statistic,
            **{key: getattr(args, key) for key in ("series_id", "start", "end") if getattr(args, key) is not None},
        }, compact=False)
        return {**result, "input": data}
    if args.command == "infer":
        if args.input:
            if args.horizon is None:
                raise GnomonError("INVALID_ARGUMENTS", "--input requires --horizon.")
            data = _inspect(session, args)
            task = {"data_ref": data["data_ref"], "horizon": args.horizon}
            if args.series_id is not None:
                task["series_id"] = args.series_id
        else:
            if any(getattr(args, key) is not None for key in (
                    "horizon", "series_column", "series_id", "frequency", "as_of",
                    "recorded_as_of", "store_path", "unit", "regrid")) or args.repair != "off" \
                    or args.time_column != "timestamp" or args.target_column != "value":
                raise GnomonError("INVALID_ARGUMENTS", "File/schema flags apply only with --input, not --request.")
            task = {"request": read_json_argument(args.request)}
        result = session.call("gnomon_forecast", {"provider": args.provider, **task,
                              "use_cache": not args.no_cache}, compact=False)
        return {**result, "input": data} if args.input else result
    arguments = read_json_argument(args.arguments)
    if args.command in ("evaluate", "route") and "data" in arguments:
        if "data_ref" in arguments or (args.command == "evaluate" and "study_id" in arguments):
            raise GnomonError("INVALID_ARGUMENTS", "data cannot be combined with data_ref/study_id.")
        data = session.call("gnomon_inspect", arguments.pop("data"), compact=False)
        arguments["data_ref"] = data["data_ref"]
    return session.call("gnomon_" + args.command, arguments, compact=False)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "self-check":
            from .selfcheck import leakage_self_check
            result = leakage_self_check(args.cases, args.seed)
        elif args.command == "temporal":
            with GnomonSession(enable_temporal=True) as session:
                result = _execute(session, args)
        else:
            resolve_mcp_profile(getattr(args, "profile", None))
            with GnomonSession.from_config(getattr(args, "providers_config", None)) as session:
                if args.command == "mcp":
                    from .mcp_server import serve
                    return serve(session=session)
                result = _execute(session, args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("structural_claim_proven", True) else 2
    except GnomonError as exc:
        error = exc
    except FileNotFoundError as exc:
        error = GnomonError("INPUT_NOT_FOUND", str(exc))
    except ValueError as exc:
        error = GnomonError("INVALID_ARGUMENTS", str(exc))
    except KeyboardInterrupt:
        print(json.dumps(GnomonError("INTERRUPTED", "Operation interrupted.", retryable=True).to_dict()), file=sys.stderr)
        return 130
    except Exception:
        # Provider failures must not expose credentials or private endpoints.
        error = GnomonError("EXECUTION_FAILED", "Provider or ledger execution failed.")
    print(json.dumps(error.to_dict(), indent=2), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
