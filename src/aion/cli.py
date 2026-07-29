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
        if args.command == "capabilities":
            from .runtime import capabilities

            payload = capabilities()
        elif args.command == "inspect":
            from .runtime import inspect_dataset

            payload = inspect_dataset(
                args.input, time_column=args.time_column, target_column=args.target_column,
                series_column=args.series_column, frequency=args.frequency,
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
            config = load_config(getattr(args, "config", None))
            artifact, path = forecast(
                args.input, time_column=args.time_column, target_column=args.target_column,
                series_column=args.series_column, frequency=args.frequency, horizon=args.horizon,
                output=args.output, minimum_baseline_improvement=args.minimum_baseline_improvement,
                context_events=events, threshold=args.threshold,
                config=config,
            )
            payload = forecast_summary(artifact, path)
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except AionError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
