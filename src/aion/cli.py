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

            events = load_events_file(args.context_file) if args.context_file else None
            artifact, path = forecast(
                args.input, time_column=args.time_column, target_column=args.target_column,
                series_column=args.series_column, frequency=args.frequency, horizon=args.horizon,
                output=args.output, minimum_baseline_improvement=args.minimum_baseline_improvement,
                context_events=events, threshold=args.threshold,
            )
            payload = forecast_summary(artifact, path)
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except AionError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
