from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .contracts import AionError
from .runtime import capabilities, forecast, inspect_dataset


def _common_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input")
    parser.add_argument("--time", required=True, dest="time_column")
    parser.add_argument("--target", required=True, dest="target_column")
    parser.add_argument("--series", dest="series_column")
    parser.add_argument("--frequency")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aion", description="Evidence-backed local forecasting")
    parser.add_argument("--version", action="version", version="aion 0.1.0")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "capabilities":
            payload = capabilities()
        elif args.command == "inspect":
            payload = inspect_dataset(
                args.input, time_column=args.time_column, target_column=args.target_column,
                series_column=args.series_column, frequency=args.frequency,
            )
        else:
            artifact, path = forecast(
                args.input, time_column=args.time_column, target_column=args.target_column,
                series_column=args.series_column, frequency=args.frequency, horizon=args.horizon,
                output=args.output, minimum_baseline_improvement=args.minimum_baseline_improvement,
            )
            payload = {
                "schema_version": "0.1", "status": "complete",
                "forecast_id": artifact.forecast_id, "artifact_path": str(path),
                "results": [
                    {"series": item.series, "support": item.support,
                     "selected_model": item.selected_model, "warnings": item.warnings}
                    for item in artifact.results
                ],
            }
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    except AionError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

