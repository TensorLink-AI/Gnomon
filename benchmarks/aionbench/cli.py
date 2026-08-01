"""``aionbench`` — the command line over the harness.

    aionbench suites                       what can be run
    aionbench models --check <slugs>       verify OpenRouter slugs before spending
    aionbench run --suite tsqa ...         run a grid and grade it
    aionbench report --run <dir>           re-render a report from graded rows
    aionbench export --run <dir>           emit `aion eval compare` inputs

Errors are typed and printed as JSON, matching Aion's CLI convention: a
failing benchmark run should be as machine-readable as a failing forecast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .contracts import CONDITIONS, BenchError
from .datasets import cache_dir
from .report import export_eval_compare, load_rows, render_markdown, summarise, write_report
from .router import API_KEY_ENV, EchoClient, OpenRouterClient, ResponseCache
from .runner import RunConfig, run
from .suites import available, describe_all, get_suite

def _default_models_file() -> Path:
    """The shipped roster, from a source checkout or an installed wheel."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "configs" / "models.yaml", here.parent / "configs" / "models.yaml"):
        if candidate.is_file():
            return candidate
    return here.parent / "configs" / "models.yaml"


DEFAULT_MODELS_FILE = _default_models_file()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aionbench",
        description="Temporal-reasoning benchmarks for Aion and for LLMs alone, via OpenRouter",
    )
    parser.add_argument("--version", action="version", version="aionbench 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("suites", help="List the available benchmark suites")

    models_parser = commands.add_parser("models", help="Inspect the OpenRouter model catalogue")
    models_parser.add_argument("--filter", help="Case-insensitive substring match on the slug")
    models_parser.add_argument(
        "--check", help="Comma-separated slugs to verify exist before a run",
    )
    models_parser.add_argument(
        "--roster", help="YAML roster to verify (default: benchmarks/configs/models.yaml)",
    )
    models_parser.add_argument("--limit", type=int, default=60)

    run_parser = commands.add_parser("run", help="Run a suite across models and conditions")
    run_parser.add_argument("--suite", required=True, choices=available())
    run_parser.add_argument(
        "--models", required=True,
        help="Comma-separated OpenRouter slugs, or @path to a YAML/text roster",
    )
    run_parser.add_argument(
        "--conditions", default=None,
        help=f"Comma-separated from {', '.join(CONDITIONS)} (default: the suite's control and treatment)",
    )
    run_parser.add_argument("--limit", type=int, help="Sample this many tasks (deterministic)")
    run_parser.add_argument("--seed", type=int, default=20260801)
    run_parser.add_argument("--categories", help="Comma-separated category filter")
    run_parser.add_argument("--config", dest="suite_config", help="Suite subset (e.g. tsaia multiple_choice)")
    run_parser.add_argument("--data-path", help="Local corpus file instead of downloading")
    run_parser.add_argument("--revision", default="main", help="Dataset revision to fetch")
    run_parser.add_argument("--dataset-repo", help="Hugging Face dataset repo id (tsqa generic corpora)")
    run_parser.add_argument("--dataset-file", help="File within the dataset repo")
    run_parser.add_argument("--field-map", help="key=column pairs, comma separated (tsqa)")
    run_parser.add_argument("--output", default="bench-output", help="Run directory")
    run_parser.add_argument("--cache-dir", help="Download and response cache (default ~/.cache/aionbench)")
    run_parser.add_argument("--no-cache", action="store_true", help="Do not reuse or store provider responses")
    run_parser.add_argument("--repeats", type=int, default=1, help="Trials per cell (pass^k, not best-of)")
    run_parser.add_argument("--concurrency", type=int, default=4)
    run_parser.add_argument("--budget-usd", type=float, help="Stop scheduling once spend reaches this")
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--max-tokens", type=int, default=4096)
    run_parser.add_argument("--max-tool-steps", type=int, default=8)
    run_parser.add_argument("--tool-profile", default="analysis",
                            choices=["analysis", "forecast", "decision", "full"])
    run_parser.add_argument("--include-hint", action="store_true",
                            help="Supply the dataset's reasoning hint (TimeSeriesExam ablation)")
    run_parser.add_argument("--include-concepts", action="store_true",
                            help="Supply the dataset's concept glossary (TimeSeriesExam ablation)")
    run_parser.add_argument("--decimals", type=int, default=1, help="Decimal places when rendering series")
    run_parser.add_argument("--max-points", type=int, help="Truncate rendered series to the last N points")
    run_parser.add_argument("--allow-pickle", action="store_true",
                            help="Load pickled dataset assets (TSAIA). Unpickling runs arbitrary code")
    run_parser.add_argument("--allow-code-execution", action="store_true",
                            help="Execute model-written Python in a subprocess sandbox")
    run_parser.add_argument("--exec-timeout", type=int, default=60)
    run_parser.add_argument("--threshold", action="append", default=[],
                            metavar="METRIC=VALUE", help="Override a metric pass bar")
    run_parser.add_argument("--no-resume", action="store_true", help="Ignore rows already in the run directory")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Run the full pipeline against a scripted local client — no key, no spend")
    run_parser.add_argument("--quiet", action="store_true")

    report_parser = commands.add_parser("report", help="Summarise graded rows")
    report_parser.add_argument("--run", required=True, help="Run directory or rows.jsonl")
    report_parser.add_argument("--format", choices=["md", "json"], default="md")
    report_parser.add_argument("--write", action="store_true", help="Also write report.md/report.json")

    export_parser = commands.add_parser("export", help="Emit `aion eval compare` inputs")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--model", required=True)
    export_parser.add_argument("--control", default="direct")
    export_parser.add_argument("--treatment", default="aion")
    export_parser.add_argument("--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except BenchError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed rows are on disk; rerun the same command to resume.",
              file=sys.stderr)
        return 130


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "suites":
        print(json.dumps({"schema_version": "0.1", "suites": describe_all()}, indent=2))
        return 0

    if args.command == "models":
        return _models(args)

    if args.command == "run":
        return _run(args)

    if args.command == "report":
        rows = load_rows(args.run)
        summary = summarise(rows)
        if args.write:
            write_report(Path(args.run) if Path(args.run).is_dir() else Path(args.run).parent, rows)
        print(json.dumps(summary, indent=2) if args.format == "json" else render_markdown(summary))
        return 0

    if args.command == "export":
        rows = load_rows(args.run)
        control_path, treatment_path = export_eval_compare(
            rows, args.output, model=args.model,
            control=args.control, treatment=args.treatment,
        )
        print(json.dumps({
            "baseline": str(control_path),
            "treatment": str(treatment_path),
            "next": (
                f"aion eval compare --baseline {control_path} --treatment {treatment_path}"
            ),
        }, indent=2))
        return 0

    raise BenchError("UNKNOWN_COMMAND", f"No such command: {args.command}")


def _models(args: argparse.Namespace) -> int:
    client = OpenRouterClient()
    catalogue = client.list_models()
    slugs = {str(item.get("id")) for item in catalogue}

    if args.check or args.roster or not args.filter:
        wanted = _requested_slugs(args)
        if wanted:
            missing = [slug for slug in wanted if slug not in slugs]
            print(json.dumps({
                "checked": len(wanted),
                "available": [slug for slug in wanted if slug in slugs],
                "missing": missing,
                "note": (
                    "Missing slugs are renamed or retired. Fix the roster before "
                    "running — an unknown slug fails every cell it appears in."
                ) if missing else "All slugs resolve.",
            }, indent=2))
            return 1 if missing else 0

    needle = (args.filter or "").lower()
    rows = [
        {
            "id": item.get("id"),
            "context_length": item.get("context_length"),
            "prompt_usd_per_mtok": _per_mtok(item, "prompt"),
            "completion_usd_per_mtok": _per_mtok(item, "completion"),
        }
        for item in catalogue
        if needle in str(item.get("id", "")).lower()
    ]
    rows.sort(key=lambda item: str(item["id"]))
    print(json.dumps({"matched": len(rows), "models": rows[:args.limit]}, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    suite = get_suite(args.suite)

    extra: dict[str, Any] = {}
    if args.dataset_repo:
        extra["dataset_repo"] = args.dataset_repo
    if args.dataset_file:
        extra["dataset_file"] = args.dataset_file
    if args.field_map:
        extra["field_map"] = _key_values(args.field_map)

    config = RunConfig(
        suite=args.suite,
        models=tuple(_resolve_models(args.models)),
        conditions=tuple(_split(args.conditions)),
        repeats=max(1, args.repeats),
        concurrency=max(1, args.concurrency),
        output_dir=args.output,
        cache_dir=args.cache_dir,
        budget_usd=args.budget_usd,
        resume=not args.no_resume,
        data_path=args.data_path,
        revision=args.revision,
        limit=args.limit,
        seed=args.seed,
        categories=tuple(_split(args.categories)),
        config=args.suite_config,
        allow_pickle=args.allow_pickle,
        extra=extra,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_tool_steps=args.max_tool_steps,
        tool_profile=args.tool_profile,
        include_hint=args.include_hint,
        include_concepts=args.include_concepts,
        decimals=args.decimals,
        max_points=args.max_points,
        allow_code_execution=args.allow_code_execution,
        exec_timeout_seconds=args.exec_timeout,
        thresholds=_thresholds(args.threshold),
    )

    if args.allow_code_execution and not args.quiet:
        print(
            "Executing model-written Python. The sandbox is a subprocess with "
            "wall-clock and memory caps and no proxy credentials — containment "
            "against carelessness, not against a hostile generator. Run the "
            "whole harness in a container for untrusted models.",
            file=sys.stderr,
        )

    if args.dry_run:
        client: Any = EchoClient()
    else:
        cache = ResponseCache(None if args.no_cache else cache_dir(args.cache_dir) / "responses")
        client = OpenRouterClient(cache=cache, seed=args.seed)
        if not client.api_key:
            raise BenchError(
                "OPENROUTER_KEY_MISSING",
                f"Set {API_KEY_ENV}, or pass --dry-run to exercise the harness without a provider.",
                {"env": API_KEY_ENV},
            )

    def progress(row: dict[str, Any], spent: float) -> None:
        if args.quiet:
            return
        mark = {True: "✓", False: "✗", None: "·"}[row.get("correct")]
        print(
            f"{mark} {row['model']:<34} {row['condition']:<10} {row['task_id']:<24} "
            f"${spent:0.4f}",
            file=sys.stderr,
        )

    result = run(suite, client, config, progress=progress)
    summary = write_report(result.run_dir, result.rows)

    print(json.dumps({
        "schema_version": "0.1",
        "status": "stopped_early" if result.stopped_early else "complete",
        "reason": result.reason or None,
        "run_dir": str(result.run_dir),
        "rows": len(result.rows),
        "spend_usd": result.manifest.get("spend_usd"),
        "arms": {
            name: {"graded": arm["graded"], "accuracy": arm["accuracy"]}
            for name, arm in summary["arms"].items()
        },
        "uplift": summary["uplift"],
        "report": str(result.run_dir / "report.md"),
    }, indent=2))
    return 0


# -- argument helpers ------------------------------------------------------

def _split(raw: str | None) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _key_values(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in _split(raw):
        if "=" not in pair:
            raise BenchError("INVALID_KEY_VALUE", f"Expected key=value, got {pair!r}.")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _thresholds(raw: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in raw:
        if "=" not in pair:
            raise BenchError("INVALID_THRESHOLD", f"Expected METRIC=VALUE, got {pair!r}.")
        key, value = pair.split("=", 1)
        try:
            out[key.strip()] = float(value)
        except ValueError as exc:
            raise BenchError("INVALID_THRESHOLD", f"{value!r} is not a number.") from exc
    return out


def _resolve_models(raw: str) -> list[str]:
    if not raw.startswith("@"):
        models = _split(raw)
        if not models:
            raise BenchError("NO_MODELS", "--models must name at least one slug.")
        return models
    return _read_roster(Path(raw[1:]).expanduser())


def _read_roster(path: Path) -> list[str]:
    if not path.is_file():
        raise BenchError("ROSTER_NOT_FOUND", f"No roster at {path}.")
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        return _roster_from_yaml(text, path)
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _roster_from_yaml(text: str, path: Path) -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise BenchError(
            "PYYAML_REQUIRED",
            "Reading a YAML roster needs PyYAML. Install the bench extra, or pass "
            "a plain-text roster with one slug per line.",
        ) from exc

    payload = yaml.safe_load(text) or {}
    entries = payload.get("models", payload) if isinstance(payload, dict) else payload
    slugs: list[str] = []
    if isinstance(entries, dict):
        for group in entries.values():
            slugs.extend(_slugs_from(group))
    else:
        slugs.extend(_slugs_from(entries))
    if not slugs:
        raise BenchError("EMPTY_ROSTER", f"No model slugs found in {path}.")
    # De-duplicate while keeping roster order — grouped rosters overlap.
    seen: set[str] = set()
    return [slug for slug in slugs if not (slug in seen or seen.add(slug))]


def _slugs_from(entry: Any) -> list[str]:
    if isinstance(entry, str):
        return [entry]
    if isinstance(entry, dict):
        return [str(entry["id"])] if "id" in entry else []
    if isinstance(entry, list):
        out: list[str] = []
        for item in entry:
            out.extend(_slugs_from(item))
        return out
    return []


def _requested_slugs(args: argparse.Namespace) -> list[str]:
    if args.check:
        return _split(args.check)
    roster = Path(args.roster).expanduser() if args.roster else DEFAULT_MODELS_FILE
    return _read_roster(roster) if roster.is_file() else []


def _per_mtok(item: dict[str, Any], key: str) -> float | None:
    pricing = item.get("pricing") or {}
    try:
        return round(float(pricing.get(key, 0)) * 1_000_000, 4)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
