"""Small CLI over the same operator-owned session used by Python and MCP."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import sys
import subprocess
from tempfile import TemporaryDirectory
from typing import Sequence

from .contracts import GnomonError
from .build_info import build_info
from .session import EVALUATE_SCHEMA, INSPECT_SCHEMA, ROUTE_SCHEMA, GnomonSession, read_json_argument
from .forecast_adapter import ForecastAdapterError
from .repair import REPAIR_HELP


_CONFIG_HELP = 'Path to operator TOML (not JSON); e.g. ledger_path = "ledger.db". Relative paths resolve against the configuration file directory, not cwd.'
_SERIES_HELP = ("Select an existing series, not a new label. Unlabeled input uses __default__; "
                "use --series-column to read series labels from input.")
_SCHEMA_COMMANDS = {name: f"gnomon {name} --schema" for name in ("infer", "evaluate", "route", "ledger", "temporal")}
_SCHEMA_COMMANDS["configuration"] = "gnomon capabilities --config-schema"
_EXAMPLES = {
    "evaluate": '{"data":{"input":"data.csv"},"candidates":["historical_mean"],'
                '"baseline":"last_value","horizon":2,"folds":4,"budget":{"max_calls":8}}',
    "route": '{"data":{"input":"data.csv"},"study_id":"STUDY_ID",'
             '"candidates":["historical_mean"],"baseline":"last_value","horizon":2,'
             '"source_as_of":"2026-01-31T00:00:00Z","recorded_as_of":"2099-01-01T00:00:00Z"}',
    "ledger": '{"operation":"search","limit":10}',
    "infer": '{"history":[1,2,3],"horizon":2}',
    "temporal": '{"operation":"interval","left":{"start":"2026-01-01T00:00:00Z","end":"2026-01-03T00:00:00Z"},'
                '"right":{"start":"2026-01-02T00:00:00Z","end":"2026-01-04T00:00:00Z"}}',
}


class _UsageError(GnomonError):
    """Keep JSON errors, without treating a CLI usage mistake as evidence."""

    def __init__(self, message, prog="gnomon"):
        super().__init__("INVALID_ARGUMENTS", message, repair_options=[{
            "action": "show_usage", "description": f"Run {prog} --help. Run gnomon schemas for schema entry points.",
        }])

    def to_dict(self, *, compact=False):
        result = super().to_dict(compact=compact)
        result.pop("rejection")
        return result


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message, self.prog)


def _input_options(parser):
    # Short aliases share the canonical dest; help lists the long form first.
    parser.add_argument("--time-column", "--time", dest="time_column", default="timestamp", metavar="NAME",
                        help="Time column name (default: timestamp, for every provider); use --time-column ts for ts,value CSV. --time is an alias")
    parser.add_argument("--target-column", "--target", dest="target_column", default="value", metavar="NAME",
                        help="Target column name (default: value). --target is an alias")
    parser.add_argument("--timezone", help="Declare input timezone, e.g. UTC or Australia/Brisbane; required for routing date-only data")
    parser.add_argument("--window", choices=("latest_contiguous",),
                        help="Use the latest uninterrupted observed segment per series; requires --frequency and discloses excluded rows")
    for name in ("series-column", "frequency", "as-of", "recorded-as-of", "store-path", "unit"):
        parser.add_argument("--" + name)
    parser.add_argument("--repair", choices=("off", "safe", "aggressive"), default="safe", help=REPAIR_HELP.replace("%", "%%"))
    parser.add_argument("--regrid", choices=("business_daily", "month_start"))


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="gnomon", description="Run your time-series models and keep explicit evidence.")
    parser.add_argument("--version", action="version", version=f"gnomon {build_info()['build_id']}")
    errors = parser.add_mutually_exclusive_group()
    errors.add_argument('--compact-errors', dest='compact_errors', action='store_true', default=True, help='Reference the canonical error object (default)')
    errors.add_argument('--expanded-errors', dest='compact_errors', action='store_false', help='Include duplicate legacy rejection details')
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schemas", help="List command and operator configuration schema entry points")
    caps = commands.add_parser("capabilities", help="List registered providers and enabled tools")
    caps.add_argument("--output", choices=("json",), default="json")
    caps.add_argument("--providers-config", help=_CONFIG_HELP)
    caps.add_argument("--config-schema", action="store_true", help="Describe operator TOML configuration keys without loading providers")
    caps.add_argument('--show-resolved-config', action='store_true', help='Inspect resolved paths/settings without loading providers or secrets. Relative TOML paths resolve against the configuration file directory.')
    shape = caps.add_mutually_exclusive_group()
    shape.add_argument('--brief', dest='brief', action='store_true', default=True, help='Return shared request schemas once (default).')
    shape.add_argument('--expanded', dest='brief', action='store_false', help='Repeat complete request schemas under each provider.')
    caps.add_argument('--cache', action='store_true', help='Inspect cache policy; add --provider and --request to validate a lookup without executing a forecast')
    caps.add_argument('--provider')
    caps.add_argument('--request', help='JSON request or @file for --cache --provider')
    examples = commands.add_parser('providers', help='Discover a complete unit-bearing custom-provider example')
    output = examples.add_mutually_exclusive_group()
    output.add_argument('--raw', action='store_true', help='Print executable Python source instead of JSON')
    output.add_argument('--write-to', metavar='FILE.py', help='Write executable Python source to a new file')
    examples.add_argument('action', choices=['example'])
    infer = commands.add_parser("infer", aliases=["forecast"], help="Forecast with a named provider without implicit backtesting",
        epilog="Example: gnomon infer --provider last_value --request '{\"history\":[1,2,3],\"horizon\":2}'. "
               "Run gnomon infer --schema for the --request schema, and gnomon capabilities for provider names.")
    infer.usage = 'gnomon infer (forecast) ([INPUT | --input INPUT] --horizon HORIZON | --request JSON | --schema) [--provider PROVIDER] [options]'
    infer.add_argument("input_positional", nargs="?", metavar="INPUT", help="Data file (alias for --input)")
    infer.add_argument("--provider", help="Registered provider name. With a data file and only the built-in baselines "
                                          "registered, defaults to last_value and discloses that in assumptions")
    infer.add_argument('--verify', action='store_true', help='Include independent arithmetic verification for deterministic built-ins')
    infer.add_argument('--json', action='store_true', help='Print the full JSON envelope even on a terminal (the default whenever stdout is not a TTY)')
    source = infer.add_mutually_exclusive_group()
    source.add_argument("--schema", action="store_true", help="Print the JSON Schema for --request and exit")
    source.add_argument("--request", help="Forecast request JSON or @file.json")
    source.add_argument("--input", help="Data file or store:<dataset>. With no --provider and no column flags, a two-column "
                                        "CSV's timestamp-like and numeric columns are inferred and disclosed in assumptions")
    infer.add_argument("--horizon", type=int)
    infer.add_argument("--season", type=int, help="Seasonal period in observations for --input (default: 1); weekly daily data uses 7")
    infer.add_argument("--quantiles", type=float, nargs="+", help="Requested quantiles for --input, e.g. 0.1 0.5 0.9; provider must support them")
    infer.add_argument("--series-id", help=_SERIES_HELP)
    infer.add_argument("--providers-config", help=_CONFIG_HELP)
    infer.add_argument("--no-cache", action="store_true", help=(
        "Bypass cache lookup. Cache is off by default and session-only; each CLI invocation "
        "starts a fresh session, so separate infer runs cannot reuse results. "
        "Enable with cache_size = 8 in TOML and --providers-config providers.toml. "
        "For a two-call Python hit example, use help(GnomonSession.from_config); "
        "list settings with gnomon capabilities --config-schema."))
    _input_options(infer)
    inspect = commands.add_parser("inspect", help="Validate data and show its snapshot provenance")
    inspect.add_argument("input", nargs="?", help="Data file, store:<dataset>, or - for stdin")
    inspect.add_argument("--input", dest="input_option", help="Alias for the positional input")
    inspect.add_argument("--for", dest="purpose", choices=("infer", "evaluate", "route"), default="infer",
                         help="Check suitability for the next operation before proceeding (default: infer)")
    inspect.add_argument("--save-snapshot", metavar="FILE.gnomon", help="Save frozen data for reuse via --input FILE.gnomon in later processes")
    inspect.add_argument('--diagnose', action='store_true', help='Dry-run all repair modes on a local file (up to 8 MiB), with no writes or forecasts; cannot combine with --repair or --save-snapshot')
    _input_options(inspect)
    describe = commands.add_parser("describe", help="Calculate an exact statistic over observed data")
    describe.add_argument('--brief', action='store_true', help='Omit the repeated inspection object; retain statistic and snapshot provenance')
    describe.add_argument("input", nargs="?", help="Data file, store:<dataset>, or - for stdin")
    describe.add_argument("--input", dest="input_option", help="Alias for the positional input")
    describe.add_argument("--statistic", required=True,
                          choices=("mean", "median", "latest", "minimum", "maximum", "sum"))
    describe.add_argument("--series-id", help=_SERIES_HELP)
    for name in ("start", "end"):
        describe.add_argument("--" + name)
    _input_options(describe)
    for name in ("evaluate", "route", "ledger"):
        epilog = None
        if name in {"evaluate", "route"}:
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
                "Evaluation exits 0 when complete, 3 when partial, and 2 when unscored; inspect issues for recovery.\n"
                "Routing defaults to at least 3 replayable matched folds. Check evaluation routing_readiness; "
                "a complete study can still have too few folds."
            )
        if name == "ledger":
            epilog = ("Example: gnomon ledger --ledger-path evidence.db --arguments '" + _EXAMPLES[name] +
                      "'\nRun gnomon ledger --schema for operations and required fields. "
                      "Outcome writes require allow_outcome_writes=true in operator TOML.\n\n"
                      "Complete synthetic compare_history example (run in a fresh directory):\n"
                      "  python -m gnomon.examples.compare_history\n"
                      "Decision summaries, context comparison, reviews and portable lessons:\n"
                      "  python -m gnomon.examples.decision_memory\n"
                      "Uses a controlled clock, history/future timestamps, units, provider revisions and actual availability.\n\n"
                      "Ledger evaluate defaults allow_partial=true: exit 0/status ok means the operation succeeded.\n"
                      "Read scoring_status/complete and result.status/result.coverage (each result for batches).\n"
                      "Check result.coverage_basis for saved versus reconstructed legacy coverage; result.current_coverage refreshes query diagnostics.\n"
                      "Set allow_partial=false to reject incomplete horizons. Scoring persists evidence without actual-write opt-in; exact retries reuse scores.\n\n"
                      "Omitted defaults (source cutoff | recording cutoff | unit):\n"
                      "  search:          now       | now       | all units\n"
                      "  pending:         now       | now       | each execution's unit\n"
                      "  actuals_as_of:   unbounded | unbounded | unitless\n"
                      "  evaluate/compare: unbounded | unbounded | each execution's unit\n"
                      "  compare_history: required  | required  | unitless\n"
                      "  study/evaluations: n/a     | unbounded | n/a\n"
                      "  decision:        unbounded | unbounded | n/a\n"
                      "Cutoffs filter source availability and local recording times. Units match exactly; no conversion.")
        sub = commands.add_parser(name, help=f"Run the session's {name} operation",
                                  epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
        if name == 'route':
            sub.usage = 'gnomon route (--input INPUT --study STUDY --source-as-of SOURCE --recorded-as-of RECORDED | --arguments JSON | --schema) [options]'
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument("--arguments", help="JSON object or @file.json" + (
            "; see example below" if name in _EXAMPLES else ""))
        source.add_argument("--schema", action="store_true", help="Print the CLI arguments JSON Schema and exit")
        if name in {"evaluate", "route"}:
            source.add_argument("--input", help="Data file, saved .gnomon snapshot, store:<dataset> or -; use task flags instead of JSON")
            sub.add_argument("--candidates", nargs="+", help="Candidate provider names; excludes the baseline")
            sub.add_argument("--baseline", help="Explicit baseline provider")
            sub.add_argument("--horizon", type=int, help="Forecast steps per fold")
            sub.add_argument("--season", type=int, help="Seasonal period in observations (default: 1)")
            sub.add_argument("--series-id", help=_SERIES_HELP)
            if name == "evaluate":
                sub.add_argument('--json', action='store_true', help='Print the full JSON envelope even on a terminal (the default whenever stdout is not a TTY)')
                sub.add_argument('--preflight', action='store_true', default=None, help='Explain planned fold visibility and replay before any provider calls or saved study')
                sub.add_argument('--replay', choices=['recorded', 'source_available'], help='Explicit temporal replay choice; recording-bounded snapshots default to recorded. See --preflight.')
                sub.add_argument('--verify', action='store_true', default=None, help='Include fold error vectors, metric sums and exact scored-pair hashes')
                source.add_argument('--compare', nargs=2, metavar=('ORIGINAL_STUDY', 'RESCORED_STUDY'), help='Compare saved studies and verify prediction reuse; zero provider calls')
                sub.add_argument('--rescore', metavar='STUDY_ID', help='With --input, rescore original executions using revised actuals; requires both evidence cutoffs and a ledger')
                sub.add_argument('--source-as-of', help='For --rescore: inclusive actual source availability cutoff; never changes original forecast origins')
                sub.add_argument('--no-partial', action='store_true', default=None, help='For --rescore: require actuals for every original fold')
                sub.add_argument("--folds", type=int, help="Requested folds (default: 4); routing needs at least 3 successful replayable matched folds")
                for option in ("min-history", "stride", "max-calls"):
                    sub.add_argument("--" + option, type=int)
            else:
                sub.add_argument("--study", help="Study ID or @study.json saved by evaluate; both supply candidates, baseline, horizon, season and series from the recorded study")
                sub.add_argument("--source-as-of", help="Required source cutoff, with explicit timezone")
                sub.add_argument("--min-folds", type=int, help="Minimum replayable matched folds (default: 3, minimum: 3)")
                sub.add_argument("--min-improvement", type=float)
                sub.add_argument("--require-evidence", action="store_true", default=None,
                                 help="Reject any routing fallback with exit 2; default permits baseline fallback with exit 0. Check evidence_based and routing_status.")
            _input_options(sub)
        sub.add_argument("--providers-config", help=_CONFIG_HELP)
        sub.add_argument("--ledger-path", help="SQLite ledger path; route/ledger require this or ledger_path in provider TOML")
        sub.add_argument("--save-result", help="Save the result JSON atomically; stdout still returns the full result")
    temporal = commands.add_parser("temporal", help="Calculate explicit dates, intervals and event order",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=(
            "Operations: normalize, duration, shift, interval, order_events.\n"
            f"Example:\n  gnomon temporal --arguments '{_EXAMPLES['temporal']}'\n\n"
            "Intervals use left/right objects with start/end timestamps and half-open [start,end) boundaries.\n"
            "Run gnomon temporal --schema for every operation, required field and type."))
    temporal_source = temporal.add_mutually_exclusive_group(required=True)
    temporal_source.add_argument("--arguments", help="JSON object or @file.json")
    temporal_source.add_argument("--schema", action="store_true", help="Print the complete temporal JSON Schema and exit")
    check = commands.add_parser("self-check", help="self-check leakage: verify installed snapshot cutoff behavior")
    check.add_argument("check", choices=("leakage", "families"))
    check.add_argument('--detailed', action='store_true', help='Include actual and expected values for each self-check assertion')
    check.add_argument("--cases", type=int, default=8)
    check.add_argument("--seed", type=int, default=7)
    from .selfcheck import FAMILIES
    check.add_argument('--families', nargs='+', choices=FAMILIES, help='Additional finite offline checks; snapshot checks always run. Select explicit families to test revisions, recording, folds, covariates, series, DST, repairs or cache.')
    mcp = commands.add_parser("mcp", help="mcp serve: serve the execution session to an agent")
    serve = mcp.add_subparsers(dest="mcp_command", required=True).add_parser("serve")
    serve.add_argument("--providers-config", help=_CONFIG_HELP)
    infer.add_argument("--ledger-path", help="Record forecasts in this SQLite ledger without a TOML file")
    infer.add_argument("--save-result", help="Save the result JSON atomically")
    commands.add_parser("environment", help="Show the exact Python, import name, package path and installed build")
    python = commands.add_parser("python", help="Run Python in Gnomon's environment; e.g. gnomon python -c 'import gnomon'")
    python.add_argument("python_arguments", nargs=argparse.REMAINDER)
    releases = commands.add_parser("releases", help="List install.sh environments and preview removal of old releases")
    releases.add_argument("--prune", action="store_true", help="Preview old environments eligible for removal")
    releases.add_argument("--keep", type=int, default=1, help="Keep this many inactive environments, in addition to active/running ones (default: 1)")
    releases.add_argument("--apply", action="store_true", help="Apply --prune; active and running environments are protected")
    commands.add_parser("rollback", help="Activate a release listed by gnomon releases").add_argument("release_id")
    update = commands.add_parser("update", help="Install a Git ref into a new managed environment and activate it after validation")
    update.add_argument("--version", default="main", help="Tag, branch or commit (default: main)")
    return parser


def _arguments_schema(command):
    """Extend the shared tool schema with the CLI's inline inspection form."""
    if command == "infer":
        from .session import REQUEST_SCHEMA
        return deepcopy(REQUEST_SCHEMA)
    if command == "ledger":
        from .session import ledger_schema
        return {**ledger_schema(allow_outcome_writes=True), "description":
                "Schema for --arguments. append_actual, record_decision and append_decision_outcome require "
                "allow_outcome_writes=true in operator TOML. Ledger/route require an existing ledger."}
    if command == "temporal":
        from .temporal_ops import TEMPORAL_SCHEMA
        return deepcopy(TEMPORAL_SCHEMA)
    schema = deepcopy(EVALUATE_SCHEMA if command == "evaluate" else ROUTE_SCHEMA)
    variants = schema["oneOf"] if "oneOf" in schema else [schema]
    inline = deepcopy(variants[0])
    inline["required"] = ["data" if key == "data_ref" else key for key in inline["required"]]
    inline["properties"].pop("data_ref")
    inline["properties"]["data"] = deepcopy(INSPECT_SCHEMA)
    more_inline = []
    for variant in variants[1:]:
        if 'data_ref' in variant['properties']:
            converted = deepcopy(variant)
            converted['required'] = ['data' if k == 'data_ref' else k for k in converted['required']]
            converted['properties'].pop('data_ref')
            converted['properties']['data'] = deepcopy(INSPECT_SCHEMA)
            more_inline.append(converted)
    return {"type": "object", "oneOf": [inline, *variants, *more_inline]}


def _validate_cli_args(args):
    prog = f"gnomon {args.command}"
    if args.command == "infer":
        if args.input_positional is not None:
            if args.input is not None:
                raise _UsageError("Supply input either positionally or with --input, not both.", prog)
            args.input = args.input_positional
        if sum(1 for chosen in (args.schema, args.request, args.input) if chosen) != 1:
            raise _UsageError("one of the arguments --schema --request --input (or a positional INPUT) is required", prog)
        if not args.schema and not args.input and not args.provider:
            raise _UsageError("--provider is required; run gnomon capabilities for registered provider names.", prog)
    if args.command in {"evaluate", "route"} and args.input:
        args.task_arguments = _task_flags(args)
    if args.command in ("inspect", "describe"):
        if args.input is not None and args.input_option is not None:
            raise _UsageError("Supply input either positionally or with --input, not both.", prog)
        args.input = args.input if args.input is not None else args.input_option
        if args.input is None:
            raise _UsageError("An input is required: supply a positional path or --input PATH.", prog)
        if args.command == 'inspect' and args.diagnose and (args.save_snapshot or 'repair' in args._explicit_fields or args.input == '-'):
            raise _UsageError('--diagnose requires a local source file and cannot combine with --repair or --save-snapshot.', prog)
    if args.command in {"route", "ledger"} and not getattr(args, "schema", False) \
            and args.providers_config is None and args.ledger_path is None:
        raise _UsageError("Supply --ledger-path evidence.db or --providers-config providers.toml with ledger_path set.", prog)
    if args.command in {"evaluate", "route"} and not args.input:
        keys = ["candidates", "baseline", "horizon", "season", "series_id", "timezone", "series_column",
                "frequency", "as_of", "recorded_as_of", "store_path", "unit", "regrid", "window"]
        keys += (["folds", "min_history", "stride", "max_calls", "rescore", "source_as_of", "no_partial", "verify", "preflight", "replay"] if args.command == "evaluate" else
                 ["study", "source_as_of", "min_folds", "min_improvement", "require_evidence"])
        if any(getattr(args, key) is not None for key in keys) or args.repair != "safe" \
                or args.time_column != "timestamp" or args.target_column != "value":
            raise _UsageError("Task and data flags require --input; put them inside JSON when using --arguments.", prog)


_INPUT_KEYS = ("input", "time_column", "target_column", "series_column", "frequency",
               "as_of", "recorded_as_of", "store_path", "unit", "repair", "regrid", "timezone", "purpose", "window")
_FLAG_ALIASES = {"time": "time_column", "target": "target_column"}


def _explicit_fields(argv):
    """Fields the caller set on the command line, with short aliases mapped to their dest."""
    fields = {item.split('=')[0][2:].replace('-', '_') for item in argv if item.startswith('--')}
    return {_FLAG_ALIASES.get(field, field) for field in fields}
MAX_STDIN_BYTES = 8 * 1024 * 1024


def _stdout_is_tty():
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _number(value):
    text = f"{float(value):.6g}"
    return text


def _forecast_text(result):
    """Compact terminal view: one line per step, then provider, execution and snapshot."""
    forecast = result["result"]
    timestamps = forecast.get("timestamps") or [f"+{i + 1}" for i in range(len(forecast["point"]))]
    quantiles = forecast.get("quantiles") or [None] * len(forecast["point"])
    lines = []
    for timestamp, point, row in zip(timestamps, forecast["point"], quantiles):
        line = f"{timestamp}  {_number(point)}"
        if row:
            keys = sorted(row, key=float)
            line += f"  [{_number(row[keys[0]])} {_number(row[keys[-1]])}]"
        lines.append(line)
    lines.append(f"provider: {result['provider']} ({result.get('revision') or 'unversioned'})")
    lines.append(f"execution: {result['execution_id']}")
    snapshot = result.get("snapshot")
    if snapshot:
        lines.append(f"snapshot: {snapshot['snapshot_id']} as_of={snapshot['as_of']}")
    else:
        lines.append("snapshot: request supplied directly")
    lines.extend(_disclosure_lines(result, forecast.get("unit")))
    return "\n".join(lines)


def _disclosure_lines(result, unit=None):
    """Disclosures the JSON carries that a terminal reader must not lose: repairs, assumed choices, unit."""
    lines = []
    repairs = (result.get("input") or {}).get("repairs", result.get("repairs")) or []
    if repairs:
        codes = ", ".join(f"{r['code']}x{r['count']}" for r in repairs)
        assumptive = any(r.get("assumptive") for r in repairs)
        lines.append(f"repairs: {len(repairs)} ({codes}){' assumptive: values were changed or invented' if assumptive else ''}")
    assumptions = result.get("assumptions") or []
    if assumptions:
        lines.append("assumed: " + ", ".join(f"{a['field']}={a['value']} ({a['basis']})" for a in assumptions))
    if unit:
        lines.append(f"unit: {unit}")
    return lines


def _evaluate_text(result):
    """Per-fold MAE per provider, then the recorded ranking and study identity."""
    providers = list(result.get("provider_order") or result["providers"])
    width = max(len(name) for name in providers)
    lines = ["  ".join(["origin".ljust(19), *(name.rjust(max(width, 8)) for name in providers)])]
    for fold in result["folds"]:
        cells = []
        for name in providers:
            run = fold["runs"].get(name) or {}
            actuals = [item["value"] for item in fold.get("actuals") or []]
            if run.get("status") == "ok" and len(actuals) == len(run.get("point", ())) and actuals:
                error = sum(abs(p - a) for p, a in zip(run["point"], actuals)) / len(actuals)
                cells.append(_number(error).rjust(max(width, 8)))
            else:
                cells.append("-".rjust(max(width, 8)))
        lines.append("  ".join([str(fold["origin"])[:19].ljust(19), *cells]))
    scores = result.get("scores") or {}
    ranked = [f"{name} (mae {_number(scores[name]['mae'])})" if name in scores else name for name in result.get("ranking", [])]
    lines.append(f"ranking: {' > '.join(ranked) if ranked else 'unscored'}  [{result['status']}]")
    lines.append(f"study: {result['study_id']}")
    lines.extend(_disclosure_lines(result, result.get("unit")))
    return "\n".join(lines)


def _human_text(command, result):
    """Return terminal text for a complete forecast or evaluation, else None for JSON."""
    try:
        if command == "infer" and result.get("status") == "ok" and "result" in result:
            return _forecast_text(result)
        if command == "evaluate" and isinstance(result.get("folds"), list) and "study_id" in result:
            return _evaluate_text(result)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _is_number(text):
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _is_timestamp(text):
    from .data import _parse_timestamp
    try:
        _parse_timestamp(text, 0)
    except GnomonError:
        return False
    return True


def _infer_two_columns(path):
    """Name the timestamp-like and numeric columns of a plain two-column CSV.

    Returns None whenever the choice is not unambiguous: any other column
    count, a column that is neither kind, or both columns qualifying for the
    same role. The caller then falls through to the usual column error.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            sample = [row for _, row in zip(range(64), reader)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if header is None or len(header) != 2 or not sample:
        return None
    roles = []
    for index in range(2):
        values = [row[index].strip() for row in sample if len(row) == 2 and row[index].strip()]
        kinds = set()
        if values and all(_is_number(v) for v in values):
            kinds.add("numeric")
        if values and all(_is_timestamp(v) for v in values):
            kinds.add("timestamp")
        roles.append(kinds)
    timestamps = [i for i, kinds in enumerate(roles) if "timestamp" in kinds]
    numerics = [i for i, kinds in enumerate(roles) if "numeric" in kinds]
    if len(timestamps) != 1 or len(numerics) != 1 or timestamps == numerics:
        return None
    return header[timestamps[0]].strip(), header[numerics[0]].strip()


def _infer_defaults(session, args):
    """Fill the provider and column names a bare `forecast FILE --horizon N` leaves open.

    Column inference is limited to that bare form. Once a provider is named,
    column mapping stays explicit and the usual column-correction guidance applies.
    """
    if args.provider is not None:
        return []
    from .models import BASELINES
    if set(session.engine.capabilities()) - set(BASELINES):
        raise _UsageError("--provider is required; run gnomon capabilities for registered provider names.", "gnomon infer")
    args.provider = "last_value"
    assumptions = [{"field": "provider", "value": "last_value",
                    "basis": "default_reference_baseline_only_built_ins_registered",
                    "note": "A reference baseline, not a chosen model. Pass --provider to select one."}]
    explicit = getattr(args, "_explicit_fields", set())
    plain_csv = (not args.input.startswith("store:") and args.input != "-"
                 and Path(args.input).suffix.lower() == ".csv" and Path(args.input).is_file())
    if plain_csv and not {"time_column", "target_column"} & explicit:
        inferred = _infer_two_columns(args.input)
        if inferred is not None:
            for field, value in zip(("time_column", "target_column"), inferred):
                if value != getattr(args, field):
                    setattr(args, field, value)
                    args._explicit_fields = explicit = explicit | {field}
                    assumptions.append({"field": field, "value": value, "basis": "inferred_from_two_column_csv_header"})
    return assumptions


def _inspect(session, args):
    arguments = {key: getattr(args, key) for key in _INPUT_KEYS if getattr(args, key, None) is not None
                 and (key not in {'time_column', 'target_column', 'repair'} or key in getattr(args, '_explicit_fields', set()))}
    if getattr(args, 'diagnose', False):
        arguments['diagnose'] = True
    if args.command in {"evaluate", "route"}:
        arguments["purpose"] = args.command
    if args.command == "route" or (args.command == 'evaluate' and args.rescore):
        # The route/rescore recording cutoff governs evidence, not the input file's
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
        if args.cache:
            if bool(args.provider) != bool(args.request):
                raise _UsageError('--cache requires both --provider and --request for a request diagnostic, or neither for policies.', 'gnomon capabilities')
            return {'schema_version': '1', 'status': 'ok', 'cache': session.engine.cache_diagnostic(args.provider, read_json_argument(args.request))
                    if args.provider else {p: session.engine.cache_policy(p) for p in session.engine.capabilities()}}
        if args.provider or args.request:
            raise _UsageError('--provider and --request require --cache.', 'gnomon capabilities')
        return session.capabilities(brief=args.brief)
    if args.command in ("inspect", "describe"):
        data = _inspect(session, args)
        if args.command == "inspect":
            if args.diagnose:
                return data
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
        return result if args.brief else {**result, "input": data}
    if args.command == "infer":
        if args.input:
            if args.horizon is None:
                raise _UsageError("--input requires --horizon.", "gnomon infer")
            assumptions = _infer_defaults(session, args)
            data = _inspect(session, args)
            task = {"data_ref": data["data_ref"], "horizon": args.horizon}
            for key in ("series_id", "season", "quantiles"):
                if getattr(args, key) is not None:
                    task[key] = getattr(args, key)
        else:
            if any(getattr(args, key) is not None for key in (
                    "horizon", "series_column", "series_id", "frequency", "as_of",
                    "recorded_as_of", "store_path", "unit", "regrid", "timezone", "season", "quantiles", "window")) or args.repair != "safe" \
                    or args.time_column != "timestamp" or args.target_column != "value":
                raise _UsageError("File/schema flags apply only with --input, not --request.", "gnomon infer")
            task = {"request": read_json_argument(args.request)}
        result = session.call("gnomon_forecast", {"provider": args.provider, **task,
                              "use_cache": not args.no_cache, "verify": args.verify}, compact=False)
        if not args.input:
            return result
        return {**result, "input": data, **({"assumptions": assumptions} if assumptions else {})}
    if args.command == 'evaluate' and args.compare:
        return session.compare_studies(original_study_id=args.compare[0], rescored_study_id=args.compare[1])
    if args.command in {"evaluate", "route"} and args.input:
        arguments = args.task_arguments
        data = _inspect(session, args)
        arguments["data_ref"] = data["data_ref"]
    else:
        arguments = read_json_argument(args.arguments)
    if args.command in ("evaluate", "route") and "data" in arguments:
        if not isinstance(arguments["data"], dict):
            raise _UsageError('data must be an inspection object, e.g. {"data":{"input":"data.csv"}}.',
                              f"gnomon {args.command}")
        if "data_ref" in arguments or (args.command == "evaluate" and "study_id" in arguments and arguments.get('operation') != 'rescore'):
            raise _UsageError("data cannot be combined with data_ref/study_id.", f"gnomon {args.command}")
        data = session.call("gnomon_inspect", {**arguments.pop("data"), "purpose": args.command}, compact=False)
        arguments["data_ref"] = data["data_ref"]
    result = session.call("gnomon_" + args.command, arguments, compact=False)
    if args.command == "evaluate" and args.input and 'data_quality' in data:
        result["data_quality"], result["repairs"] = data["data_quality"], data["repairs"]
    if args.command == "evaluate" and 'study_id' in result:
        result["persistence"] = {"durable": session.ledger is not None,
                                 "next_step": ("Reuse study_id with the same ledger, or pass --study @study.json after --save-result study.json."
                                               if session.ledger is not None else
                                               "Routing needs recorded executions. Evaluate with --ledger-path evidence.db; saving a report alone does not record executions.")}
    return result


def _task_flags(args):
    arguments = {}
    if args.command == 'evaluate' and args.rescore:
        if not args.source_as_of or not args.recorded_as_of:
            raise _UsageError('--rescore requires --source-as-of and --recorded-as-of evidence cutoffs.', 'gnomon evaluate')
        if any(getattr(args, k) is not None for k in ('candidates', 'baseline', 'horizon', 'season', 'series_id', 'folds', 'min_history', 'stride', 'max_calls', 'verify', 'preflight', 'replay')):
            raise _UsageError('--rescore preserves original task parameters and folds; do not supply evaluation overrides.', 'gnomon evaluate')
        return {'operation': 'rescore', 'study_id': args.rescore, 'source_as_of': args.source_as_of,
                'recorded_as_of': args.recorded_as_of, 'allow_partial': not args.no_partial}
    if args.command == 'evaluate' and (args.source_as_of or args.no_partial):
        raise _UsageError('--source-as-of and --no-partial require --rescore.', 'gnomon evaluate')
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
    keys += (["folds", "min_history", "stride", "verify", "preflight", "replay"] if args.command == "evaluate" else
             ["source_as_of", "recorded_as_of", "min_folds", "min_improvement", "require_evidence"])
    arguments.update({key: getattr(args, key) for key in keys if getattr(args, key) is not None})
    if args.command == "evaluate" and args.max_calls is not None:
        arguments["budget"] = {"max_calls": args.max_calls}
    required = ["candidates", "baseline", "horizon"]
    if args.command == "route":
        required = ["study_id", "source_as_of", "recorded_as_of"] if args.study else [*required, "study_id", "source_as_of", "recorded_as_of"]
    missing = ["--" + ("study" if key == "study_id" else key.replace("_", "-")) for key in required if key not in arguments]
    if missing:
        raise _UsageError("--input requires " + ", ".join(missing) + ".", f"gnomon {args.command}")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = None
    try:
        argv = list(sys.argv[1:] if argv is None else argv)
        if argv and argv[0] == "python":
            arguments = argv[1:]
            if arguments[:1] == ["--"]:
                arguments = arguments[1:]
            return subprocess.call([sys.executable, *arguments])
        args = build_parser().parse_args(argv)
        args._explicit_fields = _explicit_fields(argv)
        if args.command == "forecast":
            args.command = "infer"
        _validate_cli_args(args)
        if args.command == "schemas":
            print(json.dumps({"schema_version": "1", "status": "ok", "schemas": _SCHEMA_COMMANDS}, indent=2))
            return 0
        if args.command == 'providers':
            from importlib.resources import files
            source = files('gnomon.examples').joinpath('custom_provider.py').read_text()
            if args.raw:
                print(source, end='')
                return 0
            if args.write_to:
                with Path(args.write_to).open('x') as handle:
                    handle.write(source)
                print(json.dumps({'status': 'ok', 'written': str(Path(args.write_to).resolve())}))
                return 0
            print(json.dumps({'schema_version': '1', 'status': 'ok', 'python': source,
                              'run': 'python -m gnomon.examples.custom_provider'}))
            return 0
        if getattr(args, 'show_resolved_config', False):
            if args.cache or args.provider or args.request or args.config_schema:
                raise _UsageError('--show-resolved-config cannot be combined with cache/schema request flags.', 'gnomon capabilities')
            from .session import resolved_configuration
            print(json.dumps(resolved_configuration(args.providers_config), indent=2))
            return 0
        if getattr(args, "config_schema", False):
            from .session import configuration_schema
            print(json.dumps(configuration_schema(), indent=2))
            return 0
        if getattr(args, "schema", False):
            print(json.dumps(_arguments_schema(args.command), indent=2))
            return 0
        if args.command in {"environment", "releases", "rollback", "update"}:
            from . import installation
            if args.command == "environment":
                result = installation.environment_info()
            elif args.command == "releases":
                result = installation.releases(prune=args.prune, apply=args.apply, keep=args.keep)
            elif args.command == "rollback":
                result = installation.rollback(args.release_id)
            else:
                result = installation.update(args.version)
        elif args.command == "self-check":
            from .selfcheck import leakage_self_check, family_contracts
            result = family_contracts() if args.check == 'families' else leakage_self_check(args.cases, args.seed, args.families, args.detailed)
        elif args.command == "temporal":
            with GnomonSession(enable_temporal=True) as session:
                result = _execute(session, args)
        else:
            with GnomonSession.from_config(getattr(args, "providers_config", None),
                                           ledger_path=getattr(args, "ledger_path", None),
                                           discovery_only=args.command == 'capabilities' or bool(getattr(args, 'preflight', False)),
                                           create_ledger=args.command not in {"ledger", "route"}) as session:
                if args.command == "mcp":
                    from .mcp_server import serve
                    return serve(session=session)
                result = _execute(session, args)
        if getattr(args, "save_result", None):
            from .snapshot_files import write_json
            result["saved_result"] = write_json(args.save_result, result)
        from .diagnostics import completion
        result = completion(result)
        if args.command == 'route' and result.get('fallback_used'):
            print('gnomon: routing used a baseline fallback (' + str(result.get('reason')) + '); evidence-based selection did not complete. Use --require-evidence to reject fallbacks.', file=sys.stderr)
        # Terminals get a compact table; pipes, agents and CI keep the envelope.
        text = _human_text(args.command, result) if not getattr(args, "json", False) and _stdout_is_tty() else None
        print(text if text is not None else json.dumps(result, indent=2, allow_nan=False))
        if result.get("status") == "unscored" or (args.command == "self-check" and result.get('checks_passed') is False):
            return 2
        return 3 if result.get("status") == "partial" else 0
    except GnomonError as exc:
        error = exc
    except FileNotFoundError as exc:
        error = GnomonError("INPUT_NOT_FOUND", str(exc))
    except ForecastAdapterError as exc:
        error = GnomonError("INVALID_ARGUMENTS", str(exc), details=exc.details, repair_options=exc.repair_options)
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
    payload = error.to_dict(compact=getattr(args, 'compact_errors', True))
    if args is not None and getattr(args, "input", None):
        # Source failures and provider validation must retain the same semantic
        # context, including units, series selection, repairs and cutoffs.
        payload["error"]["details"]["input_options"] = {
            key: getattr(args, key) for key in (*_INPUT_KEYS, "provider", "horizon", "season", "series_id", "quantiles")
            if getattr(args, key, None) is not None}
        options = payload['error']['details']['input_options']
        explicit = {k: v for k, v in options.items() if k in args._explicit_fields or k == 'input'}
        payload['error']['details'].update(supplied_arguments=explicit, explicit_input_options=explicit,
            defaulted_input_options={k: v for k, v in options.items() if k not in explicit})
    if error.code == "INVALID_ARGUMENTS" and args is not None and args.command in _EXAMPLES:
        details = payload["error"]["details"]
        if args.command == "infer" and getattr(args, "input", None):
            details.pop("example_arguments", None)
            details.pop("example_kind", None)
            details.pop("changed_fields", None)
            details["guidance"] = error.details.get("guidance", "Keep these input options and correct the reported issue; frozen snapshots cannot be overridden. Inspect source data again to change a cutoff or timezone.")
        elif args.command == "infer" and "example_arguments" in details:
            # Shared MCP examples include a provider wrapper; --request expects its request object.
            example = details["example_arguments"]
            if isinstance(example, dict) and "request" in example:
                details["example_arguments"] = example["request"]
        else:
            if "example_arguments" not in details:
                details["example_arguments"] = json.loads(_EXAMPLES[args.command])
                details["example_kind"] = "schema_illustration"
                details["example_guidance"] = ("This is a generic schema illustration, not a retry of your task. "
                                               "Use your actual input, providers, IDs, horizon and cutoffs.")
        details.setdefault("schema_command", f"gnomon {args.command} --schema")
        if args.command == "infer" and "required_history" in details:
            details.pop("example_arguments", None)
            details["request_parameters"] = {"season": details["season"], "horizon": details["horizon"]}
    from .diagnostics import recovery_metadata
    if error.code == 'MISSING_COLUMNS' and args is not None and getattr(args, 'input', None):
        from .recovery import column_recovery
        details = payload['error']['details']
        details.update(column_recovery(details, details.get('supplied_arguments', {})))
        details['argument_basis'] = 'explicit_cli_arguments'
        if details.get('example_kind') == 'task_correction':
            correction = list(argv)
            flag_index = next((i for i, v in enumerate(correction)
                               if v in {'--time-column', '--time'} or v.startswith(('--time-column=', '--time='))), None)
            if flag_index is None:
                correction += ['--time-column', 'ts']
            elif '=' not in correction[flag_index]:
                correction[flag_index + 1] = 'ts'
            else:
                correction[flag_index] = correction[flag_index].split('=', 1)[0] + '=ts'
            details['next_call'] = {'argv': ['gnomon', *correction], 'runnable': True, 'admissible': None}
    details = payload['error']['details']
    handoff = details.get('next_call')
    if args is not None and getattr(args, 'input', None) and isinstance(handoff, dict) and 'repair' in handoff.get('arguments', {}):
        # The repair handoff names the tool call; give the CLI caller the exact argv instead.
        correction, mode, skip = [], handoff['arguments']['repair'], False
        for item in argv:
            if skip:
                skip = False
                continue
            if item == '--repair':
                skip = True
                continue
            if item.startswith('--repair='):
                continue
            correction.append(item)
        correction += ['--repair', mode]
        details['next_call'] = {**{k: v for k, v in handoff.items() if k not in {'tool', 'arguments'}},
                                'argv': ['gnomon', *correction]}
        details['data_quality']['next_call'] = details['next_call']
    if args is not None and str(getattr(args, 'input', '')).endswith('.gnomon') and details.get('rejected_fields'):
        from .recovery import frozen_recovery
        rejected = details['rejected_fields']
        details.update(frozen_recovery(details.get('supplied_arguments', {}), rejected))
        flags = {'--' + key.replace('_', '-') for key in rejected}
        correction = []
        skip = False
        for item in argv:
            if skip:
                skip = False
                continue
            if item.split('=', 1)[0] in flags:
                skip = '=' not in item
                continue
            correction.append(item)
        details['next_call'] = {'argv': ['gnomon', *correction], 'runnable': True, 'admissible': True}
    payload['error']['recovery'] = recovery_metadata(details, cause=error.message, cause_code=error.code)
    if getattr(args, 'save_result', None):
        from .snapshot_files import write_json
        try:
            payload['saved_result'] = write_json(args.save_result, payload)
        except (OSError, ValueError):
            payload['save_result_error'] = {'path': str(args.save_result), 'status': 'not_saved',
                'message': 'Rejection payload could not be saved; the original rejection is retained here.'}
    print(json.dumps(payload, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
