# CLI reference

Use `-` as the input for piped CSV (up to 8 MiB); larger inputs need a file.
This works with inspect/describe, `infer --input -`, and evaluate/route with direct
`--input -` flags. The CLI freezes input
before computing results and removes its temporary file when inspection finishes.

One CLI uses the same execution session as Python and MCP. Its single structured
JSON response always goes to stdout: success exits 0, errors or unscored evaluations
exit 2, partial evaluations exit 3, and interruption exits 130.
Partial/unscored evaluations still return their report, including `issues`, row
requirements and actual call/fold counts. stderr is reserved for unstructured process diagnostics. Run
`gnomon --help` or a command's `--help`.
CLI usage errors retain `error.code`, `error.message` and help guidance without
the evidence-rejection envelope.
`gnomon python` is an interpreter passthrough: its arguments, stdout, stderr and
exit status belong to Python, rather than the CLI's JSON response contract.
Install/update progress goes to stderr; update returns its JSON result on stdout.

| Command | Purpose |
| --- | --- |
| `gnomon capabilities` | Registered providers, limits and enabled tools |
| `gnomon infer` | Explicit provider inference; no implicit backtest |
| `gnomon inspect` | Validate a file/store input and disclose snapshot semantics |
| `gnomon describe` | Exact mean, median, latest, minimum, maximum or sum |
| `gnomon evaluate` | Budgeted rolling-origin comparison |
| `gnomon route` | Select from a recorded study at explicit cutoffs |
| `gnomon ledger` | Read evidence, append scores or authorized outcomes |
| `gnomon temporal` | Explicit date, instant, interval and ordering calculations |
| `gnomon self-check leakage` | Offline check of snapshot cutoffs |
| `gnomon mcp serve` | Long-lived stdio session for an agent |
| `gnomon environment` | Exact interpreter, import path and build identity |
| `gnomon python` | Run scripts or `-c` code in Gnomon's Python environment |
| `gnomon releases` | List standalone installs; `--prune` previews cleanup |
| `gnomon update` | Install and activate a Git ref after validation |
| `gnomon rollback` | Activate an existing standalone release by ID |

`infer --schema` prints the JSON object accepted by `--request` without requiring
a provider. `ledger --schema` prints all operation shapes without opening a
database; outcome-write operations identify their operator authorization requirement.
Both commands include examples in help and argument errors. Provider-name errors
list registered names and point to `capabilities` with the same provider config.

`ledger` and `route` require an existing Gnomon ledger. A missing path reports
`LEDGER_NOT_FOUND` with the resolved path instead of creating an empty database.
`infer` and `evaluate` create a ledger when asked to record evidence there.

Ledger outcome writes require an operator TOML containing
`allow_outcome_writes = true`, supplied with `--providers-config providers.toml`.
Request JSON cannot enable writes. For `actuals_as_of` and `compare_history`, pass
the same `unit` used when recording: omitted/null selects unitless data. `search`
can discover executions across units when its unit filter is omitted.

Cutoffs use source availability (`source_available_at`), separately from the
observation's `valid_time`. Search/status operations default to the current clock;
an actual with a later source availability is invisible at that cutoff. Supply
`source_as_of` explicitly to query another cutoff. Raw `actuals_as_of` and
`evaluate` leave omitted cutoffs unbounded; use both `source_as_of` and
`recorded_as_of` for a reproducible historical view.

```bash
gnomon infer --provider last_value --request '{"history":[1,2,3],"horizon":2}'
gnomon infer --provider last_value --input data.csv --horizon 7
gnomon inspect data.csv --time-column date --target-column sales
gnomon inspect --input data.csv --time-column ts
gnomon describe data.csv --statistic mean --start 2025-01-01 --end 2025-01-31
gnomon evaluate --arguments '{"data":{"input":"data.csv"},"candidates":["historical_mean"],"baseline":"last_value","horizon":2,"folds":4,"budget":{"max_calls":8}}'
gnomon ledger --providers-config providers.toml --arguments '{"operation":"pending"}'
gnomon mcp serve --providers-config providers.toml
```

`--request` and `--arguments` accept a JSON object or `@file.json`.
`infer` takes either a typed request or `--input` plus `--horizon`.
File options are `--time-column` (default timestamp for every provider), `--target-column`
(default value), `--series-column`, `--frequency`, `--as-of`,
`--recorded-as-of`, `--store-path`, `--unit`, `--repair`, `--regrid`, `--timezone`
and `--window latest_contiguous`.
They cannot be mixed with a typed request. Panel forecasts require `--series-id`
unless only one series is present. Repairs default to off.

Forecast responses include `request_provenance` with the execution's cutoff,
known-time and recorded-time cutoffs, snapshot ID, selected series and history
bounds. File/store/saved-snapshot inference also returns a top-level `snapshot`,
including `as_of` and per-series `accesses[].max_known_time`, across CLI, Python
and MCP. `recorded: false` means no durable ledger recording; it does not mean
the cutoff was ignored. Typed requests label their provenance as caller-supplied;
Gnomon does not independently verify their source timestamps or snapshot IDs.

Column names are explicit: a `ts,value` CSV needs `--time-column ts` for both
`seasonal_naive` and `historical_mean`. Defaults do not depend on the provider.

| Repair mode | What it can change | Interior gaps |
| --- | --- | --- |
| `off` | Strict parsing and validation | Refused |
| `safe` | Formatting, identical duplicates, bounded timestamp jitter | Refused; does not invent values |
| `aggressive` | Safe fixes plus bounded interpolation, conflicting duplicates and row dropping | Filled when within repair limits, with disclosure |

For observed-only historical evaluation, select a contiguous window rather than
interpolating. A repair may be safe for inference yet unsuitable for historical
scoring; inspection reports that distinction.
For file inference, `--season 7` sets a weekly seasonal period for daily data;
the default period is 1, which makes seasonal naive repeat the last value.
`--quantiles 0.1 0.5 0.9` requests supported provider quantiles. These flags cannot
be mixed with `--request`; put `season`/`quantiles` inside that request instead.

Inspection always reports `readiness` for inference, evaluation and routing.
`inspect --for evaluate` or `inspect --for route` fails early if the data cannot
support that next operation. Readiness checks data suitability; evaluation still
checks the requested horizon, history, folds, provider capabilities and budget.
Date-only or naive timestamps need an explicit `--timezone UTC` (or the source's
actual IANA zone) for routing. Ambiguous/nonexistent daylight-saving local times
require explicit offsets; Gnomon does not guess.

Value-preserving, independent format fixes can be evaluated. Gap interpolation,
timestamp shifting and formats inferred from other rows cannot be historical
truth. For gapped data, an explicit `--window latest_contiguous --frequency D`
selects the latest uninterrupted observed segment in each series, without filling
values. Excluded row counts and the selected interval are disclosed in repairs.
Use the original source with `--repair off` for this recovery. If too little
history remains, evaluation reports the required row count instead of claiming success.

`describe` accepts `--series-id`, inclusive `--start`/`--end` and the
same file options. It computes observed statistics, not forecasts or causal claims.
A CLI data reference ends with the process. To reuse frozen data across processes,
run `inspect --save-snapshot data.gnomon`, then pass `--input data.gnomon` to the
next command. Portable snapshots retain observations, revisions, schema, units,
repairs and cutoffs; they do not reopen the original source. They are bounded to
64 MiB, use JSON with a corruption checksum, and preserve named timezones.
A checksum detects accidental changes; it is not an authenticity signature.
Do not repeat schema/repair/timezone options when loading a saved snapshot.
Python and MCP can also inspect a saved `.gnomon` file; Python exposes
`session.data.save(data_ref, path)` for explicit export.
Both `inspect` and `describe` accept either a positional input or `--input`,
including `--input -` for stdin. Supplying both is a usage error.

Provider-backed commands accept explicit operator `--providers-config` TOML.
`gnomon capabilities` includes `providers.<name>.request_schema`, with the JSON
request field names, defaults and declared provider limits. Seasonal naive uses
`"season": 7` for a seven-observation period, equivalent to `infer --season 7`.
The default is 1; history must contain at least that many observations.
Infer, evaluate, route and ledger also accept `--ledger-path evidence.db` without
a TOML file. Route/ledger require a ledger configured by either method; conflicting
paths are rejected. Evaluation/routing JSON may contain a `data`
inspection object, replaced with a frozen reference before the operation.
They use the [same contracts and budgets](production/INFERENCE.md) as MCP.

`gnomon evaluate --help` and `gnomon route --help` include direct-flag and JSON examples.
Use `gnomon evaluate --schema` or `gnomon route --schema` for the full CLI JSON
Schema, including nested `data` and budget fields; no config is needed to print it.
Pass the operation object directly, without an `arguments` or `tools/call` wrapper.
For example, `data` is `{"input":"data.csv","time_column":"ts"}`, not a path string
or an array of rows. `budget` is an object such as `{"max_calls":8}`, not a number.

Evaluation includes `ranking_policy`: ascending MAE on matched folds, with exact
unrounded ties listed explicitly. Input order within a tie is only a display
convention. Choose tied providers using known cost, latency or simplicity, or
collect more evidence; equal MAE does not establish equal predictions or future
performance. An unscored evaluation has no ranking or ties.

The direct-flag workflow needs no JSON construction or manual copying of IDs.
For daily `timestamp,value` data ending on 2026-01-31, with UTC as its declared
source timezone:

```bash
gnomon inspect --input data.csv --timezone UTC --for route --save-snapshot data.gnomon
gnomon infer --input data.gnomon --provider seasonal_naive --season 7 --horizon 2
gnomon evaluate --input data.gnomon --candidates historical_mean --baseline last_value --horizon 2 --folds 4 --max-calls 8 --ledger-path evidence.db --save-result study.json
gnomon route --input data.gnomon --study @study.json --ledger-path evidence.db --source-as-of 2026-01-31T00:00:00Z --recorded-as-of 2099-01-01T00:00:00Z
```

`--save-result` writes result JSON atomically while preserving stdout output.
`--study @study.json` supplies the recorded study ID, candidate names, baseline,
horizon, season and series; forecasts are still verified against the ledger.
Saving a report alone does not record executions needed for routing: use the
same ledger for evaluation and routing. The ledger location is an explicit
argument/configuration, never a path loaded from a study report.
Direct route flags apply the source cutoff to file/store inspection and the
recorded cutoff to store inspection; saved snapshots retain their original cutoffs.

The JSON interface remains available. To evaluate and then route across CLI
invocations, save the study in a ledger.
Create `providers.toml` (TOML, not JSON or INI):

```toml
schema_version = 1
ledger_path = "ledger.db"
```

The ledger path is relative to the config file. For this example, `data.csv` must
have `timestamp,value` columns, daily timezone-aware timestamps through
`2026-01-31T00:00:00Z`, and enough history for four two-step folds (at least 16 rows
with the default minimum history). Run:

```bash
gnomon evaluate --providers-config providers.toml --arguments '{"data":{"input":"data.csv"},"candidates":["historical_mean"],"baseline":"last_value","horizon":2,"folds":4,"budget":{"max_calls":8}}' > study.json
```

Copy `study_id` from `study.json` into `route.json`:

```json
{
  "data": {"input": "data.csv"},
  "study_id": "REPLACE_WITH_STUDY_ID",
  "candidates": ["historical_mean"],
  "baseline": "last_value",
  "horizon": 2,
  "source_as_of": "2026-01-31T00:00:00Z",
  "recorded_as_of": "2099-01-01T00:00:00Z"
}
```

```bash
gnomon route --providers-config providers.toml --arguments @route.json
```

Replace the example cutoffs with your analysis cutoffs. `source_as_of` is the last
visible observation; to select an earlier cutoff, also set `data.as_of` when
inspecting. `recorded_as_of` must be at or after the study's recording time for
that study to be available; the example's 2099 cutoff includes all evidence
recorded so far. Routing uses the saved study without new provider calls.

The forecast cache is **off by default**, in memory, and scoped to a session.
`cache_size = 128` in operator TOML enables it for versioned deterministic
providers. Each CLI invocation starts a fresh session: even identical `infer`
commands cannot hit a previous process's cache, and a ledger does not persist it.
`--no-cache` bypasses lookup; it has no practical effect on a standalone invocation
with an empty cache. Use a long-lived Python or MCP session for reuse.
Capabilities disclose cache policy, and forecast results include `cache.status`
(`disabled`, `bypassed`, `ineligible`, `miss` or `hit`), scope and persistence,
alongside the existing `cache_hit` boolean. `reference_scope` describes data
references separately from forecast caching.

`gnomon temporal --schema` prints the full schema for normalize, duration, shift,
interval and order_events. Its help includes an interval
example with `left` and `right` objects, each containing `start` and `end`.
`gnomon temporal --arguments` uses the [temporal contract](production/TEMPORAL.md).
Each operation's schema contains a runnable example. Argument errors return an
`example_arguments` object for the attempted operation, including for `@file.json`
input. Unknown or missing operations return a fallback example and supported names.
`gnomon self-check leakage --cases 8 --seed 7` tests installed cutoff behavior,
not LLM reasoning or forecast accuracy.

Optional ledger/temporal tools are enabled in operator configuration.

`gnomon environment` explains the distinction between the `gnomon-forecast`
distribution and `gnomon` import, and identifies the exact Python interpreter.
Use `gnomon python your_script.py` to use the standalone environment's API.
`gnomon releases`, `gnomon update --version REF`, `gnomon rollback RELEASE_ID`
and `gnomon releases --prune [--keep N] [--apply]` manage standalone installs.
See [installation](installation.md) for version requirements, build fingerprints
and rollback/cleanup behavior.
