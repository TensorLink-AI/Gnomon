# Operating providers and the temporal ledger

Start with the [executable provider/ledger walkthrough](../../examples/provider_plugin/README.md).
It exercises the same installed extension from Python, CLI and MCP, including
revised actuals, historical scores and a checked backup/restore. No remote model,
paid call, real decision or live-service verification is implied by that example.

## Startup and permissions

Install reviewed Gnomon and extension wheels into a dedicated environment. Pass
the provider TOML explicitly with `--providers-config`; there is no ambient
working-directory provider discovery. Relative ledger paths resolve beside this
configuration file. Installed callables/factories are trusted operator code and
run with the host account's privileges. They are not sandboxed by Gnomon.

Use an OS account with access only to the intended data/configuration directories.
Inspection accepts caller-selected file/store paths; a temporary working directory
does not constitute a filesystem security boundary. Set restrictive file creation
permissions before first use (for example, `umask 077` on Unix), and separately
review permissions of existing databases, artifacts and backups.

The default session has six tools, or eight with a ledger. Outcome/decision writes
and legacy imports require `allow_outcome_writes=true` at startup. Leave it false
unless the agent is authorized to submit those records. Direct Python ledger writes
are operator actions. Evaluation appends scores; neither scoring nor a recorded
`authorization_ref` executes or grants permission for a business action.

Discover provider identity/capabilities before inference:

```bash
gnomon capabilities --providers-config /absolute/path/to/providers.toml
```

With remote discovery enabled, startup can send `GET /models`. A forecast sends
the provider request; optional evaluation can send several budgeted requests.
Choose limits and service spending policy accordingly. A request timeout does not
prove the remote work stopped or cost nothing. Gnomon does not automatically replay
forecast POSTs. Reissuing a call creates new work and a new execution identity.

Ephemeris base URLs are deployment-specific and may include a prefix. Configure
the URL and token through the named environment variables in the provider TOML;
do not put credentials in agent arguments or committed example files. Discovery
and local HTTP tests do not attest remote weights or establish SOTA performance.
Actual deployment verification remains a distinct release gate.

## What the ledger preserves

`ledger.db` is a separate SQLite database, not the old mutable `TrackingStore` or
the observation `TemporalStore`. It preserves unique executions, content-addressed
request/result payloads, actual revisions, evaluations and decisions. Updating or
deleting records through its SQL schema is refused. This is not protection against
an administrator replacing files or changing the schema, nor encrypted storage.

Keep these times distinct:

- Valid time: when the observation applies.
- Source availability: when that exact vintage was available at its source.
- Recorded time: when this system recorded it, from its clock.

An actual correction is appended, then rescored into a **new** evaluation. The
original forecast and old evaluations do not change. Use both source and recorded
cutoffs for replay of what this system could have known; a source cutoff alone can
include a vintage ingested later. Unknown legacy recording times remain unknown.
Units are explicit labels; matching labels is not a unit conversion or proof that
the source supplied the correct quantity.

SQLite writes are serialized with a 30-second lock timeout. Independent local
processes are tested; a network filesystem or multi-host database deployment is
not certified. Native batched inference is not a whole-batch atomic transaction.
Result/data references expire with their session or retention limits; retain
execution/study IDs and the ledger for durable retrieval.

## Find, score and reuse experience

These operations use the existing `gnomon_ledger` tool, or the same named Python
methods. No background worker or additional store is needed.

```python
page = ledger.search(series_id="sales", horizon=7, status="ready", limit=20)
ids = [row["execution_id"] for row in page["items"]]
if ids:
    scores = ledger.evaluate(execution_ids=ids, allow_partial=False)
```

Search returns summaries, execution IDs and up to 20 associated study IDs (with a
truncation flag). Filters include series, horizon, provider, unit and inclusive
recording-time `start`/`end`. The maximum page size is 100. Each page examines at
most 200 filtered executions; follow `next_cursor` even if a status-filtered page
is empty. Repeat the same filters; the cursor preserves cutoffs and an execution
high-water mark. Default cutoffs are the ledger clock at the first page. Supply
explicit source/recorded cutoffs for historical reads. Omitted search unit means
all units; returned labels remain explicit and must not be pooled.

Actual coverage and score freshness are separate: `score_state` is `unscored`,
`current` or `stale`. The `status` filter uses `waiting` for incomplete actuals,
`ready` for complete unscored forecasts, `stale` for complete forecasts whose score
needs refreshing, `scored` for current complete scores, and `unscorable` for missing
identity/time semantics. Missing-step previews contain at most 20 positions; use
`missing_count` and the execution's full target grid for the complete requirement.
`pending()` remains an unbounded compatibility read, now including ready/stale work;
it returns the complete missing-step list. New agents should use search.
No queue state is persisted. Unknown units can be supplied as JSON `null` or omitted.

Authorized ingestion accepts `ledger.append_actual(actuals=[...])`, at most 1,000
rows with the same fields as a single actual. Evaluation accepts at most 100
distinct execution IDs. A failed batch rolls back all its writes. Identical
actual submissions are idempotent; evaluation with identical actual IDs, matched
steps, metric version and source/recorded cutoff arguments returns the existing
score. Changed evidence appends a score. Neither ingestion nor search automatically
scores forecasts. Study truth remains in study payloads, not online actuals.

For matched production evidence:

```python
comparison = ledger.compare_history(
    series_id="sales", unit="USD", horizon=7,
    providers={"model-a": "weights-and-config-v1", "model-b": "weights-and-config-v3"},
    start="2026-08-01T00:00:00Z", end="2026-08-20T00:00:00Z",
    source_as_of="2026-09-01T00:00:00Z", recorded_as_of="2026-09-01T00:00:00Z",
)
```

Here `start`/`end` select **forecast origins**, not recording times. Unit matching
is exact (omitted means unknown/null). Two to eight explicit provider revisions
are required; version labels must identify weights and configuration. At most
1,000 executions are considered; larger windows fail without a partial ranking.
The comparison uses one database read snapshot and reuses the single-origin input
matching checks. It takes the first recorded retry, excludes ambiguous inputs,
requires complete matched horizons, and averages their MAEs equally. It reports
execution/actual IDs, missing providers, exclusions, origin counts, total forecast
steps and unique actuals. Overlapping horizons are not independent observations;
the output makes no statistical-significance or future-performance guarantee.

Declared frequencies must match the target grid, including calendar month/day
steps. The origin's offset from the last observation must also match across origins.
Without a frequency, exact elapsed lead-time vectors must match; one-day and
seven-day forecasts are not pooled just because both have horizon one. Unresolved
or inconsistent declared grids are excluded with a reason. Individually matched
origins and their sample counts remain visible when the overall task/provider
cohort is incompatible, but no aggregate model scores are returned.

Backtest/imported runs are excluded from production comparisons. Forecasts must
have been recorded before their first target timestamp; historical input cutoffs
and declared pretrained training cutoffs are checked. New executions preserve
provider lifecycle/capabilities. Older records without this identity remain
readable and scoreable but cannot qualify for production comparison; it is not
inferred during upgrade. Identity declarations are not independent attestations.

Comparison does not call providers, append scores or change `gnomon_route`'s
original-backtest contract. Only observed forecasts can supply matched evidence:
if another model never ran, its performance is unknown. `next_step` gives guidance,
not authorization to collect data, run evaluations or act on a recommendation.

## Backup before an upgrade

Use SQLite's backup facilities, not a plain copy of a live `.db` file that could
omit committed WAL data. This repository includes a runnable, read-only-source
backup example; it refuses an existing destination, creates a private file and
checks SQLite integrity/foreign keys:

```bash
python examples/provider_plugin/walkthrough.py backup \
  --source /absolute/path/to/ledger.db \
  --destination /absolute/path/to/new-backup.db
```

The source is not opened as `TemporalLedger`, so backing up a v1/v2 ledger does
not migrate it. The destination's parent directory must already exist. An error
or interruption may leave a partial newly created destination; do not restore
from it without validation. Choose a new destination for a retry. Keep backups
outside the live data directory, with your normal access/retention controls.

For an upgrade, stop application writers, create and verify the backup, and make
another backup copy into a **new candidate path**. Then open only the candidate
with the new runtime:

```bash
python -c 'import sys; from gnomon import TemporalLedger; ledger = TemporalLedger(sys.argv[1]); print(ledger.pending())' \
  /absolute/path/to/candidate-ledger.db
```

Schema v2 added import identities; v3 added immutable studies and their execution
links. Opening v1/v2 with the current class upgrades transactionally. Failed
upgrades roll back schema/version changes and can be retried. Unrelated databases
and newer unsupported schema versions are refused; do not force a version using
`PRAGMA user_version` in production.

Verify representative execution IDs, score vintages, pending work and application
queries on the candidate, then explicitly update the operator configuration after
review. Keep the previous runtime and untouched backup. Rolling back the executable
does not downgrade a database: restore to a separate path. Reconcile writes accepted
after the backup before any cutover; blindly restoring an old snapshot can lose
new executions/outcomes. Neither this example nor Gnomon automates deployment.

## Importing historical artifacts

For a different legacy store, use explicit import methods into a separate ledger,
not schema conversion in place:

```python
from gnomon import TemporalLedger

ledger = TemporalLedger("new-ledger.db")
report = ledger.import_tracking("old-registry.db", project="sales")
print(report["execution_ids"], report["skipped"])
# Or: ledger.import_artifact("old-artifact-directory", project="sales")
```

The old registry is read-only and unchanged. Re-importing the same artifact under
the same namespace/timezone binding is idempotent. Missing artifacts are listed;
missing original histories stay null. Overwritten legacy score vintages cannot be
reconstructed. Imports are not fresh inference or fully matched-input evidence.

Naive historical times remain unresolved for calendar scoring unless the operator
explicitly supplies `naive_timezone="UTC"`; this is a recorded assumption, not an
inferred timezone. Do not add it merely to make an otherwise invalid comparison
pass. The observation store's recorded-time migration likewise does not invent
historical recording times.
