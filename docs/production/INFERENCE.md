# Direct inference and the temporal ledger

This is the implemented extension boundary, shared by Python `GnomonSession`,
`gnomon infer`, and MCP. Backtesting is an explicit
`session.evaluate(...)` operation, not an implicit step in forecasting.
Live-service and actual agent-comparison acceptance remain pending; see
[validation and limits](../agent-evaluation.md).

For a complete runnable starting point, use the
[installable provider/ledger example](../../examples/provider_plugin/README.md).
[Operating instructions](OPERATIONS.md) cover backup and permissions.

## Shared session configuration

```toml
# providers.toml: explicitly supplied by the operator, never a tool argument
schema_version = 1
ledger_path = "ledger.db"  # relative to this configuration file
allow_outcome_writes = false
enable_temporal = false  # optional explicit date/interval/event calculations

[providers.remote]
kind = "ephemeris"
base_url_env = "EPHEMERIS_BASE_URL"
token_env = "EPHEMERIS_API_TOKEN"
mode = "route"
discover = true  # GET /models at startup; names become remote/<model>

[providers.preferred]
kind = "callable"
entrypoint = "my_project.forecasting:forecast"
revision = "my-project-config-v1"
```

Python: `GnomonSession.from_config("providers.toml")`. CLI:
`gnomon infer --providers-config providers.toml --provider remote --request @request.json`.
MCP: `gnomon mcp serve --providers-config providers.toml`.
There is no automatic working-directory configuration search. Python applications
can instead pass an already-configured engine to `GnomonSession(engine)` and then
to `gnomon.mcp_server.serve(session=session)`.

The MCP server exposes capabilities, inspect, describe, forecast, evaluate, read and optional
ledger operations. With a ledger it also exposes cutoff-bound study routing.
Forecast calls supply registered provider names and typed data or frozen references;
service URLs, provider entrypoints, credentials and ledger paths are startup configuration.
Inspection reads caller-selected file/store paths.
Built-in last_value, seasonal_naive and historical_mean references are registered
by `from_config`, including when no file is supplied.

`enable_temporal=true` adds the independent `gnomon_temporal` tool at startup;
ordinary discovery remains six tools (eight with a ledger). It does not change
provider behavior, grant write permissions or load advanced context machinery.
See [the temporal calculation contract](TEMPORAL.md) for exact operations and limits.

Set `kind="factory"` with a zero-argument entrypoint for fresh fitting state.
Optional `[providers.preferred.capabilities]` fields use AdapterCapabilities.
Outcome submissions and decision/outcome writes require
`allow_outcome_writes=true`; tool arguments cannot enable this permission.
Scoring is allowed with a configured ledger and appends immutable evaluations.

## Frozen data references and exact summaries

```python
with GnomonSession.from_config() as session:
    inspected = session.call("gnomon_inspect", {
        "input": "sales.csv", "time_column": "date", "target_column": "sales",
        "unit": "USD", "frequency": "D",
    })
    ref = inspected["data_ref"]
    summary = session.call("gnomon_describe", {
        "data_ref": ref, "statistic": "mean",
        "start": "2025-01-01T00:00:00Z", "end": "2025-01-31T00:00:00Z",
    })
    forecast = session.call("gnomon_forecast", {
        "provider": "last_value", "data_ref": ref, "horizon": 7,
    })
```

References retain the inspected snapshot, including observation vintages, and
never reopen the original file/database. A changed input gets a different reference.
They live only within the session: defaults retain at most 16 references and 100,000
vintage rows in total, with least-recently-used eviction. Operator settings
`max_data_refs` and `max_data_rows` adjust retention, not parser peak memory;
deployments must also constrain input size. Expired references fail explicitly.
Select one exact `series_id` when a source contains multiple series.

`describe` computes mean, median, latest, minimum, maximum or sum over an inclusive
start/end timestamp window (or all frozen observations if omitted). It reports the
actual window and count. Units are caller-declared labels, never inferred conversions;
no cross-unit aggregation is performed. Timestamp windows must match the data's
timezone awareness. Unknown window, unit or aggregation arguments are rejected.
Supported arithmetic is neither calibration nor action permission. Other analyses
belong in the user's chosen software, not invented describe operations.

Inspection and backtesting use one loader. `as_of` bounds source
availability before any file repair; `recorded_as_of` additionally bounds locally
recorded vintages for `store:<dataset>` inputs. Plain files cannot reconstruct
recording-time history and reject that option. Repair is off unless explicitly
requested and every repair is disclosed. File values assume availability at their
valid timestamp; this assumption is not proof of historical availability.

Reference forecasts derive the future grid from the last observed timestamp and
validated frequency. They reject stale-history grids whose next step is already
at/before `as_of`, rather than silently backcasting under a future label. Raw typed
requests remain available for explicitly constructed tasks and covariates. General
covariate/reference joins and irregular grids are not implicitly guessed.

The one-shot CLI freezes and forecasts in one session:
`gnomon infer --input sales.csv --time-column date --target-column sales --frequency D --horizon 7 --provider last_value`.
Its response includes inspection provenance; a returned reference cannot be reused
in another CLI process. Long-lived Python/MCP sessions can reuse references.
For cross-process input reuse, explicitly export `session.data.save(ref, "data.gnomon")`
or `gnomon inspect --save-snapshot data.gnomon`, then inspect that saved input.
Portable files preserve frozen vintages and repairs; they never reopen mutable
sources. Inspection returns per-operation `readiness`, and `purpose="route"`
checks timezone and repair suitability before a costly evaluation. Declare a
file's IANA `timezone` explicitly when its timestamps have no offset.
`window="latest_contiguous"` with an explicit frequency selects observed history
after the last gap per series, disclosing the excluded rows and selected interval.

## Bring your own software

Users own their library objects, fitting configuration and conversions. Gnomon
requires one callable, `ForecastRequest -> ForecastResult`. It neither imports
StatsForecast/NeuralForecast/Darts on your behalf nor guesses how their outputs
map to timestamps, targets or uncertainty.

```python
from gnomon import ForecastRequest, ForecastResult, InferenceEngine

def my_forecast(request):
    # Replace this baseline with your preferred library's fit/predict calls.
    points = (request.history[-1],) * request.horizon
    return ForecastResult(
        point=points,
        timestamps=request.future_timestamps,
        series_id=request.series_id,
        unit=request.unit,
    )

engine = InferenceEngine()
engine.register("my-model", my_forecast, revision="my-config-v1")
run = engine.forecast("my-model", ForecastRequest(history=(1, 2, 3), horizon=2))
assert run.result.point == (3, 3)
assert run.evidence == "inference_only"
assert not run.action_authorized
```

Declare optional capabilities with `AdapterCapabilities`. Unsupported covariates,
panels, probabilities, sample counts, history lengths and horizons are rejected
before dispatch. Requested uncertainty must be returned with finite, aligned,
monotone marginal quantiles or aligned sample paths. Native uncertainty is not
proof of calibration.

`forecast_adapter.conformance_report` makes three explicit calls and checks output
shape, finite values, request immutability and variable horizons. Stochastic outputs
are valid by default; set `require_deterministic=True` to require equal numeric
results. Per-call metadata such as request IDs is excluded from that equality
check. This protocol test does not establish accuracy, training cutoffs or calibration.

Past/future covariate matrices are row-major: one row per history/horizon step.
Related series are series-major, with each series history-aligned. Optional
column names preserve covariate identity. Result timestamps, series identity
and units must exactly echo the request. The engine does not infer units or
resample data. Plain positional requests remain possible without timestamps;
they cannot be matched to calendar outcomes later without explicit identities.

For fitting software that retains mutable state, use
`engine.register_factory(name, factory, ...)`. The zero-argument factory returns
a fresh callable or object implementing `forecast(request)` for every invocation,
including each batch member/fold. An optional `close()` is called after use.
`register(..., lifecycle="pretrained")` retains an already-loaded provider.
This isolates provider instances; arbitrary user Python is not a security sandbox
and must not read future training data from external state.

Providers may implement `forecast_batch(requests)` returning results in request
order. The engine validates all batch inputs before dispatch and all returned
shapes before recording results. Factories use independent per-request calls.

The in-memory cache is off by default. Enabling `cache_size` also requires an
explicit revision and `deterministic=True` on the provider. Fingerprints include
the exact request, snapshot and cutoffs. Every cache hit still gets a unique
execution ID. A revision is a provider assertion, not an attestation by Gnomon.

## Optional budgeted evaluation

```python
with GnomonSession.from_config("providers.toml") as session:
    ref = session.data.inspect("sales.csv", time_column="date", target_column="sales")["data_ref"]
    study = session.evaluate(
        ref, candidates=["preferred"], baseline="last_value", horizon=7,
        folds=4, min_history=28, stride=7, budget={"max_calls": 8},
    )
```

The public lower-level function is `evaluate_reference(engine, references, data_ref,
..., budget=EvaluationBudget(...))`. It uses registered request/result providers;
fresh factories are reconstructed and closed per fold. Inference caching is bypassed.
Every provider receives exactly the same history, timestamps, cutoffs, horizon,
season and unit for each fold. This release's new evaluator scores point forecasts;
it does not guess covariate vintage joins or claim probability calibration.
Reports include `issues` with actionable history and budget diagnostics and
`routing_readiness` for the inspected data. CLI evaluations return exit 2 for an
unscored report and exit 3 for a partial report. MCP marks unscored evaluations
with `isError: true`, including compact result receipts; the report remains
available for diagnosis. Independent value-preserving format fixes are allowed;
format inference using other rows, interpolation and timestamp restamping remain
excluded from historical scoring.

An explicit baseline is mandatory and counts toward all limits. Defaults permit at
most four providers (baseline included), eight folds and 32 provider attempts.
Session dispatch has a 30-second wall budget; startup TOML `[evaluation_limits]`
can set `max_providers`, `max_folds`, `max_calls` and `max_seconds`. Tool calls may
only tighten those limits. Python's lower-level budget has no time limit unless
specified. Failures count as attempts; no implicit retry or hidden fallback model
is run. Ephemeris may fan out internally, so a provider-call budget is not a GPU
model-call or monetary budget. The report explicitly marks internal counts unknown.

Time/cancellation checks stop *new dispatches*. They cannot safely kill an arbitrary
running Python callable; a slow final call can exceed the wall limit and that
overrun is reported. Providers should also enforce their own timeouts. Python may
pass a `cancelled()` callback; KeyboardInterrupt preserves already completed work.
The current synchronous stdio server does not promise concurrent MCP cancellation.

Folds are trailing expanding-window origins on the validated grid. Source-time
replay reads only the vintages available at each origin. Recorded-time replay also
narrows local recording time independently at every origin. Both remain bounded by
the parent frozen snapshot, and neither rereads mutable input. Source-only replay
is the default unless inspection specified recorded_as_of. Files disclose assumed
source availability and cannot claim historical recording-time replay. Missing
vintage history, incompatible capabilities and gaps yield unscored folds, not a
final-history prefix fallback. Data-dependent repairs are refused for backtesting
until they can be reconstructed separately at each historical origin; simple input
reordering is safe. Provider training cutoffs remain unattested, even when all
observation inputs are replay-correct.

Reports preserve requested, planned, attempted, failed and matched counts. MAE,
RMSE and bias are computed on the *same intersection of successfully completed
folds for every provider*, including the baseline. Failed/unfinished folds cannot
quietly disappear from completion accounting. A ranking over a partial cohort is
diagnostic, not an automatically selected production model or action permission.
The baseline wins exact error ties. Fewer available folds remain disclosed even
when every available fold succeeds.

With a ledger, every successful execution and full study is immutable. Study truth
vintages are stored within its content-addressed payload, not silently treated as
online actuals. Retrieval can enforce the study's recording-time
cutoff. Rescoring/repeating a study creates a new ID and does not overwrite results.

Python `session.evaluate` returns the full report. MCP `gnomon_evaluate` returns a
compact summary; call it with only `study_id` to retrieve full evidence (possibly
via a bounded result reference). The latest three study IDs are retained without
a ledger, subject to the shared result LRU and byte capacities; durable retrieval
uses `gnomon_ledger` operation `study`.

## Bounded results and exact retrieval

`session.call` and default MCP use one compact projection. Small payloads are
unchanged. Above the operator byte limit, the exact canonical JSON is retained in
a private temporary session directory; the response contains `result_ref`, a small
scalar summary, `partial: true`, and a `gnomon_read` call. It does not show a partial
forecast as if it were complete. No new inference occurs during retrieval.

`gnomon_read(result_ref, pointer="", offset=0, max_chars=4096)` returns `text`,
`next_offset`, `total_chars` and a SHA-256 of the original canonical result.
Concatenate pages until `next_offset` is null, then parse JSON. Offsets count Unicode
codepoints in the selected JSON text, not UTF-8 bytes. Optional JSON pointers
select exact values (`/result/point/0`), with standard `~0`/`~1` escaping. A reference
is not a filesystem path; cross-session, expired or corrupted receipts are refused.

Operator TOML may set `[result_limits]`: `max_response_bytes=8192`,
`max_result_bytes=16777216`, `max_retained_bytes=67108864`, `max_results=16`.
The response limit applies to compact UTF-8 structured payloads; MCP repeats them
in text/structured forms, adding bounded serialization overhead. Discovery schemas,
and explicit full output are not governed by this payload budget.
The limits bound encoded retention, not trusted provider execution or parser peak
memory. Reading a pointer can parse the retained result up to its individual limit.
LRU eviction and session closure remove temporary receipts, not ledger history.

A result too large to retain yields `RESULT_RETENTION_LIMIT` after execution:
`operation_completed` and available execution/study identifiers disclose completed
work. Do not retry blindly; that may repeat inference. No false claim of retained
evidence is made. Durably recorded executions remain in the independently configured
ledger. Error projections can likewise retain full details without flooding context.

Use Python `session.call(..., compact=False)` or full `forecast`/`evaluate` methods
when consuming unabridged data. Short-lived CLI commands deliberately use full JSON
mode so they cannot return expired temporary references; redirect output as needed.
Full mode also returns complete initial evaluation studies, so a CLI without a
durable ledger does not leave only a study ID from a closed session.

Stdio rejects non-object, nonfinite, malformed or oversized requests and drains
oversized physical lines before processing the next request. Its frame limit is
1 MiB including the newline; IDs/methods are capped at256 UTF-8 bytes. Only the
implemented2025-06-18 protocol is advertised during negotiation, consistent with
the [MCP lifecycle specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle).

## Cutoff-bound study routing

```python
recommendation = session.route(
    ref, study_id=study["study_id"], candidates=["preferred"],
    baseline="last_value", horizon=7,
    source_as_of="2025-01-31T00:00:00Z",
    recorded_as_of="2026-09-05T00:00:00Z",
)
```

The source and recorded cutoffs are both required. The input must already be
inspected at compatible cutoffs, with timezone-aware valid times; no implicit UTC
binding or future-containing input is accepted. Parent snapshot bounds cannot
be widened: responses disclose requested and effective cutoffs separately.

Routing reads one explicit original study, never a similarity-weighted mixture
of unrelated leaderboards. It checks the study and each execution existed by
the recorded cutoff, the exact task/series/unit/frequency/horizon/season and
provider cohort agree, and declared revisions/lifecycles/capabilities still match.
Unknown/unversioned providers cannot earn a prior; direct inference remains usable.
Pretrained providers additionally need a returned metadata `training_cutoff` at
or before each historical origin. This is a provider declaration, not independent
attestation of its weights or training data.

Every selected fold's historical inputs must replay from the frozen source at its
original cutoffs. Its outputs are checked against immutable execution records.
Outcomes are realigned to the query's source/recorded vintage. Revised store
outcomes yield a new immutable rescore with original execution references; the
original study and forecasts never change. Plain-file availability remains an
explicit assumption; changed file outcomes with unknown revision timing cannot
be substituted. No new model calls or implicit online actual ingestion occurs.

At least three replayable matched folds are required; `min_folds` may raise that
floor. The candidate with lowest matched MAE is recommended only if it strictly
beats the explicit baseline by `min_improvement` (default0.02). Exact ties retain
the baseline. Missing/incompatible evidence returns an explained baseline fallback.
Recommendations are advisory and never grant action/deployment permission. A
successful route appends a rescore study, retrievable through the ledger; its
recording time is now, never backdated to the queried historical instant.

## Ephemeris

Ephemeris is one optional remote inference connector. Its deployment base URL
is configured by the operator, not determined by the provider name.

```python
import os
from gnomon import InferenceEngine, ForecastRequest, EphemerisProvider

provider = EphemerisProvider(
    os.environ["EPHEMERIS_BASE_URL"],
    token_env="EPHEMERIS_API_TOKEN",  # environment variable name, never a token literal
)
engine = InferenceEngine()
engine.register("ephemeris/route", provider, lifecycle="pretrained")
# Optional discovery makes enabled, healthy explicit models available too:
names = provider.register_models(engine)
run = engine.forecast("ephemeris/route", ForecastRequest((1, 2, 3), 2, quantiles=(.1, .5, .9)))
```

Configuration appends `/forecast` and `/models` to the base URL, preserving a
deployment-specific prefix. No production endpoint is hard-coded. For a custom
authentication header, supply a configured `gnomon.http_transport.JSONTransport`.
HTTPS is required outside loopback unless explicitly opting into unauthenticated
HTTP. Credentials may not be sent over non-loopback HTTP. Redirects are refused,
payload sizes/socket timeouts are bounded, and forecast POSTs are not retried.

The client was implemented against the service source at revision
`f5d3f53b17e56d8c7ed0cbae61e043b8667f236a`, including the deployed Chutes routes.
It requests the median to produce the point forecast and records that definition.
Ephemeris chooses models in `route` mode; Gnomon does not silently run another
model-selection backtest first. Explicit mode requires a model; ensemble mode
supports `vincentize` or `mixture`. Service `models_used`, request ID and notes
are preserved. Neither inspected discovery nor forecasting attests a weights
revision, so these providers remain unversioned and are not cached by default.

Univariate batch requests must share horizon and frequency. Named past/future
covariates are supported, with future keys a subset of past keys. Multivariate
output semantics, sample paths and an explicit fixed seasonal period are not yet
supported by this client and are rejected, not silently dropped. The source
supports additional feedback operations; this client never submits feedback
implicitly or changes the remote router's skill state.

Local HTTP integration tests are not a live-service release check. Deployment
URL and credential configuration still need to be supplied and verified.

## Optional durable ledger

```python
from gnomon import TemporalLedger

ledger = TemporalLedger("gnomon-ledger.db")
engine = InferenceEngine(ledger=ledger)
engine.register("my-model", my_forecast, revision="my-config-v1")
run = engine.forecast("my-model", ForecastRequest(
    history=(1, 2), horizon=1, series_id="shop/sales", unit="USD",
    timestamps=("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
    future_timestamps=("2025-01-03T00:00:00Z",),
))
ledger.append_actual(
    series_id="shop/sales", unit="USD", value=3,
    valid_time="2025-01-03T00:00:00Z",
    source_available_at="2025-01-04T00:00:00Z",
)
score = ledger.evaluate(run.execution_id)
assert score["mae"] == 1
```

Ledger timestamps require an explicit timezone and are normalized to UTC. The
three meanings are distinct:

- `valid_time`: when an observation applies.
- `source_available_at`: when that exact vintage became available at the source.
- `recorded_at`: when this system recorded it, supplied by its clock, not the caller's data.

`actuals_as_of` accepts independent source and recorded cutoffs. `evaluate`
appends a metric-versioned pending/partial/complete result with the exact actual
IDs and matched horizon steps. Corrections never overwrite the original forecast
or prior scores. `compare` requires matched histories, snapshots, covariates,
origins, horizons, timestamps and units, then uses one shared set of actual
vintages. It is a descriptive comparison, not sufficient evidence for routing.
`pending` is a read-only query; records without calendar identity are unscorable.

`record_decision` records explicit policy/inputs/actions and optional authorization
reference; it does not execute or authorize that action. Decision outcome revisions
are appended separately.

The ledger uses a separate SQLite file, foreign keys, transactional inserts,
content hashes and update/delete guards. Small request/result payloads are stored
atomically with execution metadata; large existing report artifacts stay in files.
It detects incompatible schema versions and rejects an unrelated existing database
without modifying it. Independent
processes use separate SQLite connections with serialized writes and a 30-second
lock timeout; this is local-file coordination, not distributed database support.
It is not a tamper-proof audit system against someone able to edit the database
schema or replace the file. Back up the file using SQLite's backup facilities.

The existing `TemporalStore.snapshot(..., as_of=..., recorded_as_of=...)` supports
the same two replay questions. Every stored observation records when this runtime
ingested it. Plain CSV snapshots disclose that historical source/recording times
are unknown rather than inventing them.
