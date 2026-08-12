# Quickstart: install to first grounded answer

## 1. Serve the tools (one command)

From a checkout (until the PyPI release lands, this is the way):

```bash
git clone https://github.com/TensorLink-AI/Gnomon && cd Gnomon
uvx --from . gnomon mcp serve
```

Or `pip install -e . && gnomon mcp serve`. Once `gnomon-forecast` is published
to PyPI this becomes `uvx --from gnomon-forecast gnomon mcp serve` with no
clone.

## 2. Connect a client

**Claude Code** — one command from inside the checkout:

```bash
claude mcp add gnomon -- uvx --from "$(pwd)" gnomon mcp serve
```

**Claude Desktop** (`claude_desktop_config.json`) or any `.mcp.json`:

```json
{
  "mcpServers": {
    "gnomon": {
      "command": "uvx",
      "args": ["--from", "/absolute/path/to/Gnomon", "gnomon", "mcp", "serve"]
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`) and any generic stdio MCP client: same
`command`/`args` shape.

## 3. First grounded answer

A messy example dataset ships in the repo: `examples/messy_requests.csv`
(70 days of request counts with weekly seasonality, one transient outlier,
and a genuine regime shift). Ask your agent, or run directly:

```bash
gnomon investigate examples/messy_requests.csv --time timestamp --target requests
gnomon forecast    examples/messy_requests.csv --time timestamp --target requests --horizon 14
gnomon monitor     examples/messy_requests.csv --time timestamp --target requests \
                 --horizon 14 --threshold 340 --alert-cost 1 --miss-cost 20
```

Every response carries an `artifact_path` (immutable directory with
`artifact.json`, `lineage.json`, and for forecasts `forecast.csv` +
`summary.md`) and a `support_assessment`. Numbers live in artifacts —
agents quote them, never restate them.

## 4. Vintages (optional, compounding)

`messy_requests_revisions.csv` is the same series as above plus a
`published` column: the full 70 days of history, with the last ten days
carrying a preliminary figure published same-day and a revised figure
published three days later. One ingest is all it needs.

```bash
gnomon ingest examples/messy_requests_revisions.csv --dataset requests \
     --time timestamp --target requests --known-at published
gnomon store list
gnomon forecast store:requests --time timestamp --target requests --horizon 7 \
     --as-of 2026-06-03
```

Re-supplied corrected files append revision rows; `--as-of` replays any
historical instant using only what was known then, and the artifact's
`snapshot_access` evidence proves it — the run above reports
`max_known_time: 2026-06-03T00:00:00`, so the revisions published on the
4th through the 12th were invisible to it.

Do **not** also ingest `messy_requests.csv` into this dataset. It has no
`published` column, so every row would be recorded as known the day it
applies — asserting that each day's *final* figure was available before
the revision that produced it. `gnomon ingest` says so on any file without
`--known-at`, via the `known_time_assumed` warning, and the resulting
dataset reports `known_time_provenance: partially_assumed`.

## Profiles

| Profile | Tools | Intended session |
| --- | ---: | --- |
| `core` | 7 | capabilities, inspect, forecast, investigate, detect, artifact read/explain |
| `decision` | 12 | `core` plus decide, monitor, route, status, and outcome resolution |
| `data` | 10 | `core` plus ingest, dataset listing, and actuals scoring |
| `full` | 18 | Every stable tool, including context/covariate validation and TSFM installation |

`full` is currently the default. Start a narrower surface with
`gnomon mcp serve --profile core|decision|data|full`; `gnomon_capabilities`
reports the active profile under `mcp_profile`. A future default-profile
change is an evaluation decision, not a claim about this build. The matched
[surface experiment](design/mcp-surface-experiment.md) records schema bytes,
cumulative tokens, call counts, yield, and accuracy before that choice is made.

## Where artifacts land

Omit `output_dir`: artifacts go to the server's default, reported by
`gnomon_capabilities` under `workspace.default_output_dir` (the resolved
`./gnomon-output` next to the server's working directory). Sandboxed
hosts should start the server with its working directory inside the
writable jail so the disclosed default is the allowed path.

## Graduated support

`gnomon_forecast` answers by default at the highest tier the evidence
achieves (`minimum_support: "best_effort"`): a fully evaluated forecast,
an evaluated prefix plus a labelled naive remainder (`horizon_split`),
or a naive extrapolation alone. Every row carries a `tier`
(`supported` / `conditionally_supported` / `best_effort`), the response
`headline` is one deterministic sentence naming the weakest tier
present — safe to relay verbatim — and the verifier rejects any claim
quoting a sub-supported value without its tier. Pass
`minimum_support: "supported"` for the strict refusal with typed
recovery; a series where nothing is computable still abstains.

## Response sizes

Tool responses are budgeted (`RESPONSE_BUDGET_BYTES`, reported by
`gnomon capabilities` under `forecast_surface.response_budget`).
`gnomon_forecast` answers in `brief` format by default — the q50 path
with one q10–q90 interval per step plus the complete support assessment;
pass `format: "full"` for every quantile level inline. Any response over
the budget trims its long arrays to first/last entries, sets
`truncated: true`, and points at the artifact, which always carries the
complete data. Support assessments, warnings, assumptions, and
error/repair payloads are never trimmed.

Every non-error verb response also carries a compact routing projection where
the underlying result makes it applicable: `artifact_id`, `tier_floor`, typed
`limitation_groups`, and aggregated `recovery_actions`. Repeated warning text
is grouped with its affected-series count and up to three examples; the full
warning remains attached to every series in the immutable artifact. These
fields, the headline, support, assumptions, staleness, and artifact references
are protected from trimming.

When schema ambiguity blocks a forecast, each repair option contains a literal
`tool_call` with the complete argument object—one per candidate plus the
batched `target_column: "auto"` form. A host can issue it directly rather than
spending another model turn composing selector syntax.

## Reusing data without resending it

Every data-reading response includes an opaque `data_ref`. It is scoped to the
running MCP process and binds the source, resolved columns, temporal cutoff,
frequency, and repair policy. Pass that reference to another verb instead of
repeating `input` or `observations`:

```json
{"data_ref": "data_…", "horizon": 14}
```

Unknown references fail with a `resupply_data` recovery action, and a call
cannot silently override the schema or temporal view bound to a reference.
Inline observations are capped at 500 rows because the full payload remains in
conversation history; send them once, then reuse `data_ref`. References expire
when the MCP server process exits and are not portable across hosts.

## Tool surface

Primary macros: `gnomon_forecast`, `gnomon_investigate_change`,
`gnomon_detect_anomalies`, `gnomon_decide`, and `gnomon_monitor`. The
data-reading tools infer the way the CLI does:
`time_column`, `target_column`, and (for forecasts) `horizon` may be
omitted, are filled only when the file leaves no choice, and every
inference is disclosed in the result's `support_assessment.assumptions` —
`{"input": "data.csv"}` is a complete first `gnomon_forecast` call.
Publishing verbs fail loudly on target ambiguity with candidate columns and
machine-readable repair options. Read-only `gnomon_inspect` instead examines
every qualifying numeric column and discloses that assumption; it also accepts
a comma list or `auto`. `store:<dataset>` inputs still need explicit columns.
Support: `gnomon_capabilities`, `gnomon_inspect`,
`gnomon_get_artifact`, `gnomon_explain_run`, `gnomon_preflight_context`
(dry-run admission for proposed context events, with the accepted span
grammar in the response), `gnomon_validate_covariates` (its description
carries the point-in-time format contract; `gnomon_forecast` takes every
covariate argument directly), and the tracking lifecycle
(`gnomon_submit_actuals` to score, `gnomon_status` to read — its
`section` parameter returns the open-forecast, performance, or decision
slice on its own). Foundation models install from the surface too:
`gnomon_install_tsfm` starts a detached sandbox install for any name in
`gnomon_capabilities` under `models.tsfm_available` and reports state
(absent / installing / ready / failed) on each call — no shell needed.

Calendar-shaped data needs no upstream preprocessing: pass
`regrid: "business_daily"` for Mon-Fri market data (weekends and
holidays are forward-filled onto the continuous daily grid) or
`regrid: "month_start"` for month-end-stamped monthly feeds — both
disclosed as warnings, neither charged against the repair ceiling.

## Migrating from v0.2

Every v0.2 tool name, schema, and the `gnomon_forecast` contract are
preserved (see `COMPATIBILITY.md`), but the deprecated decision pair
(`gnomon_record_decision` / `gnomon_resolve_decision`) is no longer on
the default surface — tools compete for model attention, and these two
argued for their own replacement in every session. Start the server with
`GNOMON_V02_COMPAT=1` to restore them, schemas and behaviour unchanged;
`gnomon capabilities` reports the state under `compat.v02_tools`.
New in this release:
the three additional macros, `store:<dataset>` inputs, `as_of` replay,
`support_assessment` on every result, `lineage.json` in artifacts, and
machine-readable `repair_options` on every structured error.
