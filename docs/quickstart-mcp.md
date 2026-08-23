# Quickstart: install to first grounded answer

## 1. Serve the tools (one command)

From PyPI:

```bash
uvx --from gnomon-forecast gnomon mcp serve
```

For repository development, use `uvx --from . gnomon mcp serve` from a checkout.

## 2. Connect a client

**Claude Code** — one command:

```bash
claude mcp add gnomon -- uvx --from gnomon-forecast gnomon mcp serve
```

**Claude Desktop** (`claude_desktop_config.json`) or any `.mcp.json`:

```json
{
  "mcpServers": {
    "gnomon": {
      "command": "uvx",
      "args": ["--from", "gnomon-forecast", "gnomon", "mcp", "serve"]
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`) and any generic stdio MCP client: same
`command`/`args` shape.

**Hermes Agent** — install the packaged Gnomon skill and the CLI, then expose
the same stdio server through Hermes's MCP configuration:

```bash
uv tool install gnomon-forecast
gnomon mcp serve
```

The distribution includes `share/gnomon/skills/use-gnomon`; copy or link that
directory into the host's configured skills directory. The skill teaches the
host to call forecast directly, preserve tiers and artifact identities, and
avoid redundant artifact fetches. Host-specific plugin discovery paths change
independently of Gnomon, so verify the path against the installed Hermes
version rather than letting an installer guess it.

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

Every response carries an `artifact_path` (integrity-sealed directory with
`artifact.json`, `lineage.json`, and for forecasts `forecast.csv` +
`summary.md` + offline `report.html`) and a `support_assessment`. Numbers live in artifacts —
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
| `core` | 6 | capabilities, inspect, forecast, investigate, detect, explain |
| `describe` | 7 | Experimental `core` plus fast descriptive temporal evidence |
| `evidence` | 2 | **Default:** fast description plus evaluated forecast |
| `mega` | 3 | Experimental inspect/run/track consolidation arm |
| `decision` | 11 | `core` plus decide, monitor, route, status, and outcome resolution |
| `data` | 9 | `core` plus ingest, dataset listing, and actuals scoring |
| `full` | 18 | Every stable tool, including context/covariate validation and TSFM installation |

`evidence` is the default. Select a broader surface explicitly with
`gnomon mcp serve --profile core|describe|evidence|mega|decision|data|full`; `gnomon_capabilities`
reports the active profile under `mcp_profile`. A future default-profile
The earlier [surface experiment](design/mcp-surface-experiment-results.md)
retained `full`; it is superseded by the fresh workflow experiment after the
response-contract and routing fixes. `full` remains available, but its measured
53.6K tokens per case makes it unsuitable as the ambient agent surface.

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

A normal forecast starts with `gnomon_forecast` itself. It performs safe
schema inference and validation and returns a quotable preview plus canonical
`temporal_facts` (`seasonal_period_steps`, its human label, frequency, and
computed source). Do not spend calls on `gnomon_capabilities`,
`gnomon_inspect`, or `gnomon_get_artifact` first unless the user explicitly
asked for feature discovery, the schema is genuinely ambiguous, or deeper
artifact evidence is required.

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

Every data-bearing response also reports `series_end` and
`wall_clock_now`; when the gap exceeds one grid step it includes a protected
`staleness` sentence. Wide forecast responses return the three most notable
series (threshold crossing first, then relative path movement), summarize the
remainder by support tier, and point to artifact selectors for the full panel.
The integrity-sealed artifact still contains every series and every row.

`gnomon_describe` also returns deterministic multi-series triage: the largest
absolute final-step change, the named ranking rule, top entries, and a
`remainder_preserved` fact. These typed fields are safer to quote than asking
the host model to reconstruct a ranking from prose.

Typed temporal answers distinguish the behavior of the immutable published
forecast from claims about the future process. In particular,
`forecast_path_behavior` is only a deterministic description of the point
path; calibrated or explicitly weak process evidence lives in `process_claim`.
The top-level `support` object states whether automation is eligible. When
context supports a different scenario, `conditional_answer` preserves its
provenance and keeps `primary_forecast_unchanged: true`; it is never a silent
replacement for the governed forecast.

Every non-error verb response also carries a compact routing projection where
the underlying result makes it applicable: `artifact_id`, `tier_floor`, typed
`limitation_groups`, and aggregated `recovery_actions`. Repeated warning text
is grouped with its affected-series count and up to three examples; the full
warning remains attached to every series in the integrity-sealed artifact. These
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

## Reusing validated context

Context-aware calls similarly return a persistent `context_ref`. Supply it on
later forecasts instead of repeating `context_events` or
`context_events_file`:

```json
{"data_ref": "data_…", "context_ref": "context_…", "horizon": 14}
```

`context_ref` is content-addressed and project-scoped. Configure its location
with `GNOMON_CONTEXT_STORE` and isolate tenants or projects with
`GNOMON_CONTEXT_NAMESPACE`. Every replay verifies the immutable receipt and
reapplies `known_at`, `as_of`, and numeric admission; cached context is never a
permission to rewrite the canonical primary forecast. Supplying a reference
and raw context together fails loudly.

The CLI context workflow stores validated compiler output and prints its
reference:

```bash
gnomon context validate --response response.json --file operations.md \
  --context-store .gnomon/context-store --context-namespace production
```

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

## Migrating older integrations

The deprecated v0.2 compatibility tools have been removed; setting
`GNOMON_V02_COMPAT` has no effect. Migrate agent integrations to the current
MCP registry: use `gnomon_status` for due forecasts and performance,
`gnomon_resolve_outcome` for decision outcomes, and pass covariates through
`gnomon_forecast` after validating them with `gnomon_validate_covariates`.
The original `gnomon_forecast` input schema remains the one frozen exception.
See [`COMPATIBILITY.md`](../COMPATIBILITY.md) for the complete mapping.
# Shadow adapter outcomes

The experimental `gnomon_track` verb also accepts
`record_adapter_shadow` for a paired realised challenger/baseline error and
`assess_adapter_shadow` for an outcome-backed recommendation. Assessment is
advisory only: it never changes the publishing candidate, and an unpinned
adapter cannot graduate.
