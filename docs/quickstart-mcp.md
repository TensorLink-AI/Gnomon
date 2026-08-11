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

`gnomon mcp serve --profile core` exposes only the analytical verbs plus
inspection and artifact reads; `decision` adds the decide/monitor/route/
status/resolve_outcome lifecycle; `data` adds the bitemporal store and
actuals scoring; `full` (default) is everything. `gnomon capabilities`
reports the active profile under `mcp_profile`.

## Where artifacts land

Omit `output_dir`: artifacts go to the server's default, reported by
`gnomon_capabilities` under `workspace.default_output_dir` (the resolved
`./gnomon-output` next to the server's working directory). Sandboxed
hosts should start the server with its working directory inside the
writable jail so the disclosed default is the allowed path.

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

## Tool surface

Primary macros: `gnomon_forecast`, `gnomon_investigate_change`, `gnomon_decide`,
`gnomon_monitor`. The data-reading tools infer the way the CLI does:
`time_column`, `target_column`, and (for forecasts) `horizon` may be
omitted, are filled only when the file leaves no choice, and every
inference is disclosed in the result's `support_assessment.assumptions` —
`{"input": "data.csv"}` is a complete first `gnomon_forecast` call.
Ambiguity fails loudly with the candidate columns and machine-readable
repair options; `store:<dataset>` inputs still need the explicit columns. Support: `gnomon_capabilities`, `gnomon_inspect`,
`gnomon_get_artifact`, `gnomon_explain_run`, `gnomon_preflight_context`
(dry-run admission for proposed context events, with the accepted span
grammar in the response), `gnomon_validate_covariates` (its description
carries the point-in-time format contract; `gnomon_forecast` takes every
covariate argument directly), and the tracking lifecycle
(`gnomon_submit_actuals` to score, `gnomon_status` to read — its
`section` parameter returns the open-forecast, performance, or decision
slice on its own).

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
