# Quickstart: install to first grounded answer

## 1. Serve the tools (one command)

From a checkout (until the PyPI release lands, this is the way):

```bash
git clone https://github.com/TensorLink-AI/Aion && cd Aion
uvx --from . aion mcp serve
```

Or `pip install -e . && aion mcp serve`. Once `aion-forecast` is published
to PyPI this becomes `uvx --from aion-forecast aion mcp serve` with no
clone.

## 2. Connect a client

**Claude Code** — one command from inside the checkout:

```bash
claude mcp add aion -- uvx --from "$(pwd)" aion mcp serve
```

**Claude Desktop** (`claude_desktop_config.json`) or any `.mcp.json`:

```json
{
  "mcpServers": {
    "aion": {
      "command": "uvx",
      "args": ["--from", "/absolute/path/to/Aion", "aion", "mcp", "serve"]
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
aion investigate examples/messy_requests.csv --time timestamp --target requests
aion forecast    examples/messy_requests.csv --time timestamp --target requests --horizon 14
aion monitor     examples/messy_requests.csv --time timestamp --target requests \
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
aion ingest examples/messy_requests_revisions.csv --dataset requests \
     --time timestamp --target requests --known-at published
aion store list
aion forecast store:requests --time timestamp --target requests --horizon 7 \
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
the revision that produced it. `aion ingest` says so on any file without
`--known-at`, via the `known_time_assumed` warning, and the resulting
dataset reports `known_time_provenance: partially_assumed`.

## Tool surface

Primary macros: `aion_forecast`, `aion_investigate_change`, `aion_decide`,
`aion_monitor`. Support: `aion_capabilities`, `aion_inspect`,
`aion_get_artifact`, `aion_explain_run`, covariate tools, and the tracking
lifecycle (`aion_submit_actuals`, `aion_list_open_forecasts`,
`aion_model_performance`, decision record/resolve).

## Migrating from v0.2

Nothing breaks: every v0.2 tool name, schema, and the `aion_forecast`
contract are preserved (see `COMPATIBILITY.md`). New in this release:
the three additional macros, `store:<dataset>` inputs, `as_of` replay,
`support_assessment` on every result, `lineage.json` in artifacts, and
machine-readable `repair_options` on every structured error.
