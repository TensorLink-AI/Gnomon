# Gnomon plugin for Hermes Agent

Gives any Hermes agent evidence-backed local forecasting: `gnomon_capabilities`,
`gnomon_inspect`, `gnomon_forecast`, and `gnomon_propose_context_events`, plus tools
to submit actuals, discover due forecasts, inspect realised performance, and
record/resolve agent decisions. The `gnomon:forecasting` skill encodes the
safe-use workflow.

Covariate enrichment adds `gnomon_covariate_guide`,
`gnomon_validate_covariates`, and `gnomon_propose_covariates`. Hermes decides what
external data may matter and fetches it with its own permitted tools. Gnomon only
accepts a local point-in-time CSV and admits features that demonstrate stable
backtest lift.

`gnomon_propose_context_events` is the "LLM flair with honest numbers" path:
Gnomon emits the extraction prompt (`gnomon context prompt`), the plugin runs it
on the **host's own model** via Hermes's `ctx.llm` facade (no Gnomon-side API
key), and Gnomon deterministically validates the response (`gnomon context
validate`). Validated events are still only proposals — `gnomon_forecast`
admits them into the numbers solely when they demonstrate stable improvement
on identical backtest folds.

Hermes's policy is that third-party integrations ship standalone rather than
in the hermes-agent tree, so this plugin lives here, next to the runtime it
wraps.

## Prerequisites

- Hermes Agent installed and configured.
- The Gnomon CLI installed and on PATH (`uv tool install gnomon-forecast`, or
  `bash install.sh` from a checkout). Verify with `gnomon capabilities`.

## Install

The repository root carries a `plugin.yaml` and `__init__.py` that re-export
this directory, so the standard install flow works directly:

```bash
hermes plugins install https://github.com/TensorLink-AI/Gnomon.git
hermes plugins enable gnomon
```

From a local checkout, copying this directory works equally well:

```bash
cp -r integrations/hermes ~/.hermes/plugins/gnomon
hermes plugins enable gnomon
```

Hermes plugins are opt-in; enabling writes `plugins.enabled` in
`~/.hermes/config.yaml`. The plugin's `check_fn` runs a version handshake
(`gnomon capabilities`) — if the CLI is missing or speaks an incompatible
schema version, the tools stay hidden instead of failing mid-conversation.

The bundled skill is registered as `gnomon:forecasting`. Plugin skills are
explicit loads in Hermes (they are not listed in the system prompt's skill
index), so the tool descriptions point the agent at it. To put the skill in
the auto-surfaced index instead, install it from this repo's tap-compatible
copy:

```bash
hermes skills install TensorLink-AI/Gnomon/skills/forecasting
```

## Configuration

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `GNOMON_PLUGIN_CLI` | Command used to invoke Gnomon (shell-quoted, e.g. `"/opt/venv/bin/python -m gnomon"`) | `gnomon` |
| `GNOMON_PLUGIN_TIMEOUT` | Per-call timeout in seconds | `300` |

## Behaviour

- Success payloads and Gnomon's structured errors are passed through
  **verbatim** — this layer never touches a number.
- Transport failures (CLI missing, timeout, non-JSON output) come back in
  the same error envelope with codes `GNOMON_NOT_INSTALLED`, `GNOMON_TIMEOUT`,
  and `GNOMON_PROTOCOL_ERROR`.
- Forecast values are read from the immutable artifact directory
  (`forecast.csv`, `summary.md`, `artifact.json`) that `gnomon_forecast`
  returns, keeping the numerical source of truth on disk.

## Try it

Ask Hermes:

> Inspect examples/daily_requests.csv (timestamp column `timestamp`, target
> `requests`, daily) and, if it is supported, forecast the next 7 days.

A recurring forecast is one Hermes cron job away:

```bash
hermes cron create "0 8 * * 1" \
  "Forecast the next 7 days of requests from ~/metrics/daily_requests.csv using Gnomon and summarise support and warnings" \
  --name weekly-forecast --deliver telegram
```

For a feedback loop, include a `project` when forecasting, record the action
with `gnomon_record_decision`, periodically check `gnomon_list_open_forecasts`,
then submit complete actuals and resolve the decision outcome. See
[`docs/agent-evaluation.md`](../../docs/agent-evaluation.md) for the full flow.
