# Aion plugin for Hermes Agent

Gives any Hermes agent evidence-backed local forecasting: `aion_capabilities`,
`aion_inspect`, `aion_forecast`, and `aion_propose_context_events`, plus tools
to submit actuals, discover due forecasts, inspect realised performance, and
record/resolve agent decisions. The `aion:forecasting` skill encodes the
safe-use workflow.

Covariate enrichment adds `aion_covariate_guide`,
`aion_validate_covariates`, and `aion_propose_covariates`. Hermes decides what
external data may matter and fetches it with its own permitted tools. Aion only
accepts a local point-in-time CSV and admits features that demonstrate stable
backtest lift.

`aion_propose_context_events` is the "LLM flair with honest numbers" path:
Aion emits the extraction prompt (`aion context prompt`), the plugin runs it
on the **host's own model** via Hermes's `ctx.llm` facade (no Aion-side API
key), and Aion deterministically validates the response (`aion context
validate`). Validated events are still only proposals — `aion_forecast`
admits them into the numbers solely when they demonstrate stable improvement
on identical backtest folds.

Hermes's policy is that third-party integrations ship standalone rather than
in the hermes-agent tree, so this plugin lives here, next to the runtime it
wraps.

## Prerequisites

- Hermes Agent installed and configured.
- The Aion CLI installed and on PATH (`uv tool install aion-forecast`, or
  `bash install.sh` from a checkout). Verify with `aion capabilities`.

## Install

The repository root carries a `plugin.yaml` and `__init__.py` that re-export
this directory, so the standard install flow works directly:

```bash
hermes plugins install https://github.com/TensorLink-AI/Aion.git
hermes plugins enable aion
```

From a local checkout, copying this directory works equally well:

```bash
cp -r integrations/hermes ~/.hermes/plugins/aion
hermes plugins enable aion
```

Hermes plugins are opt-in; enabling writes `plugins.enabled` in
`~/.hermes/config.yaml`. The plugin's `check_fn` runs a version handshake
(`aion capabilities`) — if the CLI is missing or speaks an incompatible
schema version, the tools stay hidden instead of failing mid-conversation.

The bundled skill is registered as `aion:forecasting`. Plugin skills are
explicit loads in Hermes (they are not listed in the system prompt's skill
index), so the tool descriptions point the agent at it. To put the skill in
the auto-surfaced index instead, install it from this repo's tap-compatible
copy:

```bash
hermes skills install TensorLink-AI/Aion/skills/forecasting
```

## Configuration

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `AION_PLUGIN_CLI` | Command used to invoke Aion (shell-quoted, e.g. `"/opt/venv/bin/python -m aion"`) | `aion` |
| `AION_PLUGIN_TIMEOUT` | Per-call timeout in seconds | `300` |

## Behaviour

- Success payloads and Aion's structured errors are passed through
  **verbatim** — this layer never touches a number.
- Transport failures (CLI missing, timeout, non-JSON output) come back in
  the same error envelope with codes `AION_NOT_INSTALLED`, `AION_TIMEOUT`,
  and `AION_PROTOCOL_ERROR`.
- Forecast values are read from the immutable artifact directory
  (`forecast.csv`, `summary.md`, `artifact.json`) that `aion_forecast`
  returns, keeping the numerical source of truth on disk.

## Try it

Ask Hermes:

> Inspect examples/daily_requests.csv (timestamp column `timestamp`, target
> `requests`, daily) and, if it is supported, forecast the next 7 days.

A recurring forecast is one Hermes cron job away:

```bash
hermes cron create "0 8 * * 1" \
  "Forecast the next 7 days of requests from ~/metrics/daily_requests.csv using Aion and summarise support and warnings" \
  --name weekly-forecast --deliver telegram
```

For a feedback loop, include a `project` when forecasting, record the action
with `aion_record_decision`, periodically check `aion_list_open_forecasts`,
then submit complete actuals and resolve the decision outcome. See
[`docs/agent-evaluation.md`](../../docs/agent-evaluation.md) for the full flow.
