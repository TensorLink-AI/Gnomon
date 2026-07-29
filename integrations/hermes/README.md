# Aion plugin for Hermes Agent

Gives any Hermes agent evidence-backed local forecasting: `aion_capabilities`,
`aion_inspect`, and `aion_forecast` tools wrapping the Aion CLI, plus the
`aion:forecasting` skill encoding the safe-use workflow.

Hermes's policy is that third-party integrations ship standalone rather than
in the hermes-agent tree, so this plugin lives here, next to the runtime it
wraps.

## Prerequisites

- Hermes Agent installed and configured.
- The Aion CLI installed and on PATH (`uv tool install aion-forecast`, or
  `bash install.sh` from a checkout). Verify with `aion capabilities`.

## Install

```bash
cp -r integrations/hermes ~/.hermes/plugins/aion
hermes plugins enable aion
```

Hermes plugins are opt-in; enabling writes `plugins.enabled` in
`~/.hermes/config.yaml`. The plugin's `check_fn` runs a version handshake
(`aion capabilities`) — if the CLI is missing or speaks an incompatible
schema version, the tools stay hidden instead of failing mid-conversation.

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
