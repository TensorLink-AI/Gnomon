---
name: setup-gnomon-ephemeris
description: Install Gnomon, configure its Ephemeris forecasting connector, connect an agent over MCP, and verify provider discovery or troubleshoot setup. Use when an agent needs to set up Ephemeris with Gnomon; for Hermes secure API-key entry use connect-ephemeris.
---

# Set up Gnomon with Ephemeris

This skill works with any agent that can run shell commands or connect to an MCP
stdio server. Configure the connector in Gnomon's startup TOML; credentials and
service URLs are not forecast tool arguments.

## API references

- [Ephemeris API documentation](https://ephemeris.cascade.industries/docs)
- [Ephemeris OpenAPI JSON](https://ephemeris.cascade.industries/openapi-m1.json)
  — the download URL advertised by the documentation. On 2026-09-30 this URL
  returned HTTP 404; if unavailable, use the docs page and check its current
  OpenAPI link. Do not treat an unavailable specification as a working schema.
- [Gnomon connector and configuration](https://github.com/TensorLink-AI/Gnomon/blob/main/docs/production/INFERENCE.md)

Use Ephemeris docs for the remote API contract and Gnomon's installed schemas for
CLI/MCP inputs. They are different: Gnomon accepts `history` and `frequency` and
translates them to gateway `series[].values` and `series[].freq`.

## Install and configure

Use Python 3.11 or newer in the environment the agent host will execute. From a
Gnomon checkout, install that checkout, especially when testing connector changes:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

Without a checkout, install `gnomon-forecast` instead of `.`. Inspect the installed
configuration schema with `gnomon capabilities --config-schema`; older releases
may lack `api_format`. Use a release or checkout supporting the configuration below.

Set the gateway base URL in the process environment:

```bash
export EPHEMERIS_BASE_URL='https://ephemeris.cascade.industries/api/v1'
```

Supply `EPHEMERIS_API_TOKEN` through the host's secret configuration or an existing
environment variable. A gateway key can be created in the
[Ephemeris dashboard](https://ephemeris.cascade.industries/dashboard).
Do not write real tokens into the skill, TOML, committed host settings, or output.
Gnomon does not automatically load `.env` files. The MCP server process must
inherit both variables; an export in a separate terminal may not reach a GUI host.

Create `providers.toml` in the user's chosen project/configuration directory:

```toml
schema_version = 1

[providers.ephemeris]
kind = "ephemeris"
base_url_env = "EPHEMERIS_BASE_URL"
token_env = "EPHEMERIS_API_TOKEN"
api_format = "gateway"
mode = "route"
discover = true
```

Merge this provider table into an existing configuration rather than overwriting
other providers or ledger settings. Pass the configuration path explicitly;
Gnomon does not search the working directory for it.

This configuration registers the routing provider as `ephemeris`. Startup
discovery calls `GET /models` and registers enabled, healthy models as
`ephemeris/<returned model name>`. Discover actual names rather than guessing a
catalog. Respect a user-selected model; `route` delegates model choice to Ephemeris.

For a different deployment, use its supplied base URL and matching credentials.
Gnomon appends `/models` and `/forecast` to that URL. A direct inference service
uses `api_format = "direct"`; `auto` selects gateway format only when the URL path
ends in `/api/v1`. Set the format explicitly for custom proxy paths.

## Verify discovery and connect the agent

Run discovery before inference; it does not submit a forecast:

```bash
gnomon capabilities --providers-config /absolute/path/providers.toml
```

Check that `ephemeris` and the expected discovered model names are present. The
gateway does not advertise `/health`; use model discovery to check connectivity.

Configure the agent host to launch this MCP stdio process, adapting the surrounding
configuration structure to that host:

```json
{
  "command": "/absolute/path/.venv/bin/gnomon",
  "args": ["mcp", "serve", "--providers-config", "/absolute/path/providers.toml"]
}
```

Replace both paths with real absolute paths. Restart the host's server connection
after changing startup configuration or environment. Read MCP `tools/list`, then
call `gnomon_capabilities` with `{}` to confirm the connected process sees the
provider. The skill supplies instructions; it does not start or configure MCP by itself.

## Run a requested forecast

Remote inference may incur charges. Run this smoke forecast only when remote
inference is within the user's authorized task; setup alone can stop at discovery.

```bash
gnomon infer --providers-config /absolute/path/providers.toml \
  --provider ephemeris \
  --request '{"history":[10,12,11,13,12,14,13,15],"horizon":2,"frequency":"D","quantiles":[0.1,0.5,0.9]}'
```

Equivalent MCP tool call:

```json
{
  "name": "gnomon_forecast",
  "arguments": {
    "provider": "ephemeris",
    "request": {
      "history": [10,12,11,13,12,14,13,15],
      "horizon": 2,
      "frequency": "D",
      "quantiles": [0.1,0.5,0.9]
    }
  }
}
```

Use a discovered `ephemeris/<model name>` provider for an explicit model. Horizon
counts grid steps; the example's two steps are two days. Real data must meet the
chosen model's history and input requirements. Preserve execution IDs, service
`models_used`, request IDs and notes. Gnomon's point forecast is the median;
quantiles do not establish calibration and discovery does not attest model weights.
When a response is partial, retrieve its `result_ref` with `gnomon_read` rather
than rerunning inference.

## Next: related skills

Load the skill matching the user's task instead of extending this one. Skills in
the same Gnomon install are linked locally; the others are in the Gnomon repository
(install a linked skill's whole directory, including scripts, before running it).

- [Use Gnomon](../use-gnomon/SKILL.md): MCP forecasting, backtests and results.
- [Use the ledger](../use-gnomon-ledger/SKILL.md): matched-history comparison and
  decision review.
- [Trade with Gnomon](../trade-with-gnomon/SKILL.md): forecast-led trade decisions.
- [Forecast with Gnomon](../forecast-with-gnomon/SKILL.md):
  Python API; use `GnomonSession.from_config("/absolute/path/providers.toml")`.
- [Forecast report](https://github.com/TensorLink-AI/Gnomon/blob/main/skills/forecast-report/SKILL.md)
  and [monitor token usage](https://github.com/TensorLink-AI/Gnomon/blob/main/skills/monitor-token-usage/SKILL.md):
  Hermes-oriented charted reports and LLM usage/cost forecasting (not Ephemeris charges).

This setup registers `ephemeris` (routing) and discovered `ephemeris/<model>`
providers, not an `ephemeris/ensemble` alias. Ephemeris forecasts can be recorded
and reviewed in the ledger, but `compare_history` cannot rank them yet: the
connector reports no attested model revision or training cutoff. Loading another
skill does not authorize further billable inference, scheduled jobs or delivery.

## Troubleshoot

- Missing provider: confirm the connected process uses the intended TOML and
  Python environment; check its capabilities after restarting.
- Missing URL/token or authentication failure: check variable presence without
  printing secrets, and match gateway credentials to the gateway endpoint.
- HTTP 404 or payload validation failure: verify the base URL prefix and
  `api_format`; do not append `/forecast` to the configured base URL.
- Missing explicit model: discovery registers only models marked enabled and
  healthy. Report its absence rather than silently choosing another model.
- Unsupported input: use `frequency` for a frequency hint. The connector does
  not support fixed seasonal-period control, sample paths or multivariate output.
  Future covariate names must have matching past covariate histories.
- Timeout or failed forecast: Gnomon does not automatically retry forecast POSTs.
  Check any returned receipt before a retry that could duplicate billable work.

Report what was configured, whether discovery succeeded, and whether a forecast
was actually run. Do not describe discovery alone as a successful forecast.
