# Bring your own forecasting software

This is a separately installable **example**, not a new Gnomon built-in adapter.
It demonstrates a stateless callable and a fresh-per-request fitting object. Both
use the same `ForecastRequest -> ForecastResult` boundary through Python, CLI and
MCP. Its two reference calculations are deliberately elementary; they do not
execute StatsForecast, NeuralForecast or Darts and establish no forecasting lift.

## Run from a checkout

Use the current checkout or a reviewed wheel containing these APIs; an older
package-index release need not contain the unreleased execution-default changes.
From the repository root, create an isolated environment and a new output location:

```bash
GNOMON_DEMO_ROOT=$(mktemp -d)
python3 -m venv "$GNOMON_DEMO_ROOT/venv"
GNOMON_DEMO_PY="$GNOMON_DEMO_ROOT/venv/bin/python"
"$GNOMON_DEMO_PY" -m pip install . ./examples/provider_plugin
"$GNOMON_DEMO_PY" examples/provider_plugin/walkthrough.py run \
  --output-dir "$GNOMON_DEMO_ROOT/run"
```

Building from source can fetch build tools. For offline use, build the Gnomon and
example wheels beforehand, then install both with `--no-index --no-deps`. The
example is not included in the core Gnomon wheel and adds no core dependency.

The walkthrough creates only the new output directory and refuses to overwrite an
existing one. It copies `providers.toml` and `request.json` there, records three
unique executions, and verifies the factory is fresh on repeated calls. The saved
last-value forecast remains `[14, 14]` throughout:

| Actuals available for scoring | Status | MAE |
| --- | --- | --- |
| None | pending | null |
| First horizon step | partial | 1 |
| Both steps | complete | 1.5 |
| Correction to the first step | complete | 4 |
| Source replay before the correction | complete | 1.5 |
| Recorded-time replay before the correction | complete | 1.5 |

It records a proposed decision without authorization or execution, backs up the
ledger with SQLite's backup API, restores into another file, and checks that the
original prediction and all score versions survive. `receipt.json` summarizes
the assertions; all data and decisions are explicitly synthetic.

Use the same installed provider package and copied configuration from the CLI:

```bash
"$GNOMON_DEMO_PY" -m gnomon infer --provider example-last \
  --providers-config "$GNOMON_DEMO_ROOT/run/providers.toml" \
  --request "@$GNOMON_DEMO_ROOT/run/request.json"
"$GNOMON_DEMO_PY" -m gnomon ledger \
  --providers-config "$GNOMON_DEMO_ROOT/run/providers.toml" \
  --arguments '{"operation":"pending"}'
```

For an MCP client, use that environment's absolute `bin/gnomon` executable with
arguments `mcp serve --providers-config /absolute/path/to/run/providers.toml`.
The session discovers the providers at startup. This configuration leaves agent
outcome writes disabled: the walkthrough's direct Python ledger writes are
operator actions, not permission granted to an agent. Scoring can still append
evaluations. File/data/result references are session-scoped; execution IDs persist.

## Replace the example implementation

Edit your own package's [provider functions](src/gnomon_example_provider/__init__.py).
Keep model construction, fitting options and library-specific output conversion
there. A factory is appropriate for mutable fitting state; a callable or retained
provider suits stateless/pretrained inference. Do not share one fitted object
across evaluation folds and then label it `fresh_per_request`.
If an instance owns resources, expose `close()` for cleanup. Factory instances are
closed after their invocation; retained providers are closed with the engine/session.

Return finite, aligned results with the request's exact timestamps, series identity
and unit label. Declare only capabilities your function implements, including
covariate shapes and uncertainty. The examples use default point-only capabilities;
Gnomon rejects unsupported requests before dispatch. No automatic resampling,
unit conversion, quantile reconstruction or calibration is implied. Include your
library version, model configuration and weights identity in your revision policy;
a declared revision is not independently attested by Gnomon.

Register the installed module using the `module:attribute` entrypoint in
[providers.toml](providers.toml). Installing arbitrary Python extensions is an
operator trust decision, not an agent tool operation. No per-library Gnomon adapter
or plugin discovery framework is necessary.

See [ledger operating and migration instructions](../../docs/production/OPERATIONS.md).
