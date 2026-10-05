<p align="center">
  <img src="https://raw.githubusercontent.com/TensorLink-AI/Gnomon/main/docs/assets/gnomon-logo.png" alt="Gnomon logo" width="180">
</p>

# Gnomon

<!-- mcp-name: io.github.TensorLink-AI/gnomon -->

**Gnomon is a forecasting toolkit for AI agents, with model comparison and memory
of past results.**

Agents can discover available models, run forecasts, compare them on historical
outcomes, and retrieve relevant past results to inform future model choices.
People and agents share Python, CLI and MCP interfaces with structured evidence
behind the results.

Use your own forecasting software. Add a remote service or a persistent ledger
when you need one.

## Quick start

Python 3.11–3.13. No required third-party dependencies.

```bash
python -m pip install 'gnomon-forecast==1.4.0'
gnomon infer --provider last_value --request '{"history":[10,12,11],"horizon":2}'
```

From a checkout, use `python -m pip install .`; this also works before the
versioned package is published. These docs describe the checkout, which may contain
changes newer than an installed release. Use `gnomon environment` and
`gnomon capabilities` to check the installed build and supported operations.
The forecast command runs an offline baseline.
To use your own local model, install Gnomon in the **same Python environment** as
the model and its dependencies, then run that environment's `gnomon` or
`python -m gnomon`. An isolated Gnomon environment cannot import PyTorch or
another model library installed elsewhere.

## Connect an agent

[Agent reading index](llms.txt) · [Copyable MCP configurations](integrations/mcp/README.md)

Use the [main agent workflow](skills/use-gnomon/SKILL.md) for connected tools,
or [Python forecasting](skills/forecast-with-gnomon/SKILL.md) for scripts.
The workflow is: **inspect → recall relevant evidence → forecast or compare →
explain → review observed outcomes**. Skip steps the task does not need.
A one-off calculation needs neither a ledger nor a model comparison.

Run `gnomon mcp serve` in your agent host. The agent gets 6 tools by default:

- `gnomon_inspect`: check data and freeze a reusable snapshot.
- `gnomon_describe`: calculate an observed statistic.
- `gnomon_capabilities`: find available models and their limits.
- `gnomon_forecast`: run the selected model.
- `gnomon_evaluate`: compare models on past data with an explicit budget.
- `gnomon_read`: retrieve saved results without running the model again.

A ledger adds `gnomon_ledger`, `gnomon_route` and `gnomon_memory`;
optional time calculations add `gnomon_temporal`. Discover the tools actually
available in the session instead of assuming every feature is enabled.

Forecasting and evidence operations return an `agent_summary` (or an
`agent_summary_read` call): the task scope, result, supporting evidence,
limitations and available follow-ups. Large details
remain retrievable without rerunning models. The host supplies observed outcomes
and decides when to review them; follow-ups do not run automatically.

Start the server with the [MCP quickstart](docs/quickstart-mcp.md).
[Response semantics and schema discovery](docs/agent-operations.md) explain how
to read results and recover from errors. Python, CLI and MCP share the session's
execution contract; low-level Python engine/ledger objects retain native results.

## Find the right workflow

| Task | Start here |
| --- | --- |
| Install or verify the Python environment | [Installation](docs/installation.md) |
| Complete a first forecast and outcome review | [First run](docs/getting-started.md) |
| Use connected tools as an agent | [Use Gnomon](skills/use-gnomon/SKILL.md) |
| Forecast and save results in Python | [Python skill](skills/forecast-with-gnomon/SKILL.md) · [API](docs/python-api.md) |
| Compare models on past observations | [Local evidence workflow](docs/local-evidence-workflow.md) |
| Configure a router that learns from outcomes | [Adaptive routing](docs/adaptive-routing.md) |
| Review recorded outcomes or recall lessons | [Ledger skill](skills/use-gnomon-ledger/SKILL.md) · [Memory](docs/memory-bridge.md) |
| Research forecast-led trading | [Trading skill](skills/trade-with-gnomon/SKILL.md) |
| Find an exact argument or resolve an error | [CLI/schema reference](docs/cli-reference.md) · [Troubleshooting](docs/troubleshooting.md) |

[All guides](docs/README.md) group the detailed contracts and worked examples by task.

## Bring your own model

Register your model as a callable:

```python
from gnomon import ForecastRequest, ForecastResult, InferenceEngine

def my_forecaster(request):
    # Replace this baseline with your preferred model.
    return ForecastResult((request.history[-1],) * request.horizon)

engine = InferenceEngine()
engine.register("my-model", my_forecaster)
execution = engine.forecast("my-model", ForecastRequest((10, 12, 11), 2))
print(execution.result.point)  # (11.0, 11.0)
```

StatsForecast, NeuralForecast, Darts or your own code: wrap the call and return a
`ForecastResult`. Use `register_factory` for a fresh model on each evaluation fold.
Gnomon checks inputs and outputs; you choose and install the model software.
See [provider integration](docs/production/INFERENCE.md).

For a scoreable ledger record, give the forecast a nonempty `series_id` and
explicit `future_timestamps`. Every ledger timestamp needs an explicit timezone
such as `+00:00`; each actual must use the forecast's exact `series_id`, unit and
one of its future timestamps. The [first-run guide](docs/getting-started.md#record-and-score-a-forecast)
shows the complete CLI loop.

## Optional connectors

Local models need no account. To connect optional Ephemeris models, run
`gnomon connect ephemeris`; enter credentials in the hidden terminal prompt.
For Hermes, `gnomon connect ephemeris --install-hermes-skill` installs native
secure setup. See [connection setup](docs/ephemeris-onboarding.md).


Ephemeris is one connector for remote time-series inference. Set its deployment
URL and credentials in operator configuration; they are never agent tool arguments.
Local models work without it.
See [connector setup](docs/production/INFERENCE.md#ephemeris).
Agents can follow the portable [Ephemeris setup skill](skills/setup-gnomon-ephemeris/SKILL.md)
for installation, credentials, MCP configuration and API documentation links.
For trading research and execution workflows across brokers, exchanges and
competitions, use [Trade with Gnomon](skills/trade-with-gnomon/SKILL.md), including
StatsForecast and Ephemeris integration examples.

## Keep the history straight

The optional SQLite ledger saves forecasts, revised actuals, scores and decisions.
Later corrections do not overwrite earlier predictions. You can ask what was known
at a particular time, find forecasts that need scoring, and compare models on
matched past results. Reading stored evidence does not rerun models. Gnomon does
not automatically retrain them.

Two kinds of memory serve different purposes: [router episodic memory](docs/adaptive-routing.md#episodic-memory)
uses similar completed forecasts to select a model; [agent evidence recall](docs/memory-bridge.md)
returns recorded decisions and lessons for an agent to inspect. Recalling a lesson
does not itself choose a model or validate its written explanation.

Optional date and time tools handle timezones, calendar shifts, intervals and event
order. They calculate supplied facts; they do not claim to improve an LLM's reasoning.

## Related tools and research

Gnomon brings model discovery, forecasting, evaluation and outcome review into a
shared agent workflow. Several projects cover related parts of that workflow:

| Project | Focus and overlap |
| --- | --- |
| [sktime-mcp](https://github.com/sktime/sktime-mcp) | Exposes sktime's estimator registry and execution through MCP so agents can discover, compose and run time-series workflows. |
| [AutoGluon-TimeSeries](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html) | Automates training, selection and ensembling across forecasting models. |
| [Darts](https://github.com/unit8co/darts) | Provides a common Python interface for forecasting models, backtesting, covariates and ensembles. |
| [Nixtla / TimeGPT](https://github.com/Nixtla/nixtla) | Provides a pretrained forecasting model and API tooling for forecasting and anomaly detection. |
| [FASE](https://arxiv.org/abs/2609.32689) | Research on forecasting agents that combine episodic memory with online ranking-policy learning from delayed outcomes. |

Forecasting libraries can supply models through Gnomon's
[provider interface](docs/production/INFERENCE.md). Gnomon adds recorded outcomes
and evidence retrieval around those calls; model software is installed separately.
Its optional [FASE-inspired memory settings](docs/adaptive-routing.md#episodic-memory)
implement parts of the research approach, not the paper's full agent or learned
ranking policy. [Matched replay results](benchmarks/memory_release_eval/RESULTS.md)
were mixed, so these settings remain opt-in. Gnomon does not claim superior
forecasting accuracy over the projects above.

## Status and guides

The provider-neutral execution API is stable. Live-service verification and a
real-agent comparison remain pending. A forecast is not permission to act; model quantiles are not proof
of calibrated uncertainty. See [validation and limits](docs/agent-evaluation.md).

- [First run](docs/getting-started.md) · [Python API](docs/python-api.md) · [CLI](docs/cli-reference.md)
- [Ledger](docs/production/OPERATIONS.md) · [Time calculations](docs/production/TEMPORAL.md) · [All docs](docs/README.md)
- [Changelog](CHANGELOG.md)

*A gnomon is the part of a sundial that casts the shadow.*
