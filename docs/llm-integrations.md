# LLMs, API keys, and agent integrations

## Aion does not use an LLM

The current runtime does not call OpenRouter, OpenAI, Anthropic, Google, Hermes,
or any other model provider. It does not read provider API-key environment
variables. Forecasting is local and deterministic, so no API key is required.

Setting `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` currently
has no effect on the runtime. `aion mcp serve` exposes the tools over stdio MCP
without any LLM of its own.

The one place in this repository that *does* call a provider is
[AionBench](../benchmarks/README.md), the benchmark harness in `benchmarks/`.
It reads `OPENROUTER_API_KEY` to drive models across published
temporal-reasoning benchmarks with and without Aion's tools. It is a separate
package (`pip install -e '.[bench]'`), it is not importable from the runtime,
and it does not change the boundary described below: it measures the boundary,
it does not cross it.

## Bring your own brain

Aion ships LLM *workflows* (prompt + response schema + deterministic
validation) without shipping an LLM. `aion context prompt` emits an
Aion-owned extraction prompt for permitted documents; the host runs it on
its own model; `aion context validate` grounds and validates the response
into typed context events. The Hermes plugin wires this through the host's
`ctx.llm` facade, so no API key is ever configured on the Aion side. Events
enter a forecast only through the deterministic admission gate
(identical-fold ablation).

## Is OpenRouter a planned option?

Yes, OpenRouter is a reasonable future provider for the *runtime's* LLM
workflows because it offers a common API for multiple models. That is not
implemented yet. Provider support should be adapter-based so a user can choose
OpenRouter, a direct model vendor, or a local OpenAI-compatible endpoint
without changing the numerical runtime.

AionBench already uses OpenRouter for exactly that reason — one code path and
one key to compare open- and closed-weight models — but as a benchmark client,
not as a runtime adapter.

## Intended LLM boundary

An eventual LLM layer may:

- map user intent to typed task fields;
- propose timestamp, target, and series columns;
- search explicitly permitted local context;
- propose bounded experiments; and
- explain immutable forecast artifacts and evidence.

It must not generate, edit, or override forecast values, evaluation scores,
quantiles, model selection, warnings, or abstention. Provider failure must not
change numerical results.

## Proposed future configuration

The following is a design example, not working configuration:

```text
AION_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=<secret>
AION_LLM_MODEL=<provider/model-name>
```

Before implementing this interface, Aion needs provider-independent typed
contracts, secret-redaction tests, explicit network disclosure, timeout and cost
budgets, prompt-injection defenses, and deterministic no-LLM fallbacks.

## Using an external agent today

An external agent can invoke the CLI as a subprocess and parse its JSON output.
Treat `aion capabilities`, the structured error envelope, and persisted
artifact values as authoritative. The agent should preserve support and warnings
verbatim and must not manufacture values for unsupported series.

## Agents and TSFM sandboxes

Foundation-model sandboxes are installed per model, from the shell, never
through an MCP tool: `aion tsfm install <name>`. An agent should read
`aion_capabilities` first — `models.tsfm_available` lists what can be
installed, `models.tsfm_sandboxes` what already is, and
`models.tsfm_capabilities[<name>].tasks` which tasks each model has
verifiably implemented (`forecast`, and for `moment_small` also
`detect_anomalies` / `impute` / `embed`). If a wanted model is missing,
an agent with shell access can run the install command itself (uv must be
installed; weights download from the Hugging Face Hub on first
inference); a tool-only agent should ask the operator to run it. Once the
sandbox exists there is nothing else to wire: the model competes in the
next evaluated run and is selected only if it wins the backtest or
grader.

