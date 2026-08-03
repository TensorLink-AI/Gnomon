# LLMs, API keys, and agent integrations

## Gnomon does not use an LLM

The current runtime does not call OpenRouter, OpenAI, Anthropic, Google, Hermes,
or any other model provider. It does not read provider API-key environment
variables. Forecasting is local and deterministic, so no API key is required.

Setting `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` currently
has no effect. `gnomon mcp serve` exposes the tools over stdio MCP without any
LLM of its own.

## Bring your own brain

Gnomon ships LLM *workflows* (prompt + response schema + deterministic
validation) without shipping an LLM. `gnomon context prompt` emits an
Gnomon-owned extraction prompt for permitted documents; the host runs it on
its own model; `gnomon context validate` grounds and validates the response
into typed context events. The Hermes plugin wires this through the host's
`ctx.llm` facade, so no API key is ever configured on the Gnomon side. Events
enter a forecast only through the deterministic admission gate
(identical-fold ablation) — or, behind `context.future_events: on`, through
the textual-verifiability lane for future-dated `constraint:*`/`override:*`
events, whose numbers are re-parsed deterministically from evidence quotes
verified verbatim against the caller's own documents. Forecasts that lane
influences report the distinct `context_trusted` support state.

## Is OpenRouter a planned option?

Not in the runtime, and there is no committed date. OpenRouter is a
reasonable future provider because it offers a common API for multiple
models; provider support should be adapter-based, so a user can choose
OpenRouter, a direct model vendor, or a local OpenAI-compatible endpoint
without changing the numerical runtime.

One thing to be aware of if you are grepping the repository: `benchmarks/`
*does* call OpenRouter, and reads `OPENROUTER_API_KEY`. That is the
benchmark harness serving LLM **controls** to compare against Gnomon — it is
not part of the runtime, and no forecast path touches it. Installing Gnomon
and forecasting never makes a network call to a model provider.

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
GNOMON_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=<secret>
GNOMON_LLM_MODEL=<provider/model-name>
```

Before implementing this interface, Gnomon needs provider-independent typed
contracts, secret-redaction tests, explicit network disclosure, timeout and cost
budgets, prompt-injection defenses, and deterministic no-LLM fallbacks.

## Using an external agent today

An external agent can invoke the CLI as a subprocess and parse its JSON output.
Treat `gnomon capabilities`, the structured error envelope, and persisted
artifact values as authoritative. The agent should preserve support and warnings
verbatim and must not manufacture values for unsupported series.

## Agents and TSFM sandboxes

Foundation-model sandboxes are installed per model, from the shell, never
through an MCP tool: `gnomon tsfm install <name>`. An agent should read
`gnomon_capabilities` first — `models.tsfm_available` lists what can be
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

