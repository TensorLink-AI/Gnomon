# LLMs, API keys, and agent integrations

## v0.1 does not use an LLM

The current runtime does not call OpenRouter, OpenAI, Anthropic, Google, Hermes,
or any other model provider. It does not read provider API-key environment
variables. Forecasting is local and deterministic, so no API key is required.

Setting `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` currently
has no effect. There is also no `aion mcp serve` command in v0.1.

## Is OpenRouter a planned option?

Yes, OpenRouter is a reasonable future provider because it offers a common API
for multiple models. It is not implemented yet. Provider support should be
adapter-based so a user can choose OpenRouter, a direct model vendor, or a local
OpenAI-compatible endpoint without changing the numerical runtime.

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

The following is a design example, not working v0.1 configuration:

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

