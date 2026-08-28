# LLMs, API keys, and agent integrations

## Gnomon does not use an LLM

The default runtime does not call OpenRouter, OpenAI, Anthropic, Google,
or any other LLM provider. It does not read LLM-provider API-key
environment variables. Forecasting is local and deterministic by default, so
no account or API key is required.

This is distinct from the explicit TSFM API backend. A CLI/Python caller may
configure `backends.api` in `gnomon.toml` to send numerical series to a named
forecast endpoint. That path is off by default and requires deliberate
configuration; MCP calls do not inherit ambient project config. Hosted
inference must therefore be an explicit, disclosed project decision, never a
silent upgrade from local execution.

Setting `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` currently
has no effect. `gnomon mcp serve` exposes the tools over stdio MCP without any
LLM of its own.

## Bring your own brain

Gnomon ships LLM *workflows* (prompt + response schema + deterministic
validation) without shipping an LLM. `gnomon context prompt` emits an
Gnomon-owned extraction prompt for permitted documents; the host runs it on
its own model; `gnomon context validate` grounds and validates the response
into typed context events and non-event hypotheses. An MCP host may run that
prompt with its own model, so no API key is configured on the Gnomon side. Events
enter a forecast only through the deterministic admission gate
(identical-fold ablation) — or, behind `context.future_events: on`, through
the textual-verifiability lane for future-dated `constraint:*`/`override:*`
events, whose numbers are re-parsed deterministically from evidence quotes
verified verbatim against the caller's own documents. Forecasts that lane
influences report the distinct `context_trusted` support state.

Validation also returns an immutable, content-addressed `context_receipt`
covering source fingerprints, accepted and rejected proposals, compiler
identity, and prompt version. Persist and replay that receipt when comparing
execution surfaces or repeating a run; do not pay for or introduce variance
from a fresh compiler call. Each executable event carries the receipt ID into
the forecast artifact.

The compiler treats text in two lanes. Bounded, dated events may enter the
existing admission machinery. Claims about seasonality, units, relationships,
or operational constraints are returned as `hypotheses` with their verbatim
quote and source, status `proposed_for_numeric_verification`, and
`may_affect_numbers: false`. A host may ask `gnomon_describe` to test those
hypotheses; the LLM's wording alone never changes a forecast.
For soft events it may additionally classify a qualitative effect family,
direction, and duration. It cannot supply magnitude: common numeric-effect
attribute spellings are stripped, and the runtime reports the event as
`scenario_only` unless measured history and the admission gate support a
numeric candidate. A novel, directionally grounded event may produce a
separate `hypothetical_sensitivity` path using one robust innovation scale
from the target history. Hosts must present that path as a what-if sensitivity,
never as the primary forecast, expected effect, or probability-bearing claim.

On the MCP surface, use `context_events` only when the source states a literal
numeric bound, override, or cessation. When direction or shape is known but
magnitude is not, send `qualitative_context_events` with the verbatim
`source_span`, timing, direction, effect family, and duration. Gnomon keeps the
history-only primary immutable and exposes the claim only as a labelled,
non-automatable sensitivity scenario. Use `context_submission` for non-event
claims and dossiers. Malformed context is rejected with typed repair
information rather than silently ignored or coerced into a numeric claim.
When timing is ambiguous, a fact was first known after the cutoff, no temporal
mechanism is stated, or a third party is merely forecasting a value, send the
compact `context_rejections` array. Each item requires `context_id`,
`reason_code`, `reason`, and the verbatim `source_span`; Gnomon persists that
disposition without changing any number. It rejects qualitative event windows
whose effective start date is absent from the quoted source, preventing an
agent from converting “sometime next month” into an invented timestamp.

### Provider-neutral sampled priors

For `best_effort` or `scenario` workflows, a host may let its own model propose
a conditional numeric path while Gnomon retains the immutable primary. The
shipped `gnomon.agent_context` module provides the integration primitives:

- `build_sampled_context_prior_prompt(...)` encodes host-owned regular grids
  compactly and asks for indexed values without re-echoing timestamps;
- `recommended_initial_sample_count(horizon)` starts with three independent
  paths, while `recommended_sample_count(horizon)` provides the expansion cap;
- `candidate_from_sampled_paths(...)` rejects malformed, non-finite, partial,
  or wrong-grid paths independently, aggregates valid paths into q10/q50/q90,
  and reports draw stability separately from historical skill; and
- `sampled_prior_sufficiency(...)` decides whether the elicitation is coherent
  enough to become a human-facing recommendation. It checks path survival and
  scale-free agreement only; it does not claim historical forecast skill.

The host then attaches the validated paths with
`gnomon.llm_dossier.attach_host_candidate_elicitation` and submits the sealed
dossier through `gnomon_forecast.temporal_dossiers`. A host should request the
initial sample, assess it, and expand to the cap only when paths are malformed
or materially incoherent. Too few valid paths, a low valid-path fraction, or
material disagreement keeps the prior visible as a labelled scenario but
prevents it from displacing the primary. A passing result is still
`prior_assisted`, requires human review, and is never automation eligible
unless separate historical admission exists. These helpers make no network
calls and accept no provider credentials.

### Preserve an independent decision prior

For consequential threshold decisions, a one-call evidence prompt can anchor a
model and erase useful temporal judgment it held before seeing Gnomon. A host
may preserve that view without granting it authority:

1. ask the model the user's unchanged question without Gnomon evidence;
2. validate the structured answer and call
   `seal_temporal_decision_prior(...)` with a hash of that exact question;
3. compute the normal immutable Gnomon packet; and
4. call `build_temporal_decision_reconciliation(...)` to expose agreements and
   conflicts under a sealed selection policy.

The reconciliation response is accepted through
`seal_temporal_decision_selection(...)`. It must name the selected source, a
distinct counterevidence source when the inputs conflict, confidence, and a
falsifiable `what_would_change` condition. Missing reasoning slots are a typed
contract failure rather than silently successful “reconciliation.”

The receipt is always `prior_assisted`, host-attested as captured before
evidence, and automation-ineligible. Reconciliation may choose a human-facing
action, but it cannot edit either input, upgrade support, mutate the primary,
or authorize automation. Hosts should outcome-score those choices before
making this a default; the helper preserves a prior, it does not prove that the
prior is skilled.

Record sealed selections through the existing temporal-synthesis tracking
lane and resolve them when outcomes arrive. `gnomon track decision-skill`
(or MCP `gnomon_track` with `action: "decision_skill"`) reports paired,
shrunk skill for each proposer. Reconciliation accepts that evidence only when
its proposer identity matches and its `known_at` precedes the decision cutoff.
Cold-start or non-graduated evidence stays visible but cannot create authority.

## Is OpenRouter a planned option?

Not in the runtime, and there is no committed date. OpenRouter is a
reasonable future provider because it offers a common API for multiple
models; provider support should be adapter-based, so a user can choose
OpenRouter, a direct model vendor, or a local OpenAI-compatible endpoint
without changing the numerical runtime.

One thing to be aware of if you are grepping the repository: `benchmarks/`
*does* call OpenRouter, and reads `OPENROUTER_API_KEY`. That is the
benchmark harness serving LLM **controls** to compare against Gnomon — it is
not part of the runtime. A default forecast makes no model-provider network
call; only an explicitly enabled TSFM API backend does.

## Intended LLM boundary

The rule of thumb, enforced in code by `contracts.PARAMETER_AUTHORITY`:
**the model may point, filter, frame, and prefer; it may never assert a
measurement.** Every caller-settable parameter answers one of three
questions —

- **intent** — what does the caller want computed? Free: a preference
  cannot make a number wrong.
- **data** — what is the caller's data? Validated, fingerprinted, and
  disclosed when inferred or supplied inline (`provenance: inline` on the
  artifact task block).
- **epistemic** — what counts as evidence? Priced: moving one off its
  default always leaves a trace in the artifact (a typed reason, a
  distinct support state, or a structured refusal), and
  `tests/test_parameter_authority.py` fails CI if a parameter reaches a
  front door unclassified or an epistemic one is unpriced.

Concretely, an LLM layer may:

- map user intent to typed task fields;
- propose timestamp, target, and series columns;
- search explicitly permitted local context;
- propose context events (verbatim-quote-verified) and *nominate* — never
  pick — an effect shape via `expected_shape`;
- extract quoted temporal hypotheses for deterministic verification;
- propose a separately sealed conditional path in an explicit `best_effort` or
  `scenario` lane; it remains `prior_assisted`, human-reviewed, outcome-scored,
  and non-automatable unless independent historical evidence later admits it;
- restrict the model contest via `candidates` (the mandatory baselines
  always compete, and the restriction is disclosed);
- propose bounded experiments; and
- explain immutable forecast artifacts and evidence.

It must not generate, edit, or override the immutable primary forecast,
evaluation scores, engine quantiles, model selection, warnings, or abstention.
A model-authored conditional path must remain in its typed lane and provider
failure must leave the primary numerical result unchanged.

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

Foundation-model sandboxes are installed per model with
`gnomon tsfm install <name>` or, on the `full` MCP profile, with
`gnomon_install_tsfm`. The MCP tool launches the long-running installation
detached and supports status-only polling. An agent should read
`gnomon_capabilities` first — `models.tsfm_available` lists what can be
installed, `models.tsfm_sandboxes` what already is, and
`models.tsfm_capabilities[<name>].tasks` which tasks each model has
verifiably implemented (`forecast`, and for `moment_small` also
`detect_anomalies` / `impute` / `embed`). If a wanted model is missing,
an agent with shell access can run the install command itself (uv must be
installed; weights download from the Hugging Face Hub on first
inference); a tool-only agent on a narrow profile should ask the operator to
run it. Once the sandbox exists there is nothing else to wire: the model competes in the
next evaluated run and is selected only if it wins the backtest or
grader.
