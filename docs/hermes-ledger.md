# Use the ledger with Hermes

Gnomon supplies forecast evidence; Hermes keeps its normal skills and memory.
No special memory provider or synchronization service is required. This workflow
needs a persistent project ledger configured by its operator.

Install Gnomon in an environment Hermes can execute. Make the
[`use-gnomon-ledger` skill](../skills/use-gnomon-ledger/SKILL.md) available at
`$HERMES_HOME/skills/use-gnomon-ledger/SKILL.md` (normally under `~/.hermes`).
New Gnomon wheels include the directory under
`share/gnomon/skills/use-gnomon-ledger` in their environment prefix; older
published wheels do not include this new skill, so copy it from this checkout.

Create an operator-owned configuration for your project:

```toml
schema_version = 1
ledger_path = "evidence.db"
```

Relative ledger paths resolve against this TOML file. For authorised actual,
decision-summary or lesson writes, also set `allow_outcome_writes = true`.
Do not add that permission merely to inspect historical evidence.

In Hermes's existing `config.yaml`, merge this entry into `mcp_servers`:

```yaml
mcp_servers:
  gnomon:
    command: /absolute/path/to/gnomon
    args: [mcp, serve, --providers-config, /absolute/path/to/providers.toml]
```

Restart the Hermes session to discover the server and skill. A suitable request:

> Use the use-gnomon-ledger skill to review this project's recorded forecasts.
> Compare the specified providers for my series, horizon and metric at the
> supplied evidence cutoffs. Show matched counts and exclusions. Save a short
> memory note with references so we can revisit it when actuals arrive.

Supply those task facts or let the agent ask for the missing choices. The skill
does not choose business objectives, declare timezones or infer context labels.
The earlier [record-and-score example](getting-started.md#record-and-score-a-forecast)
shows how forecast identity and actuals match. For decision reviews and versioned
lessons, see [decision memory](decision-memory.md).

Keep the same configured ledger between Hermes sessions. Save execution, study,
decision or lesson IDs; `data_ref` and `result_ref` expire with the MCP process.
Hermes memory should contain concise context and retrieval references. Query the
ledger again before treating an old ranking or lesson as current evidence.

## Discovery and limits

If an agent host omits operation-schema branches from tool descriptions, inspect
`gnomon ledger --schema` through the CLI rather than guessing the operation.
Context-filtered comparisons need `compare_context`; dropping that filter
answers a different question even when the call succeeds.

This integration supplies verifiable evidence and does not establish a
forecasting or agent-performance advantage.

## Optional Ephemeris models

The installed `use-gnomon` skill also describes optional signup through MCP
capabilities. Hermes should offer signup once, show the human the returned link
if accepted, and continue with local models if declined. See
[Ephemeris connection setup](ephemeris-onboarding.md).
