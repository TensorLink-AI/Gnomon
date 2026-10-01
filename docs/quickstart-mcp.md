# Connect an agent

For copyable host configurations, see [client recipes](../integrations/mcp/README.md).
Use `gnomon_capabilities` with `{"task":"forecast"}` for a task-specific
example and setup requirements on supported builds.

Install Gnomon in the environment your host will execute, then configure the host
to run `gnomon mcp serve` over stdio. No HTTP server is required.

```json
{"command":"gnomon","args":["mcp","serve","--providers-config","/absolute/path/providers.toml"]}
```

Omit the configuration argument for the three offline baselines and any optional
saved Ephemeris connection. With no saved connection, startup needs no credentials.
Use an explicit operator configuration when you need a fixed provider set.
The default session exposes inspect, describe, capabilities, forecast, evaluate
and read. A ledger adds ledger, route and memory; `enable_temporal=true` adds temporal.
All interfaces use the same execution contract.

## First agent call

Have the host discover `tools/list`, then use the [main agent skill](../skills/use-gnomon/SKILL.md).
The following is a tool-call description for the host, not a raw JSON-RPC message:

```json
{"name":"gnomon_forecast","arguments":{"provider":"last_value","request":{"history":[10,12,11],"horizon":2}}}
```

It returns two points of 11. Read `agent_summary` for the scope, method and
limitations, and `result.point` for the full forecast. If the response supplies
`agent_summary_read`, execute that read call instead. No model comparison or
forecast-accuracy claim is implied. This smoke call needs no remote service.

For file data, inspect it once and reuse its `data_ref` in the same session.
For unknown models, call `gnomon_capabilities` before forecasting. See the
[complete stdio example](mcp-evidence-workflow.md) for raw protocol messages,
initialization and pagination.

## Results and follow-ups

The [common overview](agent-operations.md#common-result-overview) keeps the result
separate from its supporting evidence. Recorded forecasts include outcome
follow-ups when their identity and target timestamps permit scoring. Supply
missing `requires` fields from the task and observed actuals; the host controls
when these calls run. Tool availability does not enable write permissions.

Retain each forecast's `completion` object and use `final_selection` for its
canonical final JSON selection. The host can resolve prose finals against
trusted task-matching executions with `gnomon.resolve_final_selection`; see
[final-answer preservation](final-selection.md). A successful tool call and
strict final-answer conformance are separate outcomes.

Use the [agent skill](agent-skill.md) for model choice, cutoff semantics and
safe retrieval. The host should consume `tools/list` rather than copying schemas.

Data/result references survive across calls in this process. The server closes
them on exit. Large results are paged through `gnomon_read`; persistent history
requires a ledger. Tool errors do not terminate the connection.

Startup configuration owns provider entrypoints, endpoints, authentication, limits and write
permissions. Agent arguments cannot enable them.
[Configuration and full contracts](production/INFERENCE.md).

## Exact schema discovery

Tool declarations include top-level argument properties as well as canonical
variants. If a host simplifies the declaration, retrieve the exact schema as
ordinary tool-result data:

```json
{"name":"gnomon_capabilities","arguments":{"schema_tool":"gnomon_ledger","schema_variant":"review_decision"}}
```

Omit `schema_variant` to list variant IDs and get the complete schema. Operation
names identify ledger operations; forecast/evaluation variants without an
operation discriminator use the returned numeric string IDs. Discovery only
returns tools and write operations enabled in the current session. Server-side
validation remains strict even when the host displays a simplified schema.

`gnomon_memory` recalls bounded evidence with explicit source and recording
cutoffs. See [memory recall](memory-bridge.md#cli-mcp-and-automatic-recall) and the
optional [Hermes integration](../integrations/hermes/gnomon-memory/README.md).

Operator-configured routers appear under `gnomon_capabilities` → `ledger.routers` and
are called through `gnomon_forecast` with the router's name as `provider`. See
[adaptive routing](adaptive-routing.md).

## Sharing evidence between processes

For remote agents sharing project evidence across client and server restarts, use
the separately installed [hosted MCP service](hosting/README.md). It provides
authenticated Streamable HTTP and legacy SSE, with optional verified Ditto recall.
The local stdio server above remains the lightweight single-process interface.
