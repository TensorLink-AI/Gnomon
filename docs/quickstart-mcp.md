# Connect an agent

Install Gnomon in the environment your host will execute, then configure the host
to run `gnomon mcp serve` over stdio. No HTTP server is required.

```json
{"command":"gnomon","args":["mcp","serve","--providers-config","/absolute/path/providers.toml"]}
```

Omit the configuration argument for the three offline baselines.
The default session exposes inspect, describe, capabilities, forecast, evaluate
and read. A ledger adds ledger and route; `enable_temporal=true` adds temporal.
All interfaces use the same execution contract.

Use the [agent skill](agent-skill.md) for model choice, cutoff semantics and
safe retrieval. The host should consume `tools/list` rather than copying schemas.

Data/result references survive across calls in this process. The server closes
them on exit. Large results are paged through `gnomon_read`; persistent history
requires a ledger. Tool errors do not terminate the connection.

Startup configuration owns provider entrypoints, endpoints, authentication, limits and write
permissions. Agent arguments cannot enable them.
[Configuration and full contracts](production/INFERENCE.md).
