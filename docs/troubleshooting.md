# Troubleshooting

Start with the installed environment and exposed schemas. Repository docs may
describe changes newer than that installation. `gnomon environment` identifies
the interpreter/build; `gnomon capabilities` lists available providers. Pass the
same `--providers-config` as the original operation when one was configured.

| Symptom | Next check |
| --- | --- |
| Python cannot import Gnomon or a model library | [Installation](installation.md): use the same interpreter for Gnomon and local providers |
| Agent cannot find a tool | [MCP setup](quickstart-mcp.md): inspect `tools/list`; ledger and temporal tools depend on startup settings |
| Tool schema lost operation-specific fields | `gnomon_capabilities` with `schema_tool` and the intended `schema_variant`; see [schema discovery](quickstart-mcp.md#exact-schema-discovery) |
| Response contains a read call rather than all data | [Bounded retrieval](agent-operations.md#common-result-overview): follow the returned call in the originating session |
| Forecast ran but outcomes are pending or partial | [Scoring coverage](scoring-and-recovery.md): check exact target times, units and both availability cutoffs |
| Historical comparison has no eligible evidence | [Prospective comparison](production-history-comparison.md): check timing, complete matched outcomes and provider revisions |

For a normal task rather than a failure, start with the
[main workflow](../skills/use-gnomon/SKILL.md) or [guide directory](README.md).

- Unknown provider: check `gnomon capabilities --providers-config providers.toml`.
  Install/load the model in your own software and register its callable or factory.
- Unsupported request: check provider capabilities, exact shapes, frequency and
  cutoff fields. Unsupported covariates/quantiles are errors, not silently dropped.
- Invalid input: select columns explicitly and inspect repairs before forecasting.
  Mixed naive/aware timestamps and ambiguous revisions need explicit semantics.
- Expired reference: inspect again, or retrieve durable evidence from the ledger.
- Result retention limit: execution may already have happened. Check the receipt
  before retrying paid work.
- No routing winner: retain the baseline fallback; missing comparable evidence is
  not proof that a model is worse.

[Provider and evaluation limits](production/INFERENCE.md).
