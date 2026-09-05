# Gnomon production delivery plan

Gnomon helps agents analyse time series, execute models from user-selected libraries or providers, and make decisions using explicit evidence. The user authorized implementation and a ruthless, recoverable cull on 2026-09-05.

## Completion contract

The weighted acceptance checks in [progress.json](progress.json) total exactly 100. Credit is awarded only for verified completed checks with evidence. This is a delivery score against a scoped release contract, not a universal guarantee of correctness. Existing code receives credit only after inspection and verification. A regression reopens the affected check.

100 means all retained production capabilities, documented installation paths, migrations, public entry points, conformance/regression gates and actual TSFM integration meet the recorded acceptance criteria. Mock HTTP integration does not satisfy the live-service gate. Publishing packages, deploying services or sending external messages is not implied by preparing a production codebase.

## Workstreams

| Workstream | Points |
|---|---:|
| audit | 8 |
| semantics | 14 |
| protocol | 14 |
| ledger | 16 |
| api | 12 |
| evaluation | 10 |
| surface | 10 |
| optional | 6 |
| measurement | 5 |
| release | 5 |
| Total | 100 |

## Execution sequence

1. Audit the code and dependency graph; establish the full test baseline. Remove retired experiments and obsolete designs from the active tree. Preserve independent production regressions and recoverable Git history. The user explicitly authorized the broader docs/benchmark/output cull on2026-09-06; checkpoint2cba20e retains the removed evidence and implementations.
2. Fix exact quantity, time and scope semantics. Keep valid time, source availability and local recording time distinct. Remove automatic-action claims from mere descriptive correctness.
3. Extend the existing forecasting protocol into the public callable/provider boundary. Add the minimal immutable execution and outcome ledger alongside it.
4. Route the actual inference API through the same boundary; remove local-catalogue assumptions. Separate inference from optional budgeted evaluation.
5. Make historical routing and evaluation consume revision-aware, cutoff-bound ledger records.
6. Simplify Python/CLI/MCP results and tool registration; isolate advanced context and bounded general-temporal functionality.
7. Measure the lean/full tool configurations against matched ordinary-agent tasks. Complete migration, distribution, operating documentation and live-service release checks.

## Iteration and context handoff

At each iteration: read HANDOFF.md and progress.json, select the highest-value unmet check, inspect its current implementation, make a bounded change, run relevant verification, update evidence and handoff, then continue. Run broad gates at integration boundaries. Never clear state by restarting the repository or overwriting user work.

The available agent tools do not expose a manual conversation-compaction operation. Persist concise checkpoints so automatic compaction and resumed turns can continue without relying on conversational memory. Do not claim that a context reset happened unless it did.

## Storage and extension boundaries

- Optional persistent SQLite metadata plus immutable artifacts; files and externally supplied snapshots remain first-class inputs.
- Separate unique execution identity from content/cache fingerprints.
- Append observation revisions and evaluations; preserve original predictions and metric definitions.
- Keep one package initially. User callables own library-specific objects/configuration. Remote-service credentials are operator configuration.
- Forecast, anomaly, imputation and transformation outputs need separate task semantics; shared infrastructure must not silently drop capabilities.
- Retain baseline and temporal-integrity checks. Keep model inference, measured accuracy, scenario assumptions and permission to act distinct.
