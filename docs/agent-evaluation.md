# Evaluating whether Aion improves an agent

AionBench compares the same agent and model on the same task IDs under two
conditions: a control without Aion and a treatment with Aion. Keep the system
prompt, model, temperature, token budget, data, and grader identical. Run each
task repeatedly when measuring a nondeterministic agent.

Each JSONL row is one graded run:

```json
{"task_id":"capacity-001","success":true,"temporal_leakage":false,"invented_number":false,"warning_omission":false,"appropriate_abstention":false,"tool_calls":3,"latency_seconds":4.1,"cost_usd":0.01}
```

Required fields are `task_id` and `success`. Safety, cost, latency, and tool-use
fields default to zero/false when omitted. Baseline and treatment files must
contain identical task ID sets.

```bash
aion eval compare \
  --baseline results/hermes-control.jsonl \
  --treatment results/hermes-aion.jsonl
```

The output reports absolute task-success uplift, relative error reduction,
safety deltas, average tool calls, latency, and cost. `examples/aionbench/`
contains format demonstrations only; they are synthetic and must not be quoted
as measured Aion performance.

Recommended initial task families are inventory decisions, capacity planning,
event-aware demand, anomaly investigation, unsupported-data abstention, and
temporal-leakage traps. Grade final decisions programmatically wherever
possible. Use an LLM judge only for explanation quality, separately from the
primary task-success score.

## External benchmarks

Beyond the internal task families, [`benchmarks/`](../benchmarks/README.md)
contains faithful adapters for published benchmarks — Context is Key
(context-aided forecasting, RCRPS) and AnomLLM (anomaly detection, F1) —
whose official metrics stay authoritative and whose runners also emit
AionBench JSONL rows, so the same `aion eval compare` treatment/control
comparison works there too. LLM conditions are served through OpenRouter
so control and treatment share one model and provider.

## Hermes lifecycle

1. Call `aion_forecast` with a `project`.
2. Link the action to its returned tracking ID with `aion_record_decision`.
3. Periodically call `aion_list_open_forecasts`; act on entries in `due` state.
4. Call `aion_submit_actuals` with the complete horizon.
5. Resolve the business result with `aion_resolve_decision`.
6. Use `aion_model_performance` as descriptive evidence, never as proof that a
   model caused the outcome.

This separates two claims: forecast quality and agent task improvement. The
headline Aion claim should use the treatment/control task-success result.
