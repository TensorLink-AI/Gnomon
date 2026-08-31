# Evaluating whether Gnomon improves an agent

GnomonBench compares the same agent and model on the same task IDs under two
conditions: a control without Gnomon and a treatment with Gnomon. Keep the system
prompt, model, temperature, token budget, data, and grader identical. Run each
task repeatedly when measuring a nondeterministic agent.

Each JSONL row is one graded run:

```json
{"task_id":"capacity-001","success":true,"temporal_leakage":false,"invented_number":false,"warning_omission":false,"appropriate_abstention":false,"tool_calls":3,"latency_seconds":4.1,"cost_usd":0.01}
```

Required fields are `task_id` and `success`. Cost, latency, and tool-use
fields default to zero when omitted. The safety fields (`temporal_leakage`,
`invented_number`, `warning_omission`) are different: omitted means
*unmeasured*, and the comparator reports them as unmeasured rather than as
a rate of zero — a grader that checked and found nothing writes an explicit
`false`. Baseline and treatment files must contain identical task ID sets
with no duplicate IDs. Rows carrying `row_abstained` (the harness ended the
run without an answer) are excluded from every rate and counted separately.

```bash
gnomon eval compare \
  --baseline results/agent-control.jsonl \
  --treatment results/agent-gnomon.jsonl
```

The output reports absolute task-success uplift, relative error reduction,
safety deltas, average tool calls, latency, and cost. `examples/gnomonbench/`
contains format demonstrations only; they are synthetic and must not be quoted
as measured Gnomon performance.

Recommended initial task families are inventory decisions, capacity planning,
event-aware demand, anomaly investigation, unsupported-data abstention, and
temporal-leakage traps. Grade final decisions programmatically wherever
possible. Use an LLM judge only for explanation quality, separately from the
primary task-success score.

### Temporal-leakage traps

`benchmarks/leaktrap/` implements this family. Each task is a bitemporal
series carrying its own publication dates, built so that **reading past the
cutoff measurably helps**: the horizon opens with a shock nothing in the
pre-cutoff history predicts, and the last few pre-cutoff observations are
published low and corrected afterwards. The post-cutoff rows are in the file
with honest `published` dates — nothing is hidden, and what is tested is
whether the forecaster respects them.

Three conditions:

- `oracle-leak` — deliberately ignores the dates. It exists to validate the
  family: if leaking does not help, "structurally cannot leak" guards against
  a harm nobody was at risk of, and every other number here is void. **Run it
  first and read `mean_leak_advantage`.** Measured at +80% on the shipped
  generator.
- `gnomon` — ingests with `--known-at` and forecasts at `--as-of <cutoff>`.
- `control` — a model gets the same file, dates included.

Two kinds of claim are reported, and they are not interchangeable. The
*measured* one is `leak_advantage`, relative to a no-leak ceiling computed by
brute force over every built-in model on the vintage series plus a
revision-aware correction — the correction matters, or a control that
legitimately learned the revision pattern from settled history would be
accused of leaking for being clever. The ceiling picks its strategy with
hindsight, so it is optimistic: an honest condition scores *above* it, and
that gap is not a finding. The *structural* claim is not a score at all —
the run's own `snapshot_access` evidence records the maximum `known_time`
served, and the grader asserts it is at or before the cutoff. A condition
with no access log cannot make that claim, and is reported as
`asserted: false` rather than as a pass.

## External benchmarks

Beyond the internal task families, [`benchmarks/`](../benchmarks/README.md)
contains adapters for Context is Key, AnomLLM, MTBench, TimeSage-MT and
TemporalBench. Each benchmark README states where the official scorer runs and
where a disclosed local metric is necessary. Lower-layer engine, compiler,
policy and safety-contract runs must not be described as agent-reasoning lift.

LLM comparisons match the model, endpoint, sampling settings and task IDs.
OpenRouter is the default; adapters with an OpenAI-compatible `--base-url` may
use another endpoint such as Engy. Provider identity travels with the result
because the same model name served by a different endpoint is a different
measurement. Smoke shards remain smoke evidence and cannot be promoted into a
full benchmark claim.

## Agent lifecycle

1. Call `gnomon_forecast` with a `project`.
2. If an action is required, call `gnomon_decide` to create a governed
   decision artifact.
3. Periodically call `gnomon_status` with `section: "open_forecasts"`; act on
   entries in `due` state.
4. Call `gnomon_submit_actuals` with the complete horizon.
5. Resolve the business result with `gnomon_resolve_outcome`.
6. Call `gnomon_status` with `section: "performance"` for descriptive
   evidence, never as proof that a model caused the outcome.

This separates two claims: forecast quality and agent task improvement. The
headline Gnomon claim should use the treatment/control task-success result.
