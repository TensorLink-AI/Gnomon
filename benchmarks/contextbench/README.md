# ContextBench

ContextBench is Gnomon's internal matched benchmark for one narrow question:
does outside information improve forecasts when it is genuinely predictive,
while leaving the primary answer unchanged when it is irrelevant or cannot be
numerically justified?

It complements TemporalBench. TemporalBench measures realistic agent behavior
and context propagation, but its 50-point histories and 20–46 step horizons do
not provide the four separated rolling origins required by Gnomon's context
gate. ContextBench generates 480-point histories by default, with repeated
events and a sealed counterfactual.

## Families

| Family | Hidden construction | Required behavior |
| --- | --- | --- |
| `irrelevant` | grounded events with zero causal effect | consider, reject numerical influence, primary unchanged |
| `future_covariate` | irregular known schedule with a real additive effect | admit only after identical-fold lift; improve held-out sMAPE |
| `repeated_event` | aperiodic repeated pulses with known future occurrence | learn direction/shape from history; improve held-out sMAPE |
| `prior_only` | novel unsourced event with no local precedent | scenario-only; primary unchanged |

Every case has a matched history-only run and context-enabled run. The task
file contains only information available at the cutoff. Realized futures,
counterfactual futures, effect magnitude, onset, duration, and the influence
label live in a separately hashed oracle file. An arm never receives it.

## Generate and run

```bash
PYTHONPATH=src:. python -m benchmarks.contextbench.generate \
  --output-dir results/contextbench/corpus --fresh --per-family 20

PYTHONPATH=src:. python -m benchmarks.contextbench.run_contextbench \
  --corpus-dir results/contextbench/corpus \
  --output-dir results/contextbench/run
```

Use a fixed `--seed` while debugging. A decision-ready run requires `--fresh`,
at least 20 cases per family, all gates passing, and an unchanged corpus
manifest. `--allow-gate-failure` makes an exploratory run exit zero while
retaining failed gates in `summary.json`; without it, a completed run that
fails a product gate exits 2.

Outputs:

- `observations.jsonl`: per-case forecasts, gates, truth-aware scores and
  dispositions;
- `summary.json`: family-separated accuracy, admission precision/recall,
  false influence, coverage, leakage, confidence intervals and decision gates.

## LLM arms

After validating the deterministic corpus, run the identical cases through a
raw model or Gnomon's compiler. For Engy DeepSeek V4:

```bash
PYTHONPATH=src:. python -m benchmarks.contextbench.run_llm \
  --corpus-dir results/contextbench/corpus \
  --condition raw-llm --model deepseek-v4-flash-0731 \
  --base-url https://api.engy.ai/v1 --api-key-env ENGY_API_KEY \
  --output-dir results/contextbench/raw-llm

PYTHONPATH=src:. python -m benchmarks.contextbench.run_llm \
  --corpus-dir results/contextbench/corpus \
  --condition compiled-context --model deepseek-v4-flash-0731 \
  --base-url https://api.engy.ai/v1 --api-key-env ENGY_API_KEY \
  --output-dir results/contextbench/compiled
```

If an endpoint transport fails, rerun the same command with `--resume
--retry-errors`. Answered rows are retained and only errored or missing case
IDs execute again; ordinary `--resume` retains all terminal rows. The summary
discloses retained versus newly executed counts.

The raw arm sees history, narrative, and future-known covariates but never the
oracle. It makes two history-only calls around the contextual call. Their
absolute sMAPE difference is the model's matched stochastic noise floor;
reports show contextual lift both before and after subtracting that floor.
This is a conservative benefit check, not a causal estimator: inspect both
values, especially when the contextual call is harmful.

The compiled arm gives the same narrative to Gnomon's schema-bound,
quote-validated compiler, then runs only accepted events through the numeric
engine. Reports keep compiler event recall/false events separate from engine
admission and forecast accuracy. `compiler_event_recall` is event-weighted;
`compiler_event_recall_macro` weights event-bearing cases equally. Families
with no extractable events report recall as null rather than perfect.

Token usage is request-local so concurrent cases cannot double-count global
client deltas. On a resumed legacy run whose observations predate that
accounting version, cumulative observation usage is explicitly null; the
separate invocation usage still truthfully reports the calls made by resume.

## Surface matrix

Run the paired history/context task through a real MCP profile with
`run_surfaces`. Give every arm the same immutable compiler receipt directory;
the report excludes compiler preparation from agent-call counts and discloses
executed compiler calls separately.

```bash
PYTHONPATH=src:. python -m benchmarks.contextbench.run_surfaces \
  --corpus-dir results/contextbench/corpus \
  --profile evidence --model deepseek-v4-flash-0731 \
  --base-url https://api.engy.ai/v1 --api-key-env ENGY_API_KEY \
  --context-receipts-dir results/contextbench/receipts \
  --output-dir results/contextbench/evidence-r1
```

Use `--resume --retry-errors` only for provider or harness failures. Agent
non-submission is retained as a product failure and is never erased by retry.
The summary reports successful pairs, agent completion failures, provider
failures, and harness failures against the attempted-case denominator.

Aggregate replicated arms with repeated `--run-dir` arguments using
`report_surfaces`. Its 95% bootstrap intervals resample case IDs, not individual
replicate rows. Accuracy metrics are necessarily conditional on successful
pairs, so interpret them together with completion rate; a surface cannot make
its accuracy look better by failing hard cases.

## Interpretation

Never use pooled sMAPE alone. A covariate win cannot compensate for failure to
learn repeated events, and accuracy cannot compensate for false influence.
Accordingly the decision gates require each useful family to improve, at least
90% admission precision, at least 80% recall, under 1% observed false influence,
zero leakage, exact publication parity, exact disposition behavior, no
prior-only primary changes, and at least 70% interval coverage.

The generator is deliberately randomized across level, trend, phase, effect
sign, magnitude, event timing, onset and duration. Fresh seeds plus manifest
hashes make optimizing against a fixed fixture visible. This is internal
product evidence, not a published community benchmark.
