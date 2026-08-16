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

Recurring-event admission also uses displaced copies of the observed event
schedule as negative controls. This prevents densely overlapping validation
windows from turning an accidental alignment between model residuals and a
calendar into apparently independent evidence. These controls use historical
data only; the sealed outcome remains untouched by selection.

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

A product decision requires independent fresh corpora, not repeated sampling
of one fixture. Aggregate at least three completed runs with distinct manifest
hashes:

```bash
PYTHONPATH=src:. python -m benchmarks.contextbench.report_contextbench \
  --run-dir results/contextbench/fresh-r1 \
  --run-dir results/contextbench/fresh-r2 \
  --run-dir results/contextbench/fresh-r3 \
  --output results/contextbench/replicated-report.json
```

The pooled report preserves per-replicate failures while applying rate gates to
the denominator at which they are numerically meaningful. In particular, one
false influence among 40 negative cases is 2.5%; a sub-1% gate cannot be
resolved from a single 80-case corpus.

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

There are two distinct treatments; do not pool them:

- `compiled` is the production treatment: the host resolves explicit forecast
  intent to the forecast verb and binds the already-known schema, while the
  selected profile remains visible for submission and recovery.
- `unrouted` is a diagnostic: it requires Gnomon-authored publication but
  leaves initial verb selection to the agent. It measures navigation failure,
  not the intended product path, and cannot win the default-surface decision.

Successful forecasts should be identical across surfaces. The aggregate report
checks that parity explicitly. A numerical difference is a contract failure,
not evidence that one menu forecasts better.

```bash
PYTHONPATH=src:. python -m benchmarks.contextbench.run_surfaces \
  --corpus-dir results/contextbench/corpus \
  --profile evidence --model deepseek-v4-flash-0731 \
  --base-url https://api.engy.ai/v1 --api-key-env ENGY_API_KEY \
  --context-receipts-dir results/contextbench/receipts \
  --routing-policy compiled --baseline-mode engine --replicate-id 1 \
  --output-dir results/contextbench/evidence-r1
```

Provider and harness failures receive two bounded in-run retries by default;
set `--infrastructure-retries` explicitly for a decision run. Use `--resume
--retry-errors` to repair any remaining infrastructure rows later. Agent
non-submission is retained as a product failure and is never erased by retry.
`observations.jsonl` contains the latest canonical row per case, while the
append-only `attempts.jsonl` preserves every failed and successful execution so
retry cost cannot disappear. The summary reports successful pairs, agent
completion failures, provider failures, and harness failures against the
attempted-case denominator, with cumulative usage over all attempts.
Provider exhaustion during the final submit-only call remains a provider
failure; it is never converted into an agent abstention.
Each MCP server call also has a hard 120-second ceiling by default
(`--tool-timeout`), replacing the generic harness's deliberately generous
ten-minute ceiling; a wedged product call therefore becomes a staged harness
failure that can be retried and resumed rather than blocking the matrix.
Decision runs default to `--baseline-mode engine`: the matched history-only
forecast is computed directly by Gnomon and the LLM is spent only on the
context-enabled surface being evaluated. `--baseline-mode agent` retains the
legacy two-conversation diagnostic when the history-only interaction itself is
the subject of the experiment. Compiled prompts do not repeat the numeric
history already bound to the host-generated tool call; unrouted prompts do.
Each row records elapsed time by stage (`history_engine` or `history_agent`,
`context_compiler`, and `context_agent`), so a timeout is attributable rather
than a single opaque case failure. Run profiles sequentially and tune `--jobs` per endpoint;
parallel profile launches can measure provider saturation instead of the
product surface.

Aggregate replicated arms with repeated `--run-dir` arguments using
`report_surfaces`. Its 95% bootstrap intervals resample case IDs, not individual
replicate rows. Accuracy metrics are necessarily conditional on successful
pairs, so interpret them together with completion rate; a surface cannot make
its accuracy look better by failing hard cases.

For a decision run, prepare compiler receipts once and run three compiled-route
replicates for `core`, `describe`, `evidence`, `mega`, and `full`. Include an
unrouted diagnostic only when measuring the cost of omitting the intent
compiler. Reusing receipts keeps context extraction constant across profiles.
Context usefulness comes from the deterministic and compiled-context arms; the
surface matrix decides how cheaply and reliably an agent reaches that same
governed answer.

Treat the evaluation as three separate layers. `run_contextbench` measures the
engine and admission policy without an LLM. `run_surfaces --routing-policy
compiled` measures the production compiler-to-execution contract. The
`unrouted` policy measures agent navigation only. Reports name receipt
generation, receipt reuse, observed agent calls, required calls, redundant
calls, and prompt/completion tokens separately; cached receipt metadata is not
reported as cost incurred by the current run.

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
