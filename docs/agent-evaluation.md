# Evaluating whether Gnomon improves an agent

GnomonBench compares the same agent and model on the same task IDs under two
conditions: a control without Gnomon and a treatment with Gnomon. Keep the system
prompt, model, temperature, token budget, data, and grader identical. Run each
task repeatedly when measuring a nondeterministic agent.

Each JSONL row is one graded run:

```json
{"task_id":"capacity-001","success":true,"temporal_leakage":false,"invented_number":false,"warning_omission":false,"appropriate_abstention":false,"tool_calls":3,"latency_seconds":4.1,"cost_usd":0.01}
```

Required fields are nonempty string `task_id` and boolean `success`. Files must
contain identical unique task IDs, including tasks that failed, hit a budget or
never submitted an answer. Headline delivered success uses **all scheduled tasks**.
`row_abstained`/`voided` or an explicit error/noncompletion overrides a contradictory
successful grade. This is operational non-delivery, not a claim of an incorrect answer.

Optional cost, latency, tool-use and safety fields are unknown when missing/null,
never zero or false. A checked clean safety grade writes explicit `false`. Safety
deltas use only exact task pairs with explicit grades in both arms; coverage and
IDs accompany them. Costs include unfinished attempts. Resource `observed_total`
sums known entries; `total` stays null until every task is measured. Means are over
measured entries, with their denominators disclosed. Finite nonnegative numbers
are required; tool/token counts are integers. JSON duplicates and nonfinite numbers
are rejected. Files allow up to 10,000 tasks, 1 MiB per physical JSONL row and
128 JSON collection levels. These are input bounds, not a global memory guarantee.

Use `completed`, `error`, `budget_exceeded` and `abstained` to record distinct
operational outcomes. A `row_abstained` reason beginning `cap:` identifies a budget
termination. `status="abstained"` is a submitted refusal, unlike a harness cutoff.
`appropriate_abstention` is a separate explicit grade. Completion is unknown when
an unsuccessful legacy row carries no completion evidence; report all-task bounds
alongside the measured-only completion rate. Do not infer error-free execution
from a missing error field. Optional `accuracy` in [0,1] is a conditional-quality
diagnostic, never the all-task headline.

```bash
gnomon eval compare \
  --baseline results/agent-control.jsonl \
  --treatment results/agent-gnomon.jsonl
```

The output reports absolute task-success uplift, relative error reduction,
safety deltas, average tool calls, latency, and cost. `examples/gnomonbench/`
contains format demonstrations only; they are synthetic and must not be quoted
as measured Gnomon performance.

Schema `0.2` supersedes the old answer-only `0.1` comparison semantics. The exact
paired McNemar test now includes all tasks; `conditional_completed_pairs` remains
a labelled, selection-sensitive diagnostic. Its p-value assumes independent task
pairs and has no multiple-comparison adjustment. A matched ID alone does not prove
identical prompts, graders or task contents: verify the run manifests separately.
Known incompatible per-task `success_basis` values are refused. Keep task families
with different grading semantics separate when making substantive claims.

Optional `success_probability` in [0,1] means probability of delivered task success,
not forecast interval coverage. It produces a Brier score and fixed ten-bin
reliability summaries with counts and IDs; absent probabilities remain unmeasured.
Gnomon cannot attest that supplied probabilities were recorded before outcomes.
These are descriptive diagnostics, not calibration certification. A Brier score
alone mixes calibration and discrimination, so lower is not proof of better
calibration. [Calibration reference](https://scikit-learn.org/stable/modules/calibration.html).
The paired binary test uses the symmetric exact binomial null on discordant pairs.
[Binomial test reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

Recommended initial task families are inventory decisions, capacity planning,
event-aware demand, anomaly investigation, unsupported-data abstention, and
temporal-leakage traps. Grade final decisions programmatically wherever
possible. Use an LLM judge only for explanation quality, separately from the
primary task-success score.

The historical `benchmarks.report` importer uses these same all-observed-shared-task
success and resource metrics when all rows carry binary grades. It cannot reconstruct
an unrecorded scheduled task: unmatched IDs and unverified cohort status are explicit.
Malformed rows and normalized duplicate IDs now fail import rather than disappearing.
Continuous scores remain conditional on paired observations. Baseline-score imputation
is only a hypothetical sensitivity analysis, not observed fallback quality or a bound.
Legacy Workflow Bench promotion refuses incomplete or mismatched case sets; its raw
all-case summary remains available for noncompletion analysis. Neither historical
workflow is evidence about the current default unless its actual configuration matches.

Workflow Bench now preserves started/finished attempt receipts through retries,
stages and checkpoint resume. Its append-only `observations.attempts.sqlite3`
journal records a start before launching a process. A crash or timeout does not
turn unobserved provider spend into zero. `resource_accounting` distinguishes
observed lower bounds, complete totals and missing measurements; cost gates refuse
unverified call/token totals. Legacy scalar economics remain compatibility lower
bounds. Historical observations with lost attempt records stay incomplete.

The shared LLM client also distinguishes known from missing provider usage. Cached
request history is not charged again as new subprocess usage. Where aggregate
provider-cache history cannot be matched to the outer attempt journal, completeness
remains false: do not claim end-to-end cost improvement from those totals. Billing
is provider-supplied evidence, not an independently audited invoice. See
[Workflow Bench accounting](../benchmarks/workflow/README.md#attempt-accounting).

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

## Advanced legacy agent lifecycle

The following names/arguments belong to explicit legacy profiles. Ordinary sessions
use execution IDs and `gnomon_ledger`; see [current configuration and ledger operations](production/INFERENCE.md).

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
