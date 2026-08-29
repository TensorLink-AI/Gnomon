# Context is Key (CiK)

Adapter for [Context is Key: A Benchmark for Forecasting with Essential
Textual Information](https://arxiv.org/abs/2410.18959) (ICML 2025),
official code at
[ServiceNow/context-is-key-forecasting](https://github.com/ServiceNow/context-is-key-forecasting).

CiK's 71 tasks pair a numeric history with text (background,
constraints, scenario) that a forecaster must use to score well. The
metric is RCRPS: a region-of-interest weighted, scaled CRPS with a
penalty for violating constraints stated in the context.

## What is official, what is ours

Official (imported from `cik_benchmark`, unmodified):

- all task classes, seeds, and data (`cik_benchmark.ALL_TASKS`),
- the RCRPS metric and its scaling cache (`task.evaluate(samples)`),
- the evaluation loop (`cik_benchmark.evaluation.evaluate_all_tasks`),
- the control condition's prompt, output format, and rejection sampling
  (`cik_benchmark.baselines.direct_prompt.DirectPrompt`).

Ours (this directory):

- `openrouter_direct_prompt.py` — the official DirectPrompt with its
  model routing generalised to any OpenRouter model id. Prompting and
  parsing are inherited, not overridden.
- `gnomon_forecaster.py` — the treatment. Disclosed adapter decisions:
  RCRPS needs sample paths, so each lead receives a deterministic stratified
  marginal through q10/q50/q90, with linearly extrapolated tails and a
  lead-specific stratum permutation. This removes the former clamping and
  comonotonicity advantage, while remaining a disclosed three-quantile
  approximation rather than a learned joint distribution. Other
  decisions: CiK's timezone-naive indexes are written as UTC;
  LLM-proposed events carry a verifiable `dataset` source referencing
  the task's own context text, with `known_at` at the history start (the
  text is benchmark-supplied ground context, not a dated publication).
  Gnomon's admission gate still decides whether any event influences the
  forecast. When Gnomon abstains, the run is recorded as an abstention —
  no samples are fabricated.

## Conditions

| Condition | LLM | Numbers produced by |
| --- | --- | --- |
| `control` | official DirectPrompt via OpenRouter | the LLM |
| `gnomon-pure` | none (context ignored) | Gnomon |
| `gnomon-agent` | proposes typed context events only | Gnomon |
| `gnomon-conditional` | proposes typed events; prospective effects may alter only a labelled conditional path | Gnomon |
| `gnomon-mcp` | holds Gnomon's real MCP tools, uses them or not | Gnomon (verbatim artifact) or the LLM, labeled per run |

`gnomon-mcp` is the integrated "agent chooses" arm
(`docs/design/cik-mcp-tool-arm.md`, `mcp_agent.py`): every tool
`gnomon mcp serve` publishes is handed to the model verbatim, results
(including typed errors with repair options) come back unedited, and
the run ends with `submit_forecast` — either an `artifact_path` whose
trajectory is used byte-for-byte, or the model's own per-step
quantiles. The route is classified from the transcript afterwards
(`gnomon` / `direct` / `informed-direct`), caps (10 rounds, 24 calls,
250k tokens, 7,200 s) abstain rather than fall back, and a path jail
keeps the model away from the cached benchmark datasets. Per-run
transcripts land in `<output-dir>/mcp-traces/`.

On any arm, an optional-tool win is evidence about the *pipeline*,
never about Gnomon's own forecasting quality — it can come entirely
from knowing when not to call.

For diagnostics, `--mcp-output-role canonical` scores the public artifact
trajectory, which may already be context-conditioned.
`--mcp-output-role immutable_primary` explicitly ignores that recommendation
and scores the preserved primary. Only the latter is a valid within-run
context-uplift comparator; it is not a product output.

`gnomon-conditional` is the stable, manifest-visible form of the conditional
arm. It enables Gnomon's `context.future_events` lane while retaining the
unmodified primary path in the same artifact. `gnomon-agent --future-context`
remains as a compatibility spelling. In either form,
Gnomon's `context.future_events` lane: the proposer may also quote
verbatim `source_span`s for stated bounds (`constraint:*`) and stated
deterministic windows (`override:*`). The adapter verifies each span is a
verbatim quote of the task context before Gnomon sees it; Gnomon then
re-parses the numbers deterministically and applies its own admission
checks. Influenced runs report support `context_trusted`. The A/B for
this flag is pre-registered in `results/future-context-ab/HYPOTHESIS.md`.

## Setup

```bash
python -m venv .venv-cik && source .venv-cik/bin/activate
pip install -r benchmarks/cik/requirements.txt   # heavy: official deps
pip install -e .                                 # gnomon
export ENGY_API_KEY=...

# One-time: build the official metric scaling cache (downloads task data).
python -c "import runpy; runpy.run_path('precompute_scaling_cache.py')" \
  # ... run from your cik checkout, or see the official README
```

The official package downloads task datasets on first use; set
`CIK_DATA_STORE`, `HF_HOME`, `CIK_RESULT_CACHE`, and
`CIK_METRIC_SCALING_CACHE` to control where everything lands. Without
the scaling cache the official metric returns NaN — build it first.

## Run

```bash
# Control: official LLM protocol through the default Engy endpoint
python -m benchmarks.cik.run_cik --method control \
    --model openai/gpt-4o --output-dir results/cik-gpt4o-control

# Treatment: same model, numbers owned by Gnomon
python -m benchmarks.cik.run_cik --method gnomon-agent \
    --model openai/gpt-4o --output-dir results/cik-gpt4o-gnomon

# Conditional context: same primary plus a separately labelled scenario
python -m benchmarks.cik.run_cik --method gnomon-conditional \
    --model openai/gpt-4o --output-dir results/cik-gpt4o-conditional

# Harness floor: no LLM anywhere
python -m benchmarks.cik.run_cik --method gnomon-pure \
    --output-dir results/cik-gnomon-pure

# Integrated arm: real MCP tools, the model chooses
python -m benchmarks.cik.run_cik --method gnomon-mcp \
    --model openai/gpt-4o --output-dir results/cik-gpt4o-mcp

# Production-style governed agent: the host compiles the supplied context into
# a provenance receipt, injects validated typed events into the model's one
# Gnomon call, and binds the first valid artifact. Model-authored numeric
# fallback is disabled.
python -m benchmarks.cik.run_cik --method gnomon-mcp \
    --mcp-profile evidence --model openai/gpt-4o \
    --output-dir results/cik-gpt4o-mcp-evidence

# Diagnostic only: score the sealed LLM candidate rather than the canonical
# artifact. It remains prior_assisted, non-automatable, and cannot replace the
# product answer. Ordinary canonical runs also retain this exact candidate in
# extra_info for matched post-hoc scoring without a second stochastic call.
python -m benchmarks.cik.run_cik --method gnomon-mcp \
    --mcp-profile evidence --mcp-output-role llm_candidate_shadow \
    --model openai/gpt-4o \
    --output-dir results/cik-gpt4o-mcp-candidate-shadow

# Product best-effort mode: score the candidate only after the publication
# verifier preserves the primary, support label, and automation separation.
python -m benchmarks.cik.run_cik --method gnomon-mcp \
    --mcp-profile evidence --mcp-output-role publication_best_effort \
    --model deepseek-v4-flash-0731 \
    --output-dir results/cik-deepseek-best-effort

# Controlled candidate-compute/evidence ablations. These change only the
# host's prior-assisted, non-automatable candidate lane; the immutable primary
# and support contract are unchanged. Defaults are 4-5 paths, 64 raw rows,
# and compact full-history temporal facts enabled.
python -m benchmarks.cik.run_cik --method gnomon-mcp \
    --mcp-profile evidence --mcp-output-role publication_best_effort \
    --mcp-candidate-paths 10 --mcp-candidate-history-rows 64 \
    --no-mcp-candidate-temporal-facts \
    --model deepseek-v4-flash-0731 \
    --output-dir results/cik-candidate-ablation

# Quick pass on a task family while iterating
python -m benchmarks.cik.run_cik --method gnomon-pure \
    --task-filter sensor --seeds 1 --output-dir /tmp/cik-smoke

# Preregistered held-out task instances: seed 6 only, never silently mixed
# with the development seeds recorded in earlier output directories.
python -m benchmarks.cik.run_cik --method gnomon-mcp \
    --mcp-profile evidence --mcp-output-role publication_best_effort \
    --model deepseek-v4-flash-0731 --seed-start 6 --seeds 1 \
    --output-dir results/cik-heldout-seed6-gnomon
```

CiK tasks always execute sequentially in disposable child processes. The
runner measures resident memory across each entire process tree, terminates a
case above `--case-memory-mb` (4 GiB by default), terminates it after
`--case-timeout-seconds` (15 minutes), and refuses to start another case below
`--min-free-memory-mb` (2 GiB). It writes `case-checkpoint.json` atomically
after every task/seed and resumes it by default. A killed case is retained as
an explicit error and receives the usual capped/imputed score; it cannot vanish
from the aggregate. `--max-parallel` must remain `1`; shard across separate
machines and output directories when parallel execution is required.

Outputs per run: `summary.json`, `scores.csv` (official
per-task-per-seed scores), `runs/` (the official per-run artifacts:
forecast plots, contexts, metric details), `manifest.json` (provenance:
condition, model, seeds, command), and `gnomonbench.jsonl` for:

```bash
gnomon eval compare \
    --baseline results/cik-gpt4o-control/gnomonbench.jsonl \
    --treatment results/cik-gpt4o-gnomon/gnomonbench.jsonl
```

## Aggregation

`summary.json` reports two aggregates; they answer different questions
and only one is publishable:

- `mean_rcrps_capped_imputed` — per-run RCRPS capped at 5.0, and every
  abstained or errored (task, seed) run imputed at 5.0. This mirrors the
  official aggregation rule (`compile_roi_results.py` upstream caps at
  `CAP = 5` and pads missing runs at the cap), so abstaining can never
  improve it. This is the key to put next to published means.
- `mean_rcrps_scored_only` — the uncapped mean over scored runs only.
  Abstentions and errors simply drop out, so a system that abstains on
  its hardest tasks looks better on this number. Never quote it without
  the `runs_abstained` / `runs_errored` counts beside it.

Both local aggregates are unweighted over runs: the official per-task
weights used in the paper's weighted rankings are not reproduced here.
The per-run scores in `scores.csv` are the official metric, computed by
the official code.

## Notes

- Some CiK tasks use frequencies outside Gnomon's supported grid; Gnomon
  abstains on those, and the abstention shows up in `summary.json`
  rather than as a silent skip.
- `gnomon-mcp --mcp-profile evidence` is the governed product arm. The agent
  chooses whether to invoke Gnomon. Before that loop, the host compiles the
  benchmark-supplied prose into typed events and an auditable receipt containing
  the source hash, compiler identity, rejected proposals, and an explicit
  declaration that future target observations were not exposed. The host binds
  those events, the task data, and the horizon to `gnomon_forecast`; once it
  produces a valid artifact the host publishes it verbatim. This avoids testing
  event-schema recall or path copying. Because CiK's requested verb is known,
  this arm exposes only `gnomon_forecast` rather than inviting a descriptive
  detour. `informed-direct` is structurally unavailable in the governed arm.
  Other profiles retain the direct/informed-direct exits as explicitly labelled
  autonomy experiments.

  This production-style arm stamps the assembled context as known at the
  forecast cutoff, when the host actually received it. It therefore cannot use
  a retrospective task description inside earlier backtest folds. The legacy
  `gnomon-agent` arm retains its documented benchmark assumption that the task
  context was known from the history start, so old experiments remain
  comparable rather than silently changing meaning.

  The compiler also emits a richer cited dossier—qualitative temporal claims,
  timing, mechanism, direction, uncertainty, and an optional probabilistic
  forecast candidate. Deterministic validation rejects uncited claims,
  malformed or misaligned quantiles, and grossly implausible paths. A surviving
  candidate is sealed as `prior_assisted`, never automation-eligible, and kept
  beside the immutable canonical artifact. It can be shadow-scored now and
  upgraded only by historical replay or realised outcomes; the LLM's confidence
  cannot grant publication authority.
- `--n-samples` defaults to the official `DEFAULT_N_SAMPLES`; change it
  only symmetrically across conditions.
- `--fail-on-invalid` (control only) defaults to `True`, the official
  `DirectPrompt` default: a run errors when rejection sampling cannot
  collect `n_samples` valid forecasts. `--no-fail-on-invalid` scores
  such runs on however many valid forecasts were collected — a protocol
  deviation (fewer samples than the official runs use for that
  instance), disclosed in the method's cache name; report it whenever
  you set it.
- The official result cache (`CIK_RESULT_CACHE`) makes reruns free;
  `--no-cache` forces fresh inference.
