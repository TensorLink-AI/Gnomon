# Benchmarks

Setup (two interpreters, per-benchmark datasets and dependencies, and
what each arm costs before you start): **[SETUP.md](SETUP.md)**.
Comparing arms afterwards: `python -m benchmarks.report --root <results dir>`,
which joins arms on task id, reports matched-subset means with paired
significance tests, and refuses comparisons whose manifests disagree.

Faithful, locally runnable implementations of published time-series
reasoning benchmarks, used to measure whether Gnomon improves an agent —
and by how much — on evaluations the community already trusts.

## Ground rules

Every adapter in this directory obeys three rules:

1. **The benchmark stays authoritative.** Tasks, prompts, protocols, and
   metrics come from the official benchmark code, installed or checked
   out unmodified. Nothing here reimplements a metric that the official
   package can compute; where the official scorer runs, it scores every
   condition, including Gnomon's. Where a scorer *cannot* run over a
   condition (upstream fuses LLM calls and scoring in one script, or a
   metric module is not published), the local stand-in is disclosed in
   that benchmark's README — MTBench's treatment mirrors the official
   metric block, TemporalBench's choice questions are graded by a local
   exact match (its forecast metrics do run the dataset's own module),
   and TimeSage-MT's judge is local because the official one is not
   public.
2. **OpenRouter is the single LLM source.** Every condition that needs a
   model routes completions through OpenRouter and takes a full
   OpenRouter model id (e.g. `openai/gpt-4o`,
   `anthropic/claude-sonnet-4`), so control and treatment always use the
   same model through the same provider. Most adapters read
   `OPENROUTER_API_KEY` directly; AnomLLM's control instead points the
   official code's own `credentials.yml` mechanism at OpenRouter so the
   upstream code stays untouched (disclosed in its README). The client
   speaks plain chat-completions, so a model OpenRouter does not host
   can be evaluated through any OpenAI-compatible endpoint
   (`OPENROUTER_BASE_URL`, or `--base-url` where the adapter exposes
   it). That is provenance, not a detail: the endpoint travels into
   `summary.json`'s `llm_usage` and the run manifest, because the same
   model id served from elsewhere is a different measurement — and both
   arms of a comparison must come from the same one.
3. **Adapter decisions are disclosed.** Where Gnomon's output shape and a
   benchmark's expected input differ (e.g. quantiles vs. sample paths),
   the conversion is deterministic, documented in the module docstring,
   and applied identically across runs. Abstentions are recorded as
   abstentions — never papered over with fabricated numbers.

Each run emits the benchmark's **official results** (its own file
formats and scores — the headline numbers), and, wherever the adapter
owns the run loop, a **GnomonBench JSONL** file per condition so
`gnomon eval compare` can report the treatment/control uplift and
safety view described in
[docs/agent-evaluation.md](../docs/agent-evaluation.md). Two controls
are the exception: MTBench's and AnomLLM's control arms execute the
official scripts, which write only their own result files — for those,
cross-arm comparison goes through the official tables and
`benchmarks/report.py`'s per-task join, not `gnomon eval compare`.

## Implemented benchmarks

| Benchmark | What it measures | Adapter |
| --- | --- | --- |
| [Context is Key](cik/) (ICML 2025) | Forecasting when essential information lives in accompanying text; scored by RCRPS | `benchmarks/cik` |
| [AnomLLM](anomllm/) (ICLR 2025, "Can LLMs Understand Time Series Anomalies?") | Anomaly detection on controlled synthetic series; scored by F1 and affiliation-F1 | `benchmarks/anomllm` |
| [MTBench](mtbench/) (2025) | Temporal reasoning and QA over paired financial/weather series and news; forecasting scored by MSE/MAPE | `benchmarks/mtbench` |
| [TimeSage-MT](timesage_mt/) (2026) | Multi-turn agentic time series analysis with per-turn verifiable answers across 4 tiers | `benchmarks/timesage_mt` |
| [TemporalBench](temporalbench/) (2026) | Four-tier contextual and event-informed reasoning (T1 understanding → T4 event-conditioned prediction); forecasts scored by the dataset's own metric module, choice questions by a disclosed local exact match | `benchmarks/temporalbench` |

All five were selected because they exercise what Gnomon owns — context
admission under a leakage gate, calibrated intervals, graded detection,
tool-grounded multi-turn analysis, structured abstention — rather than
an LLM's ability to read raw number sequences. See each subdirectory's
README for setup, the exact conditions, and any faithfulness caveats
(TimeSage-MT's official judge is not public; its README explains what
is and is not comparable).

Two internally-authored benchmarks sit alongside the five published
adapters:

| Benchmark | What it measures | Adapter |
| --- | --- | --- |
| [LeakTrap](leaktrap/) (internal) | Temporal-leakage traps: whether a forecaster respects publication dates when peeking would score better; graded by a hindsight no-leak ceiling, transcription detection, and a structural snapshot assertion | `benchmarks/leaktrap` |
| [Gnomon Workflow Bench](workflow/) (internal) | End-to-end correctness, trust, usability, and token/call economics across synthetic, frozen, messy, longitudinal, and multi-series workflows | `benchmarks/workflow` |
| [ContextBench](contextbench/) (internal) | Matched context value and safety: future covariates, repeated learnable events, irrelevant context, prior-only scenarios, leakage and false influence | `benchmarks/contextbench` |

LeakTrap is ours, not a community benchmark — its numbers validate
Gnomon's bitemporal contract and are not comparable to anything
published. Its README covers the trap construction, the three arms
(control / gnomon / oracle-leak), and how to read the leak flags.
Workflow Bench is also ours. Its bundled five-case corpus validates the
evaluation contract and is a CI smoke suite, not publishable product evidence.
It ships a trading version of the same harness — session calendars, price
basis, corporate actions, halts, venue dispersion, and a return-versus-level
triage trap — as a corpus and audit profile rather than a fork, so trading and
general arms are read off one scorecard.
Its arm protocol lets raw-LLM, evidence-injection, MCP-profile, and
deterministic-runtime adapters be scored against an identical versioned case
set without exposing the oracle to the arm.

## Conditions

Adapters run matched conditions on identical task sets:

- **control** — the benchmark's own LLM baseline protocol, unmodified,
  with completions served through OpenRouter.
- **treatment** — the same model constrained to Gnomon's contract: the LLM
  proposes (context events, questions), Gnomon validates, computes, or
  abstains. Variants with no LLM at all (`gnomon-pure`, the `gnomon`
  detector) measure the harness floor.

Keep model, temperature, seeds, and sample counts identical across the
two conditions of a comparison; report the official metric alongside the
abstention and error counts, never without them.

## Environment

- `OPENROUTER_API_KEY` — required for any condition that queries a
  model. Export it, or put it in an untracked `.env` file (`KEY=value`
  lines) in the working directory or repository root — a real
  environment variable always wins over the file.
- `OPENROUTER_BASE_URL` — optional. Any OpenAI-compatible
  chat-completions endpoint; defaults to OpenRouter's own. Set it (with
  that endpoint's key in `OPENROUTER_API_KEY`) to evaluate a model
  OpenRouter does not host. The resolved endpoint is recorded with the
  results.
- Gnomon importable (`bash install.sh`, `uv tool install .`, or
  `PYTHONPATH=src` from the repository root).
- Per-benchmark dependencies are deliberately not part of Gnomon's own
  (empty) dependency set — install them per subdirectory, ideally in a
  dedicated virtualenv, since official benchmark packages can be heavy.

Run everything from the repository root with `python -m`, e.g.:

```bash
python -m benchmarks.cik.run_cik --help
python -m benchmarks.anomllm.run_anomllm --help
```

## Batch runs

`benchmarks/run_all.py` drives any subset of the adapters from one YAML
(or JSON) config: shared `model`, `defaults.temperature`,
`defaults.limit`, and per-run output dirs under `output_root` are
injected where each adapter's CLI supports them, adapter-specific
options pass through `args` verbatim, and every produced `summary.json`
is collected into `<output_root>/combined_summary.json`.

```bash
python -m benchmarks.run_all --config benchmarks/configs/example.yaml --dry-run
python -m benchmarks.run_all --config my-batch.yaml --only tb-control,tb-gnomon
python -m benchmarks.run_all --config my-batch.yaml --continue-on-error
```

See `benchmarks/configs/example.yaml` for a config covering all five
benchmarks. Datasets are not downloaded by the orchestrator — run each
adapter's `--download`/setup step once first (see the per-benchmark
READMEs).

## Comparing against published results

Where an adapter scores with the benchmark's official metric
implementation, per-run scores are directly comparable to the papers'
tables and leaderboards — but only under the official protocol, and
only for the summary key each benchmark's README designates as
published-comparable (aggregation can differ from the official one
even when per-run scores are official; CiK's README documents its
capped/imputed aggregate, and TemporalBench's choice accuracy is a
local grading whose equivalence to the leaderboard's is unverified):

- full official task set (a `--limit`/`--task-filter` smoke run is
  **not** comparable to a published number),
- official seeds and sample counts (CiK: 5 seeds; TemporalBench: all
  rows of the labeled split),
- the same metric key the paper reports.

`benchmarks/compare_published.py` renders ranked side-by-side tables
from a reference YAML into which you transcribe published numbers with
their source and retrieval date — the repo ships only a template
(`benchmarks/configs/published_reference.yaml`, with pointers to each
benchmark's leaderboard/paper tables), never third-party numbers that
could go stale:

```bash
python -m benchmarks.compare_published \
    --reference benchmarks/configs/published_reference.yaml
```

TemporalBench additionally has a public leaderboard accepting
submissions (linked from its dataset card) if a result is worth
publishing beyond a local comparison.
