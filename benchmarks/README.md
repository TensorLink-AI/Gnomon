# Benchmarks

Faithful, locally runnable implementations of published time-series
reasoning benchmarks, used to measure whether Aion improves an agent —
and by how much — on evaluations the community already trusts.

## Ground rules

Every adapter in this directory obeys three rules:

1. **The benchmark stays authoritative.** Tasks, prompts, protocols, and
   metrics come from the official benchmark code, installed or checked
   out unmodified. Nothing here reimplements a metric that the official
   package can compute; where the official scorer runs, it scores every
   condition, including Aion's.
2. **OpenRouter is the single LLM source.** Every condition that needs a
   model reads `OPENROUTER_API_KEY` and takes a full OpenRouter model id
   (e.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4`), so control and
   treatment always use the same model through the same provider.
3. **Adapter decisions are disclosed.** Where Aion's output shape and a
   benchmark's expected input differ (e.g. quantiles vs. sample paths),
   the conversion is deterministic, documented in the module docstring,
   and applied identically across runs. Abstentions are recorded as
   abstentions — never papered over with fabricated numbers.

Each run emits two artifacts: the benchmark's **official results** (its
own file formats and scores — the headline numbers), and an
**AionBench JSONL** file per condition so `aion eval compare` can report
the treatment/control uplift and safety view described in
[docs/agent-evaluation.md](../docs/agent-evaluation.md).

## Implemented benchmarks

| Benchmark | What it measures | Adapter |
| --- | --- | --- |
| [Context is Key](cik/) (ICML 2025) | Forecasting when essential information lives in accompanying text; scored by RCRPS | `benchmarks/cik` |
| [AnomLLM](anomllm/) (ICLR 2025, "Can LLMs Understand Time Series Anomalies?") | Anomaly detection on controlled synthetic series; scored by F1 and affiliation-F1 | `benchmarks/anomllm` |
| [MTBench](mtbench/) (2025) | Temporal reasoning and QA over paired financial/weather series and news; forecasting scored by MSE/MAPE | `benchmarks/mtbench` |
| [TimeSage-MT](timesage_mt/) (2026) | Multi-turn agentic time series analysis with per-turn verifiable answers across 4 tiers | `benchmarks/timesage_mt` |
| [TemporalBench](temporalbench/) (2026) | Four-tier contextual and event-informed reasoning (T1 understanding → T4 event-conditioned prediction); scored by the dataset's own metric module | `benchmarks/temporalbench` |

All five were selected because they exercise what Aion owns — context
admission under a leakage gate, calibrated intervals, graded detection,
tool-grounded multi-turn analysis, structured abstention — rather than
an LLM's ability to read raw number sequences. See each subdirectory's
README for setup, the exact conditions, and any faithfulness caveats
(TimeSage-MT's official judge is not public; its README explains what
is and is not comparable).

## Conditions

Adapters run matched conditions on identical task sets:

- **control** — the benchmark's own LLM baseline protocol, unmodified,
  with completions served through OpenRouter.
- **treatment** — the same model constrained to Aion's contract: the LLM
  proposes (context events, questions), Aion validates, computes, or
  abstains. Variants with no LLM at all (`aion-pure`, the `aion`
  detector) measure the harness floor.

Keep model, temperature, seeds, and sample counts identical across the
two conditions of a comparison; report the official metric alongside the
abstention and error counts, never without them.

## Environment

- `OPENROUTER_API_KEY` — required for any condition that queries a model.
- Aion importable (`bash install.sh`, `uv tool install .`, or
  `PYTHONPATH=src` from the repository root).
- Per-benchmark dependencies are deliberately not part of Aion's own
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
python -m benchmarks.run_all --config my-batch.yaml --only tb-control,tb-aion
python -m benchmarks.run_all --config my-batch.yaml --continue-on-error
```

See `benchmarks/configs/example.yaml` for a config covering all five
benchmarks. Datasets are not downloaded by the orchestrator — run each
adapter's `--download`/setup step once first (see the per-benchmark
READMEs).

## Comparing against published results

Because every adapter scores with the benchmark's official metric
implementation, our summaries are directly comparable to the papers'
tables and leaderboards — under the official protocol only:

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
