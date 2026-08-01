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
- `aion_forecaster.py` — the treatment. Disclosed adapter decisions:
  RCRPS needs sample paths, so samples are drawn deterministically from
  the piecewise-linear inverse CDF through Aion's q10/q50/q90 with tails
  clamped at the outer quantiles; CiK's timezone-naive indexes are
  written as UTC; LLM-proposed events carry a verifiable `dataset`
  source referencing the task's own context text, with `known_at` at the
  history start (the text is benchmark-supplied ground context, not a
  dated publication). Aion's admission gate still decides whether any
  event influences the forecast. When Aion abstains, the run is recorded
  as an abstention — no samples are fabricated.

## Conditions

| Condition | LLM | Numbers produced by |
| --- | --- | --- |
| `control` | official DirectPrompt via OpenRouter | the LLM |
| `aion-pure` | none (context ignored) | Aion |
| `aion-agent` | proposes typed context events only | Aion |

## Setup

```bash
python -m venv .venv-cik && source .venv-cik/bin/activate
pip install -r benchmarks/cik/requirements.txt   # heavy: official deps
pip install -e .                                 # aion
export OPENROUTER_API_KEY=sk-or-...

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
# Control: official LLM baseline, any OpenRouter model
python -m benchmarks.cik.run_cik --method control \
    --model openai/gpt-4o --output-dir results/cik-gpt4o-control

# Treatment: same model, numbers owned by Aion
python -m benchmarks.cik.run_cik --method aion-agent \
    --model openai/gpt-4o --output-dir results/cik-gpt4o-aion

# Harness floor: no LLM anywhere
python -m benchmarks.cik.run_cik --method aion-pure \
    --output-dir results/cik-aion-pure

# Quick pass on a task family while iterating
python -m benchmarks.cik.run_cik --method aion-pure \
    --task-filter sensor --seeds 1 --output-dir /tmp/cik-smoke
```

Outputs per run: `summary.json` (mean RCRPS over scored runs, plus
abstention/error counts — always report them together), `scores.csv`
(official per-task-per-seed scores), `runs/` (the official per-run
artifacts: forecast plots, contexts, metric details), and
`aionbench.jsonl` for:

```bash
aion eval compare \
    --baseline results/cik-gpt4o-control/aionbench.jsonl \
    --treatment results/cik-gpt4o-aion/aionbench.jsonl
```

## Notes

- Some CiK tasks use frequencies outside Aion's supported grid; Aion
  abstains on those, and the abstention shows up in `summary.json`
  rather than as a silent skip.
- `--n-samples` defaults to the official `DEFAULT_N_SAMPLES`; change it
  only symmetrically across conditions.
- The official result cache (`CIK_RESULT_CACHE`) makes reruns free;
  `--no-cache` forces fresh inference.
