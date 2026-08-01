# TimeSage-MT — Multi-Turn Agentic Time Series Reasoning

Adapter for [TimeSage-MT](https://arxiv.org/abs/2606.01498): 240
multi-turn tasks (~2,680 turns) across 4 difficulty tiers and 8 domains,
built from real series with per-turn verifiable answers. Official
dataset: [Timesage/TimeSage-MT](https://hf.co/datasets/Timesage/TimeSage-MT)
(Apache-2.0).

## What is official, what is ours

Official (from the dataset, used as published):

- every task file (`MT_Bench/L1..L4/*.json`): dialogues, visibility
  contracts, reference turns, and the per-turn `finding_verify`
  scoring specs (keyword sets, numerical ranges, judge rubrics),
- the per-task visible series (`visible_ts/<tier>/<task>/agent_input/`),
  which never exposes rows beyond the visibility contract.

Ours (this directory):

- `harness.py` — dialogue replay. `direct` mirrors the paper's Direct
  Answering baseline (visible CSV + conversation, no tools);
  `aion-tools` gives the same model a function-calling loop over
  deterministic tools (summary stats, Aion season detection, Aion
  backtested forecasting, Aion graded anomaly detection) with a
  system-prompt contract that every quoted number come from a tool.
  Reference agent turns are never shown to the agent.
- `scoring.py` — applies the dataset-embedded `finding_verify` specs:
  `keyword` (all keywords present) and `numerical_range` (some number
  in the response inside the range) are scored mechanically, exactly
  as specified. Specs needing an embedding or judge are **not**
  silently approximated: they count as unscored unless you pass
  `--judge-model`, and judge scores are reported separately.

**Faithfulness caveat, stated plainly:** the official platform's own
scorer/judge and leaderboard harness are not published as importable
code (results go through their dashboard). Mechanical scores here follow
the official task files' verify specs verbatim; judge-based scores use a
local LLM judge and are *not* leaderboard-comparable. Treat cross-
condition deltas (same scorer both sides) as the meaningful number.

## Setup and run

```bash
pip install huggingface_hub   # only extra dependency
export OPENROUTER_API_KEY=sk-or-...

python -m benchmarks.timesage_mt.run_timesage --download --data-dir ~/timesage-mt

# Control vs treatment, same model, same tasks
python -m benchmarks.timesage_mt.run_timesage \
    --data-dir ~/timesage-mt --condition direct \
    --model openai/gpt-4o --tiers L1,L2 --output-dir results/ts-direct

python -m benchmarks.timesage_mt.run_timesage \
    --data-dir ~/timesage-mt --condition aion-tools \
    --model openai/gpt-4o --tiers L1,L2 --output-dir results/ts-aion

aion eval compare \
    --baseline results/ts-direct/aionbench.jsonl \
    --treatment results/ts-aion/aionbench.jsonl
```

Outputs per run: `transcripts/<task>.json` (every turn, tool calls, and
verdicts), `scores.csv`, `summary.json` (mechanical pass rate, judge
pass rate if enabled, unscored count, per-tier breakdown), and
`aionbench.jsonl` (one row per scored turn).

Start with `--tiers L1 --limit 10` to gauge cost; L3/L4 dialogues are
long and tool loops multiply requests.
