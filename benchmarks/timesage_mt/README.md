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
  `gnomon-tools` gives the same model a function-calling loop over
  deterministic tools (summary stats, Gnomon season detection, Gnomon
  backtested forecasting, Gnomon graded anomaly detection) with a
  system-prompt contract that every quoted number come from a tool.
  Reference agent turns are never shown to the agent.
- `scoring.py` — applies the dataset-embedded `finding_verify` specs.
  The mechanical rule, precisely: listed `keywords` are always checked
  (all must appear, case-insensitive), whatever the spec's `type` says;
  a numerical `range` is checked only for `numerical_range`/`keyword`/
  untyped specs (some number in the response inside the inclusive
  range). Any spec where at least one of those checks applies is graded
  mechanically on those parts alone — e.g. an `embedding_threshold`
  spec that also lists keywords is graded on the keywords only, its
  threshold ignored. Only specs where neither applies count as unscored
  unless you pass `--judge-model`, and judge scores are reported
  separately.

**Range-check leniency, stated plainly:** numbers inside date/timestamp
tokens (`2026-01-02`, `12:30:05`, ISO datetimes) are excluded from
number extraction, identically for both arms — but ANY remaining
in-range number anywhere in the response passes, and the tools arm's
prompt elicits more numbers per response, so the check's expected
benefit scales with response numerosity. Summaries therefore report the raw
mechanical rate and a `numerosity_robust` subset containing at most one numeric
candidate; neither is presented as the unpublished official judge score.
Whether the official judge
shares this leniency is not knowable from the published dataset.

**Context-bound confound, disclosed:** both conditions' system prompts
embed at most 60,000 characters of the visible CSV, but the tools
compute over the full visible series — on tasks whose CSV exceeds that
bound, the tools arm has data access the direct arm lacks, on top of
the intended treatment. The mechanism is deliberate (tools scaling past
the context window is part of what tools buy), and it is measured:
every per-turn record carries `csv_truncated` and `summary.json`
reports `tasks_csv_truncated`.

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
    --model openai/gpt-4o --tiers L1,L2 --workers 4 --timeout 180 \
    --output-dir results/ts-direct

python -m benchmarks.timesage_mt.run_timesage \
    --data-dir ~/timesage-mt --condition gnomon-tools \
    --model openai/gpt-4o --tiers L1,L2 --workers 4 --timeout 180 \
    --output-dir results/ts-gnomon

gnomon eval compare \
    --baseline results/ts-direct/gnomonbench.jsonl \
    --treatment results/ts-gnomon/gnomonbench.jsonl
```

Outputs per run: `transcripts/<task>.json` (every turn, tool calls, and
verdicts), `scores.csv`, `summary.json` (mechanical pass rate, judge
pass rate if enabled, unscored count, `tasks_failed` and
`tasks_csv_truncated` counts, per-tier breakdown), and
`gnomonbench.jsonl` (one row per scored turn; a crashed task still
emits a failed row per reference-scored turn, plus one
`<task>-error` line).

Start with `--tiers L1 --limit 10` to gauge cost; L3/L4 dialogues are
long and tool loops multiply requests. `--limit` is stratified: it takes
tasks round-robin across the requested tiers (extra slots to earlier
tiers), so a limited run samples every tier it asked for instead of
silently reducing to the earliest. Limited runs are still declared
non-comparable.

Task dialogues are independent, so `--workers` parallelizes tasks without
changing the conversation inside a task. Each completed transcript contains
its own model, endpoint, usage, and elapsed-time provenance and is written
immediately. If a provider outage interrupts a run, repeat the same command
with `--resume`; only complete transcripts matching the requested condition
and model are reused, while missing or legacy transcripts are rerun. Usage is
then reconstructed by summing those per-task receipts rather than guessed from
the surviving process. `--timeout` bounds each individual API request.
