# TemporalBench — Contextual and Event-Informed Time Series Tasks

Adapter for [TemporalBench](https://arxiv.org/abs/2602.13272) (Weng,
Cao, Yang, Sharma, Liu — the USC Melady group). Official dataset:
[Melady/TemporalBench](https://huggingface.co/datasets/Melady/TemporalBench)
(Apache-2.0), with a public
[leaderboard](https://huggingface.co/spaces/Melady/TemporalBench_Leaderboard).

Four task families over retail / healthcare / energy / physical-systems
series: **T1** historical understanding (MCQ), **T2** context-free
forecasting, **T3** contextual reasoning (MCQ), **T4** event-informed
prediction — the closest public match to Gnomon's context-event contract.

## What is official, what is ours

Official (from the dataset, used as published):

- every task row of the labeled split
  (`task_merged_dev_with_labels_tiers.jsonl`): the complete prompts,
  inputs, labels, ground-truth futures, and MCQ options,
- **all forecast metrics**: the dataset ships
  `forecast_metrics_utils.py` (the authors' reference implementation,
  including the weighted OW metrics for multi-channel MIMIC rows); this
  adapter imports that file unmodified and scores every condition with
  it. Choice questions are exact-match against the embedded labels.

Ours (this directory) — the three conditions:

| Condition | LLM | Numbers produced by |
| --- | --- | --- |
| `control` | official prompt verbatim via OpenRouter | the LLM |
| `gnomon-pure` (T2/T4) | none | Gnomon; MCQs answered `Uncertain` |
| `gnomon-agent` | answers choice questions given Gnomon's evidence | Gnomon — forecast arrays in the final answer are Gnomon's, not editable by the model |

Disclosed adapter decisions (see `gnomon_runner.py`): rows carry
index-aligned arrays rather than regular timestamps, so Gnomon models each
channel on a synthetic regular hourly axis (index-based metrics — the
axis never enters the score); nulls go through Gnomon's disclosed repair
layer; channels Gnomon abstains on stay absent and the row is recorded as
an abstention; `gnomon-pure` answers MCQs with the option sets' own
`Uncertain` — an honest abstention, reported as such.

## Setup and run

```bash
pip install huggingface_hub numpy   # numpy is required by the official metrics
export OPENROUTER_API_KEY=sk-or-...

python -m benchmarks.temporalbench.run_temporalbench --download \
    --data-dir ~/temporalbench

# Control vs treatment on the forecasting tiers (start small):
python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition control \
    --model openai/gpt-4o --tiers T2,T4 --limit 50 \
    --output-dir results/tb-control

python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition gnomon-agent \
    --model openai/gpt-4o --tiers T2,T4 --limit 50 \
    --output-dir results/tb-gnomon

# The no-LLM floor (free):
python -m benchmarks.temporalbench.run_temporalbench \
    --data-dir ~/temporalbench --condition gnomon-pure \
    --tiers T2,T4 --limit 50 --output-dir results/tb-pure

gnomon eval compare --baseline results/tb-control/gnomonbench.jsonl \
                  --treatment results/tb-gnomon/gnomonbench.jsonl
```

`--datasets` filters by source (e.g. `--datasets MIMIC`). Outputs:
`summary.json` (per-tier choice accuracy, mean official forecast
metrics, abstention counts), `details/` per row, `gnomonbench.jsonl`.

Notes: the benchmark is new (Feb 2026) and its README announces
human-annotated updates — re-download before comparing across dates.
The healthcare rows are *derived* from MIMIC-IV (no raw records), but
read the upstream data-use terms before publishing per-domain results.
The blind test split and leaderboard submission are out of scope here;
this adapter targets the labeled benchmark split the leaderboard
expects local metrics from.
