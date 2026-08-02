# MTBench — Multimodal Time Series Benchmark

Adapter for [MTBench](https://arxiv.org/abs/2503.16858) (temporal
reasoning and QA over paired financial/weather time series and news),
official code at
[Graph-and-Geometric-Learning/MTBench](https://github.com/Graph-and-Geometric-Learning/MTBench).

## What is official, what is ours

Official (from an MTBench checkout, byte-for-byte unmodified):

- all processed task datasets (their download scripts),
- every evaluation script (`evaluation/finance/*.py`,
  `evaluation/weather/*.py`): prompts, response parsing, metrics,
  per-sample failure filters, and result files.

Ours (this directory):

- `openrouter_patch.py` — replaces the send functions in
  `evaluation.api_call` *in memory* before an official script runs, so
  any of their `--model` branches is served by one OpenRouter model
  through `OPENROUTER_API_KEY` instead of four separate pasted keys.
  Return shapes are preserved; the scripts never know the difference.
- `gnomon_forecaster.py` — the Gnomon treatment for the two task families
  whose answer is a numeric trajectory: finance price forecasting
  (`--indicator time`) and weather temperature forecasting. Reads the
  same official task JSONs; aggregation mirrors the official metric
  block including the official mse>100 sample filter; MAPE is imported
  from the official `evaluation.utils` when the checkout is on the
  path. Disclosed decisions: trading bars are modelled on a synthetic
  regular daily axis (bar *k* = epoch + *k* days; the metric compares
  values only); Gnomon's q50 path is the point forecast; in `agent` mode
  the news text goes to an OpenRouter model that may propose typed
  context events for Gnomon's admission gate — never numbers.

MTBench's QA/MCQA/trend/correlation tasks run under the control path
only for now: they grade text answers, and their official scripts
already handle them end to end.

## Setup

```bash
git clone https://github.com/Graph-and-Geometric-Learning/MTBench ~/MTBench
cd ~/MTBench && pip install -r requirement.txt
python download_processed_dataset.py
export OPENROUTER_API_KEY=sk-or-...
```

## Run

Control — any official script, LLM served by OpenRouter (the script's
own `--model gpt-4o` picks the official dispatch branch; the patch
substitutes your OpenRouter model):

```bash
python -m benchmarks.mtbench.run_mtbench control \
    --mtbench-root ~/MTBench \
    --script evaluation/finance/value_prediction.py \
    --model openai/gpt-4o -- \
    --dataset_folder=../../data/processed/finance/aligned_in30days_out7days \
    --save_path=../../results/finance/pred_time_in30_out7/openrouter-gpt-4o/combined \
    --indicator=time --model=gpt-4o --mode=combined
```

Treatment — Gnomon owns the numbers (forecasting families):

```bash
python -m benchmarks.mtbench.run_mtbench gnomon \
    --mtbench-root ~/MTBench \
    --dataset-folder ~/MTBench/data/processed/finance/aligned_in30days_out7days \
    --output-dir results/mtbench-gnomon-agent \
    --mode agent --model openai/gpt-4o
```

Outputs: `summary.json` (official-style mean MSE/MAE/RMSE/MAPE over
samples passing the official filter, plus abstention counts),
`output_details/` per sample, and `gnomonbench.jsonl` for
`gnomon eval compare` against a control run.
