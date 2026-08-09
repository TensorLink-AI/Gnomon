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
  Return shapes are preserved (`send_to_openai_chatgpt` and
  `send_to_openai_o1` return message-like objects, the rest strings);
  prompts, parsing, and scoring are untouched. Known protocol deltas,
  disclosed in the module docstring: the patched functions use the
  shared client's `max_tokens=4096` and the adapter's `--temperature`
  instead of whatever the script passes, and the client retries
  transient HTTP errors and escalates the budget on empty
  length-truncated completions — upstream does neither.
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
- `tool_agent.py` (`--mode tools`) — the model gets the history and the
  article plus a tool loop over Gnomon, and `submit_forecast` has
  exactly two honest exits, mirroring the CiK MCP arm
  (`docs/design/cik-mcp-tool-arm.md`): a `forecast_ref` from a
  `gnomon_forecast` run in that sample, whose trajectory is used
  **verbatim** (the model cannot edit a digit), or the model's own
  `values` — a plain point trajectory, one number per horizon step,
  matching the MSE/MAPE scorer's shape (no quantile triples: MTBench
  scores a point path). `forecast_ref: "none"` abstains. Every sample
  records its route — `gnomon` (submitted a ref), `informed-direct`
  (own values after at least one tool call), `direct` (own values, no
  tool calls) — and `engine_abstentions`, the count of Gnomon
  abstentions the model saw in that run, so a model answer written past
  a Gnomon refusal is always labeled, never laundered into a Gnomon
  number or a silent guess; `summary.json` aggregates the route counts.
  Earlier revisions had only the ref exit, which forced every engine
  abstention to score as a benchmark abstention; the CiK two-arm
  evidence (on all 96 engine-abstention runs the model's own reasoned
  forecast beat both abstaining and the engine's best-effort fallback)
  motivated the second exit.

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

Both subcommands default to the same temperature (0.7, upstream's
chatgpt default); the ground rules require the two arms of a comparison
to share it, so configs should set it explicitly for both.

Outputs: `summary.json` (official-style mean MSE/MAE/RMSE/MAPE over
samples passing the official filter, plus abstention/error counts and
which MAPE implementation scored the run), `output_details/` per
sample, and `gnomonbench.jsonl` for the treatment arm. The control arm
emits no GnomonBench records — the official script writes its own
result files — so treatment-vs-control comparison goes through
`benchmarks/report.py`, which joins the two arms per task via its
`output_details` loader, not `gnomon eval compare`. Note also that
`benchmarks/run_all.py` injects `--limit` only into the treatment
subcommand: a config pairing a full-dataset control with a limited
treatment produces two `summary.json` means that are not directly
comparable; the per-task matched join in `report.py` is the comparison
path.
