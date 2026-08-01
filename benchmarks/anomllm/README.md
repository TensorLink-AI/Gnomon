# AnomLLM — "Can LLMs Understand Time Series Anomalies?"

Adapter for the ICLR 2025 benchmark
([paper](https://openreview.net/forum?id=LGafQ1g2D2), official code at
[rose-stl-lab/AnomLLM](https://github.com/rose-stl-lab/AnomLLM)).

The benchmark evaluates anomaly detection on controlled synthetic
datasets (point, range, frequency, trend, flat-trend, and noisy
variants; 400 series each), with LLMs prompted under many variants
(text/vision, 0/1-shot, CoT). Scoring is pointwise precision/recall/F1
plus affiliation metrics, computed by the official `result_agg.py`.

## What is official, what is ours

Official (used unmodified, from an AnomLLM checkout):

- dataset generation (`synthesize.sh`, seeds 42/3407) or their prebuilt
  archives,
- every LLM prompt variant and the control runner (`src/online_api.py`),
- all scoring (`src/result_agg.py` over `results/synthetic/...`).

Ours (this directory):

- `aion_detector.py` — runs Aion's graded anomaly detection
  (`aion.anomaly.detect_anomalies`, the same competing-detector grading
  behind `aion detect`, selection by synthetic-injection F1, no access
  to ground truth) on each eval series and writes predictions into the
  official results tree as model `aion`, in the exact row format the
  official aggregator parses. Disclosed adapter decisions: the
  benchmark's series carry no timestamps, so a synthetic hourly UTC axis
  is attached (the metric is index-based; the axis never enters it);
  flagged points are merged into the benchmark's half-open
  `{"start", "end"}` intervals.
- `run_anomllm.py` — orchestrates the Aion condition and can invoke the
  official control runner.
- `credentials.example.yml` — points the official runner's
  OpenAI-compatible client at OpenRouter, so the control uses the same
  models as every other benchmark here without touching official code.

## Setup

```bash
git clone https://github.com/rose-stl-lab/AnomLLM ~/AnomLLM
cd ~/AnomLLM && pip install -r <their environment> && bash synthesize.sh
cp /path/to/Aion/benchmarks/anomllm/credentials.example.yml credentials.yml
# edit credentials.yml: add your OpenRouter key for each model you'll run
```

Aion itself needs no extra dependencies beyond numpy-free stdlib — the
adapter reads the benchmark's `data.pkl` directly.

## Run

```bash
# Treatment: Aion as the detector (no LLM, no API key needed)
python -m benchmarks.anomllm.run_anomllm \
    --anomllm-root ~/AnomLLM --data point

# Control: official prompt variant through OpenRouter
python -m benchmarks.anomllm.run_anomllm \
    --anomllm-root ~/AnomLLM --data point \
    --skip-aion --control-model openai/gpt-4o-mini \
    --control-variant 0shot-text

# Official scoring — Aion appears as a row next to every LLM variant
cd ~/AnomLLM && python src/result_agg.py --data_name point \
    --label_name point-exp --table_caption "Point anomalies"
```

Repeat `--data` over `point`, `range`, `freq`, `trend`, `flat-trend`,
`noisy-point`, `noisy-freq` for the full suite.

Each Aion run also writes `<variant>.aionbench.jsonl` next to the
predictions (per-series success/abstention/latency rows plus an adapter
F1 preview) for `aion eval compare`; the official aggregator remains the
authoritative scorer.
