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

- `gnomon_detector.py` — runs Gnomon's graded anomaly detection
  (`gnomon.anomaly.detect_anomalies`, the same competing-detector grading
  behind `gnomon detect`, selection by synthetic-injection F1, no access
  to ground truth) on each eval series and writes predictions into the
  official results tree as model `gnomon`, in the exact row format the
  official aggregator parses. Disclosed adapter decisions: the
  benchmark's series carry no timestamps, so a synthetic hourly UTC axis
  is attached (the metric is index-based; the axis never enters it);
  flagged points are merged into the benchmark's half-open
  `{"start", "end"}` intervals.
- `run_anomllm.py` — orchestrates the Gnomon condition and can invoke the
  official control runner.
- `credentials.example.yml` — points the official runner's
  OpenAI-compatible client at OpenRouter, so the control uses the same
  models as every other benchmark here without touching official code.

## Setup

```bash
git clone https://github.com/rose-stl-lab/AnomLLM ~/AnomLLM
cd ~/AnomLLM && pip install -r <their environment> && bash synthesize.sh
cp /path/to/Gnomon/benchmarks/anomllm/credentials.example.yml credentials.yml
# edit credentials.yml: add your OpenRouter key for each model you'll run
```

The adapter's own code is stdlib-only, but the benchmark's `data.pkl`
stores numpy arrays, so numpy must be importable in the environment that
runs the Gnomon condition — unpickling fails without it. Anything that
can `import numpy` suffices; none of the rest of the AnomLLM environment
is needed for the treatment arm.

## Run

```bash
# Treatment: Gnomon as the detector (no LLM, no API key needed)
python -m benchmarks.anomllm.run_anomllm \
    --anomllm-root ~/AnomLLM --data point

# Control: official prompt variant through OpenRouter
python -m benchmarks.anomllm.run_anomllm \
    --anomllm-root ~/AnomLLM --data point \
    --skip-gnomon --control-model openai/gpt-4o-mini \
    --control-variant 0shot-text

# Official scoring — Gnomon appears as a row next to every LLM variant
cd ~/AnomLLM && python src/result_agg.py --data_name point \
    --label_name point-exp --table_caption "Point anomalies"
```

Repeat `--data` over `point`, `range`, `freq`, `trend`, `flat-trend`,
`noisy-point`, `noisy-freq` for the full suite.

`--variant-name` (default `detect`) labels Gnomon's results file. Names
matching upstream's rescaling pattern — `0shot-text-s0.3`,
`1shot-text-s0.3`, and their `-cot` forms — are rejected with an error:
for exactly those names, the official aggregator's `postprocess_configs`
multiplies every integer in the stored responses by the inverse scale
(they denote prompts over a 0.3-subsampled series), which would silently
corrupt Gnomon's full-resolution indices.

## What each arm emits, and how they compare

Treatment (Gnomon), per run:

- `results/synthetic/<data>/gnomon/<variant>.jsonl` — predictions in the
  official row format, scored by `result_agg.py`.
- `gnomonbench/synthetic/<data>/gnomon/<variant>.gnomonbench.jsonl` —
  the adapter's own per-series records (success/abstention/latency plus
  an F1 preview). This sidecar deliberately lives *outside* `results/`:
  upstream's `collect_results` sweeps that whole tree and scores every
  `*.jsonl` whose name lacks `requests`, and a sidecar placed there once
  rendered a phantom all-zero variant in the official table. Should a
  custom `records_path` point back under `results/`, the adapter inserts
  `requests` into the filename so the official filter skips it.
- `manifest.json` next to the sidecar — run provenance in the same
  format `benchmarks/run_all.py` records.

The summary's F1 preview follows the official convention — a clean
series with an empty prediction scores 1.0 and is included in the mean
(reported as `preview_mean_pointwise_f1_official_convention`, with
correct silences also counted as `clean_series_correctly_silent`) — so
it tracks the official table; `result_agg.py` stays authoritative.

Control (official LLM runner), per run:

- `results/synthetic/<data>/<model>/<variant>.jsonl` and a
  `*_requests.jsonl` cache — written by the official `online_api.py`,
  untouched by us. No GnomonBench records: the control's rows come from
  official code this adapter does not modify.
- `gnomonbench/synthetic/<data>/<model, slashes flattened>/manifest.json`
  — provenance for the control arm.

Because only the treatment arm has GnomonBench records,
treatment-vs-control comparison for this benchmark does **not** go
through `gnomon eval compare` (which needs those records on both sides).
Both arms meet in the official `result_agg.py` table instead — one row
per condition, scored by identical code over the same series ids.
