# Benchmark setup

Every step below was needed to run the suite end to end; none of it is
optional folklore. Two interpreters are required — CiK pins a torch
version with no Python 3.13 wheels, so it gets its own environment.

## 1. The main environment

```bash
uv venv .venv
uv pip install --python .venv -e . -r benchmarks/requirements/base.txt
export OPENROUTER_API_KEY=sk-or-...   # or put it in .env at the repo root
```

`.env` is read automatically by `benchmarks/common/envfile.py`, and is
git-ignored. One key serves every model in the suite.

## 2. Datasets, per benchmark

**TemporalBench** and **TimeSage-MT** fetch themselves:

```bash
.venv/bin/python -m benchmarks.temporalbench.run_temporalbench \
    --download --data-dir ~/temporalbench          #  44 MB
.venv/bin/python -m benchmarks.timesage_mt.run_timesage \
    --download --data-dir ~/timesage-mt            # 576 MB, 1875 files
```

The TimeSage download is large and parallel. On a constrained machine
(WSL2 in particular) throttle it:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_WORKERS=2 .venv/bin/python -m ...
```

**MTBench** — clone, fetch one dataset, and materialise the per-task JSONs
the official evaluation scripts read. The official
`download_processed_dataset.py` fetches parquet shards, but the official
scripts `glob("*.json")`, so the export step is what makes an official
checkout runnable at all:

```bash
uv pip install --python .venv -r benchmarks/requirements/mtbench.txt
git clone --depth 1 https://github.com/Graph-and-Geometric-Learning/MTBench ~/MTBench
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("afeng/MTBench_finance_aligned_pairs_long",
                  local_dir="~/MTBench/data/processed/finance/aligned_in30days_out7days",
                  repo_type="dataset", max_workers=2)
PY
.venv/bin/python -m benchmarks.mtbench.export_tasks \
    --parquet-dir ~/MTBench/data/processed/finance/aligned_in30days_out7days \
    --output-dir ~/MTBench/data/tasks/finance_long --limit 50
```

Point **both** conditions at `~/MTBench/data/tasks/finance_long` so they
read identical inputs.

**AnomLLM** — clone, install, write credentials, generate the data. The
official `src/` imports vendor SDKs and reads `credentials.yml` at import
time, so the file must exist even to generate data:

```bash
uv pip install --python .venv -r benchmarks/requirements/anomllm.txt
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu torch
git clone --depth 1 https://github.com/rose-stl-lab/AnomLLM ~/AnomLLM
cp benchmarks/anomllm/credentials.example.yml ~/AnomLLM/credentials.yml
# edit: one entry per model id you will run, api_key = your OpenRouter key,
# base_url = https://openrouter.ai/api/v1 — then chmod 600 it.
cd ~/AnomLLM && PYTHONPATH=src python src/data/synthetic.py --generate \
    --data_dir data/synthetic/trend/eval \
    --synthetic_func synthetic_dataset_with_trend_anomalies --seed 42
# ...repeat per split/family, or run the official synthesize.sh for all.
```

**LeakTrap** (internal) needs no dataset: tasks are generated
deterministically from the seed at run time, so all three arms see
identical series. It runs from the main environment via
`benchmarks/configs/leaktrap.yaml`; only the control arm spends LLM
tokens.

**CiK** — its own interpreter:

```bash
uv venv --python 3.12 .venv-cik
uv pip install --python .venv-cik -e . -r benchmarks/requirements/cik.txt
```

Run CiK with `.venv-cik/bin/python`. The official result cache writes
HDF5 and fails under current pandas (`Object dtype dtype('O') has no
native HDF5 equivalent`) — pass `--no-cache`.

## 3. Running

```bash
.venv/bin/python -m benchmarks.run_all --config benchmarks/configs/glm52.yaml --dry-run
.venv/bin/python -m benchmarks.run_all --config benchmarks/configs/glm52.yaml \
    --only tb-control,tb-gnomon --continue-on-error
```

Each run writes a `manifest.json` next to its results recording the
command, model, config and code revision, which
`benchmarks/report.py` uses to refuse mismatched comparisons.

## 4. Cost and time, before you start

Reasoning models dominate the bill: GLM-5.2 spends 15–20k reasoning
tokens per answer, so a 50-row arm is ~2 hours and a few dollars, and two
control arms are far more than the rest of the suite combined:

| Arm | Scale | Cost | Wall clock |
| --- | --- | --- | --- |
| TemporalBench control / Gnomon | 50 rows | ~$2.9 / ~$1.1 | ~2 h each |
| TimeSage direct / gnomon-tools | 127 turns | ~$5.9 / ~$2.6 | ~4 h / ~1.5 h |
| MTBench control / gnomon / tools | 50 tasks | ~$0.26 / ~$0.10 / ~$0.50 | < 1 h |
| CiK control | 71 tasks × 50 samples | **~$140** | many hours |
| AnomLLM control | 400 series | **~$18** | ~22 h |
| CiK / AnomLLM Gnomon arms | full | $0 (no LLM) | ~20 min |

`--limit` caps the LLM-bearing adapters (TemporalBench, TimeSage,
MTBench). CiK scales by `--seeds` and `--task-filter`; AnomLLM's official
runner has no cap but resumes by `custom_id`, so an interrupted run
continues where it stopped.
