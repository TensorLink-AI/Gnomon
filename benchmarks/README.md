# AionBench

**Does routing a temporal question through Aion make a model better at it?**

Three published temporal-reasoning benchmarks, run under two conditions on
identical items: the model on its own, and the same model with Aion's tool
surface. Models are reached through OpenRouter, so open- and closed-weight
frontier models share one code path and one key.

```bash
pip install -e '.[bench]'
export OPENROUTER_API_KEY=sk-or-...

# Check the harness wiring with no key and no spend
aionbench run --suite tsqa --limit 16 --models echo/echo --dry-run --output runs/smoke

# A real run
aionbench run \
  --suite timeseriesexam \
  --models anthropic/claude-sonnet-4.5,deepseek/deepseek-r1 \
  --conditions direct,aion \
  --limit 120 \
  --budget-usd 15 \
  --output runs/tse-2026-08
```

The run directory holds `rows.jsonl` (one graded row per cell),
`manifest.json` (every knob that could move a number), `report.md`, and
per-cell attempt traces under `work/`.

## The suites

| Suite | Source | Items | What it tests |
| --- | --- | ---: | --- |
| `timeseriesexam` | [arXiv:2410.14752](https://arxiv.org/abs/2410.14752) | 746 | Do models understand time-series *concepts* — pattern, noise, similarity, anomaly, causality? IRT-refined multiple choice. |
| `tsaia` | [arXiv:2509.01822](https://arxiv.org/abs/2509.01822) | 1,054 | Can a model act as a time-series *assistant* — constraint-aware forecasting, threshold-calibrated detection, financial decisions across 33 task types? |
| `tsqa` | [arXiv:2601.13653](https://arxiv.org/abs/2601.13653) | corpus-dependent | Does tool augmentation move agentic time-series QA? The TimeART ablation, with Aion as the toolbelt. |

### `timeseriesexam`

Cai et al., *TimeSeriesExam: A Time Series Understanding Exam*. Downloads
[`AutonLab/TimeSeriesExam1`](https://huggingface.co/datasets/AutonLab/TimeSeriesExam1)
once and runs offline thereafter. The paper's finding — models handle simple
pattern questions and collapse on causality — makes the per-category
breakdown the interesting output, not the aggregate: a deterministic detector
should move anomaly and causality items specifically, and the report says
whether it did.

This harness runs the paper's **text** arm (values to one decimal place,
comma separated, unscaled). Hints and the concept glossary are off by
default; `--include-hint` and `--include-concepts` reproduce the paper's
ablation, in which concepts sometimes *hurt*.

```bash
aionbench run --suite timeseriesexam --models @benchmarks/configs/models.yaml \
  --categories causality_analysis,anomaly_detection --limit 200 --output runs/tse
```

### `tsaia`

Ye et al., *When LLM Meets Time Series*. Two subsets that behave very
differently:

- `--config multiple_choice` (150 items) is self-contained. One CSV download,
  no pickles, no code execution. **Start here.**
- `--config analysis_questions` (904 items) is the real test. Its executor
  variables and answer key ship as pickle files and its native protocol has
  the model answer by writing Python. Both are opt-in:

```bash
aionbench run --suite tsaia --config analysis_questions \
  --models openai/gpt-5 --conditions code,aion_code \
  --allow-pickle --allow-code-execution \
  --limit 100 --output runs/tsaia
```

`--allow-pickle` is required because unpickling executes arbitrary code by
construction. `--allow-code-execution` runs model-written Python in a
subprocess with wall-clock and memory caps and no proxy or provider
credentials — containment against a careless model, not a hostile one. Run
the whole harness in a container for untrusted models.

**Two departures from the paper, both deliberate and both recorded in the
manifest.** First, the paper's protocol is CodeAct throughout; this harness
also offers non-code conditions so a tool-augmented run can be compared to a
code-writing one on identical items. Second, the paper uses task-specific
success bars that are not published per task type, so the harness applies a
documented default per metric family (`mape` ≤ 0.15, `relative_error` ≤ 0.10,
`f1` ≥ 0.50), overridable with `--threshold mape=0.2`. Every report states
the bar it used. **Do not compare this harness's TSAIA pass rate to the
paper's table** — compare control against treatment within a run.

### `tsqa`

Wu et al., *TimeART: Towards Agentic Time Series Reasoning via
Tool-Augmentation*. TimeART's claim is that a model plus strong
out-of-the-box tools behaves like a data scientist on TSQA. This suite runs
the ablation at the centre of that claim, with Aion as the tools. It does not
reproduce TimeART's training pipeline and ships no TimeART weights.

TSQA corpora (MTBench, TimeMQA) move on their own release schedules, so the
suite is corpus-agnostic: point it at a local file or a Hugging Face dataset
and supply a field map.

```bash
# The built-in offline fixture (16 items) — wiring check only, never a result
aionbench run --suite tsqa --models echo/echo --dry-run --output runs/smoke

# Your own corpus
aionbench run --suite tsqa --config generic --data-path corpora/mtbench.jsonl \
  --field-map question=q,options=choices,answer=label,series=ts \
  --models @benchmarks/configs/models.yaml --output runs/tsqa
```

Row format: `{"id", "question", "options": [...], "answer", "series": [{"name", "values": [...]}], "category"}`.
`answer` may be a letter, a 0- or 1-based index, or the option text.
Accuracy is reported per *n*-way group, because TimeART's 3-way and 5-way
numbers are not comparable and must not be pooled.

## Conditions

| Condition | What the model gets |
| --- | --- |
| `direct` | The series as text. Answers from the prompt alone. **Control.** |
| `aion` | The series as text *and* on disk as CSV, plus Aion's tool surface over OpenRouter function calling. **Treatment.** |
| `code` | Writes Python against pre-loaded variables; executed in the sandbox. Control for TSAIA. |
| `aion_code` | Same, with `aion` importable in the sandbox. |

The `aion` condition exposes Aion's *real* tool surface — the schemas and
in-process runners from `aion.toolspec` that the MCP server uses, not a
benchmark-only imitation. Benchmark rows carry bare value arrays, so the
harness materialises each series as a CSV with a regular synthetic index;
that synthesis is disclosed in the prompt and in the manifest.

The tool loop is deliberately unhelpful: it does not repair malformed tool
arguments, retry a failed call, or nudge the model toward a tool. A model
that cannot drive Aion is a finding, not a harness bug.

## Reading a report

```
| Model | Control | Treatment | Pairs | Control | Treatment | Δ | Fixed | Broken | p |
| claude-sonnet-4.5 | direct | aion | 200 | 61.5% | 74.0% | +12.5% | 31 | 6 | 0.0001 |
```

- **Pairs** — items *both* arms graded. An item one arm failed to transport is
  dropped from both, so the comparison never rests on a different question set.
- **Fixed / Broken** — items the treatment got right and the control wrong,
  and the reverse. Aion breaking items is a real signal; the report never
  hides it behind a net delta.
- **p** — exact two-sided McNemar over the discordant items. `(ns)` marks a
  difference the run cannot distinguish from noise. Benchmark tables
  routinely report two-point differences on 150 items as results; this one
  labels them.
- **Grounded** — how often an Aion tool call actually succeeded. A treatment
  arm whose wins never touched a tool is measuring the prompt, not the tools.

Every failure carries a typed reason, so "worse accuracy" can be told apart
from "never produced a parseable answer":

| Failure | Meaning |
| --- | --- |
| `structural` | No parseable answer in the response |
| `execution` | Generated code raised, timed out, or was refused |
| `constraint` | Answer violated a stated constraint |
| `threshold` | Answer parsed and ran but missed the metric's bar |
| `refusal` | Model declined |
| `transport` | Provider or network error — excluded from every denominator |

## Cost, resumption, and reproducibility

- **Cost is measured, not estimated.** OpenRouter returns the credits actually
  spent per call, upstream markup included.
- `--budget-usd N` stops scheduling new work once spend reaches `N`.
  Backpressure is real: only `2 × concurrency` cells are ever in flight.
- Responses are cached on disk, keyed by the full request body. Re-running
  the same command resumes from `rows.jsonl` and costs close to nothing.
  Change a prompt, a temperature, or a model and the key changes.
- Sampling is seeded (`--seed`), so `--limit 200` picks the same 200 items
  every time. It is a random sample, not the first N rows: these corpora are
  grouped by template, and the head of the file is never a fair sample.
- `--repeats k` runs each cell `k` times for nondeterministic models. Report
  pass^k, not best-of.

## Verify your model roster first

Slugs get renamed and retired constantly. An unknown slug fails every cell it
appears in:

```bash
aionbench models --roster benchmarks/configs/models.yaml   # non-zero if any are missing
aionbench models --filter deepseek                          # browse the catalogue with pricing
```

## Handing results to `aion eval compare`

```bash
aionbench export --run runs/tse --model anthropic/claude-sonnet-4.5 --output runs/tse/compare
aion eval compare --baseline runs/tse/compare/*-direct.jsonl \
                  --treatment runs/tse/compare/*-aion.jsonl
```

The export flags a correct treatment answer that never touched a tool as
`invented_number` — precisely the failure mode Aion exists to prevent.

## Adding a suite

Write one file in `aionbench/suites/`: subclass `Suite`, implement `load()`
and `score()`, decorate with `@register`. Prompting, transport, concurrency,
caching, budgeting, and reporting are the harness's business.

## What these numbers are not

These are public corpora of synthetic and scraped series. A result here says
a model got better at *these questions* when given Aion's tools. It does not
say the corpus resembles your operational data, and no accuracy figure on a
benchmark is a claim about a forecast on your data. The smoke corpus in
`aionbench/data/tsqa_smoke.jsonl` is a wiring check with sixteen
constructed items — never quote accuracy on it.
