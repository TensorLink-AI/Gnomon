# Benchmarking Aion on temporal reasoning

`aion eval compare` answers "did Aion help this agent on our tasks?" and
`aion eval episodes` answers "does the harness catch the trap families?"
AionBench answers the third question: **does Aion help a language model on
temporal-reasoning benchmarks other people published?**

It lives in [`benchmarks/`](../benchmarks/README.md) as a separate package,
`aionbench`, so Aion's runtime stays dependency-free. Full operating
instructions are in that README; this page covers the design and how the
three tools fit together.

## The design in one paragraph

Every suite runs the same items, the same model, and the same decoding
parameters under at least two conditions — the model on its own (`direct`),
and the model with Aion's tool surface (`aion`) — and reports the *paired*
difference on items both arms graded. Models are reached through OpenRouter,
so an open-weight 235B and a closed frontier model differ only by a slug.
Aion's real tool schemas and runners are exposed, taken from
`aion.toolspec`, the same source the MCP server uses.

## Why paired, and why a significance test

Item difficulty dominates variance on these corpora. Running control and
treatment on identical task ids removes it, so the comparison only has to
explain the items where the two conditions disagreed — which is exactly what
an exact McNemar test does. The report prints the discordant counts (`fixed`
and `broken`) alongside the p-value, and labels differences it cannot
distinguish from noise. A two-point accuracy difference on 150 items is not a
result, and the table says so.

## The three benchmarks

| Suite | Paper | Why it is here |
| --- | --- | --- |
| `timeseriesexam` | [arXiv:2410.14752](https://arxiv.org/abs/2410.14752) | Isolates *concept understanding* — pattern, noise, similarity, anomaly, causality — free of domain knowledge. Models collapse on causality, which is precisely where a deterministic detector should beat intuition. The per-category table is the finding. |
| `tsaia` | [arXiv:2509.01822](https://arxiv.org/abs/2509.01822) | The closest thing to Aion's actual job: constraint-aware forecasting, threshold-calibrated detection, and financial decisions over real data, with a failure taxonomy (structural / execution / constraint / sub-threshold) rather than a pass rate. |
| `tsqa` | [arXiv:2601.13653](https://arxiv.org/abs/2601.13653) | TimeART's thesis is that tool augmentation makes an LLM behave like a data scientist on time-series QA. This runs that ablation with Aion as the tools. |

## What is faithful and what is not

Benchmark harnesses drift from their papers quietly. These departures are
deliberate and recorded in every run manifest:

- **TimeSeriesExam** runs the paper's text arm, not the image arm. Hints and
  the concept glossary are off by default; both are flags, so the paper's
  ablation (concepts sometimes *hurt*) is reproducible.
- **TSAIA** is natively CodeAct. This harness adds non-code conditions so a
  tool-augmented run can be compared to a code-writing one on identical
  items. Its per-task success bars are not published, so the harness applies
  a documented default per metric family and prints the bar it used. TSAIA
  pass rates from this harness are **not** comparable to the paper's table;
  the within-run control-vs-treatment comparison is the sound one.
- **TSQA** does not reproduce TimeART's training pipeline and ships no
  weights. It is corpus-agnostic — MTBench and TimeMQA move on their own
  schedules — and takes a field map for whatever you point it at.

## Grading rules

Scoring is deterministic. No suite grades with a model. Aion's own
convention carries over: a missing answer and a wrong answer are different
results, and a provider failure is neither.

- Transport failures score `correct=None` and are excluded from every
  accuracy denominator, then counted separately so the exclusion is visible.
- A shape mismatch (a scalar question answered with a vector, a 9-step
  horizon answered with 8 values) is a failure, never a truncation.
- MAPE skips zero actuals rather than dividing by an epsilon, and reports
  undefined when every actual is zero. An epsilon turns one zero into an
  arbitrarily large error and silently decides the comparison.
- Treatment arms record whether a correct answer was actually *grounded* in a
  successful Aion call. A treatment arm whose wins never touched a tool is
  measuring the prompt.

## Two opt-in hazards

Both are off by default and both print a warning when enabled.

- `--allow-pickle` — TSAIA ships its executor variables and answer key as
  pickles, and unpickling executes arbitrary code by construction.
- `--allow-code-execution` — runs model-written Python in a subprocess with
  wall-clock and memory caps, in a throwaway directory, with proxy and
  provider credentials stripped from the child environment. That is
  containment against a careless model, not a hostile one. Run the harness in
  a container for untrusted models.

## How the three evaluation tools relate

| Tool | Question | Grader |
| --- | --- | --- |
| `aion eval episodes` | Does the harness catch temporal leakage, invented numbers, and silent warnings? | Simulated worlds with planted bait; mechanical |
| `aionbench run` | Does Aion improve a model on published temporal-reasoning tasks? | Deterministic scorers over public corpora |
| `aion eval compare` | Does Aion improve *your* agent on *your* tasks? | Yours |

`aionbench export` writes `aion eval compare` inputs directly, so a
benchmark run feeds the uplift tooling without a converter — and it flags a
correct-but-ungrounded treatment answer as `invented_number`, which is the
failure Aion exists to prevent.

## What a result here does and does not say

These are public corpora of synthetic and scraped series. A positive result
says a model got better at *those questions* when given Aion's tools. It
says nothing about whether the corpus resembles your operational data, and no
benchmark accuracy is a claim about a forecast on your data. Aion's
`supported` status means the deterministic checks passed — here as
everywhere else.
