# Results: verifiable future-context events on CiK

Status: **the pre-registered benchmark run has not been executed.**
Written 2026-08-03 against `HYPOTHESIS.md`, at the commit that implements
the feature. Nothing below is a score, and no hypothesis is claimed
confirmed or refuted.

## What ran

- The full Gnomon suite plus the new lane's tests: **714 passed, 2
  skipped** in `tests/` (was 627 before the feature; the lane's own file
  carries 80 tests) and **90 passed** in `benchmarks/tests/`. Coverage
  includes: span parsing (CiK's "bounded above/below by", "less than or
  equal to", `≤`/`≥`, interval notation, positivity; negation resolution
  so "no more than 60" and "will not stay below 100" read on the stated
  side; refusal of percentages, date/clock ranges, and truncated
  scientific notation); suspect-claim rejection against recent history;
  rejection of overrides that contradict an admitted constraint or each
  other (order-independent); clamp and override-window correctness
  including true-edge-only boundary widening; the `context_trusted`
  support downgrade; threshold-analysis and `point_bias_correction`
  consistency with the published rows; the history-only counterfactual
  in evidence; ID-payload stability with the flag off; the general
  workflow's verbatim-quote-to-`source_span` chain with model-supplied
  `source_span`/`claim` attributes discarded; and the adapter's
  verbatim-quote, overlong-span, and cache-separation checks. An
  adversarial review of the full diff was run and every confirmed
  finding fixed and pinned by a test. Golden artifacts are
  byte-identical with the flag off.
- Nothing else. No CiK task was executed under either condition.

## Why the CiK treatment arm did not run

The environment this change was built in has:

- **no OpenRouter (or any LLM) API key** — the `gnomon-agent` arm needs
  the proposer, so the treatment condition cannot run at all;
- **no CiK corpus, official metric scaling cache, or `.venv-cik`** —
  without the scaling cache the official RCRPS returns NaN;
- **no `results/deepseek-v4-flash/` baseline artifacts** — they are
  untracked (regenerable) and exist only on the machine that produced
  them, and H1 is defined on matched task-seeds against exactly that run.

Running a substitute (for example `gnomon-pure`, which needs no key)
would not test H1 and is not reported as if it did.

## What the executed tests do and do not establish

They establish the **mechanism**: numbers are only ever applied when a
deterministic parser re-extracts them from the quoted span; a claimed
bound that disagrees with its span is rejected; a bound the recent
history violates is rejected as suspect; constraints clamp every emitted
quantile monotonically and only inside their window; override windows
take the stated value with boundary-widened intervals; influenced
forecasts are downgraded and disclosed; and a flag-off run is
byte-identical to a build without the feature.

They establish **nothing** about H1–H4: whether the lane closes any of
the RCRPS gap, whether admissions occur at CiK's actual phrasing
distribution, whether constraints dominate overrides, or whether harm
cases appear. Those are exactly what the pre-registered run measures,
and the parser's pattern coverage against CiK's real context text is an
open risk (a span the parser cannot read is a rejection, which is safe
but scores as no improvement).

## To execute the pre-registered run

On a machine with `OPENROUTER_API_KEY`, the CiK environment
(`benchmarks/SETUP.md`), the scaling cache, and the
`results/deepseek-v4-flash/cik-gnomon-agent` baseline:

```bash
.venv-cik/bin/python -m benchmarks.cik.run_cik \
    --method gnomon-agent --model deepseek/deepseek-v4-flash \
    --future-context --no-cache \
    --output-dir results/future-context-ab/cik-gnomon-agent-future

gnomon eval compare \
    --baseline results/deepseek-v4-flash/cik-gnomon-agent/gnomonbench.jsonl \
    --treatment results/future-context-ab/cik-gnomon-agent-future/gnomonbench.jsonl
```

Then replace this file's findings with the matched-task-seed comparison
against the thresholds in `HYPOTHESIS.md` — including per-class
admission counts (from the `future_context_gate` evidence in each run's
artifacts) and every harm case, whatever the means say. The hypothesis
file's thresholds are frozen; this file is the only one that changes.
