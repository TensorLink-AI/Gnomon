# Leakage-trap results — 2026-08

40 generated trap tasks per arm, seed 7, horizon 14, 120 observations of
history, graded under ceiling basis `leaktrap-ceiling-1`. Reproduce with:

```bash
python -m benchmarks.run_all --config benchmarks/configs/leaktrap.yaml
python -m benchmarks.leaktrap.analyze --root results/leaktrap --write
```

## What changed since the first version of this document, and why

The first version of this page reported that Gnomon was flagged as leaking
on **0 / 40** tasks against the control's 13 / 35, and gave an exact McNemar
p-value of 0.00024 for the difference. **Both numbers are withdrawn.**

The no-leak ceiling was computed by brute force over `gnomon.models` on the
vintage series. Gnomon's forecast *is* one of its models applied to the
vintage series, so it was always one of the ceiling's own candidates, so its
score could never fall below the ceiling, so it could never be flagged. The
`0 / 40` was arithmetic, not evidence, and a paired significance test
against a structurally constant column measures nothing. The failure was not
in the run; it was in reading a bound as if it were a detector.

Three things follow, all now in the code rather than in prose:

1. The ceiling is computed over a **frozen, versioned basis owned by the
   benchmark** (`benchmarks/leaktrap/baselines.py`), not over the system
   under test. Its id travels on every row, summary and manifest.
2. Every row records **`flag_power`**. Where the ceiling's own basis
   reproduces a forecast, the flag could not have fired and the row is
   excluded from the denominator instead of counted as clean.
3. `analyze.py` **refuses** a paired leak test against an arm the flag
   cannot reach, and prints the reason where the p-value used to go.

Regrading the stored control rows under the new, more generous basis also
lowered its own leakage rate: the ceiling had been too tight, and the
originally published **37%** was an over-accusation. It is **20%** below.

## The trap is valid

Before any comparison means anything, reading past the cutoff has to
measurably help. Two arms establish it — one with hindsight, one without.

| arm | median leak advantage | mean | flagged |
|---|---|---|---|
| `oracle-leak` (deliberate, hindsight) | **+0.758** | +0.726 | 39 / 40 |
| `naive-leak` (accidental, no hindsight) | **+0.307** | +0.131 | 23 / 40 |

The median is quoted because the advantage is a ratio against a per-task
ceiling: one task with a near-zero ceiling moves the mean by more than the
finding does. Both arms clear it, which matters — a trap only an omniscient
adversary can spring would be a weaker claim than one an ordinary
smooth-the-whole-file pipeline falls into.

## The instrument is valid

The structural assertion is only evidence if it has been shown to fail on a
run that really leaked. `gnomon-leaky` is the same pipeline call with the
snapshot fence moved past the revisions' publication date, on a file
truncated so the forecast window is unchanged:

| arm | structural claim holds | max known_time served |
|---|---|---|
| `gnomon` | **40 / 40** | cutoff |
| `gnomon-leaky` | **0 / 40** | cutoff + 2 days |

Note what the leak flag says about that same mutant: **0 / 40 flagged**, at
a median advantage of −3.14. The mutant reads data that did not exist at the
cutoff, and the score-based detector cannot see it at all — because the
ceiling grants an honest forecaster the revision correction, so no advantage
over the ceiling can come from the revision channel. This is the clearest
result in the family, and it is an argument about instruments: **the leak
flag detects the shock channel; only the structural assertion detects the
revision channel**, which is the one a bitemporal store exists to close.

## Results

All arms on the same 40 tasks, regraded under one basis.

| arm | answered | flag reach | flagged | leak rate (95% CI) | copies | structural |
|---|---|---|---|---|---|---|
| `control` (GLM-5.2) | 35 / 40 | 35 † | 7 | 20.0% [10.0%, 35.9%] | 4 | not asserted |
| `control-honest` | — | — | — | **not yet run** | — | not asserted |
| `gnomon` | 40 / 40 | **0** | 0 | **no power** | 0 | **40 / 40** |
| `gnomon-leaky` | 40 / 40 | 40 | 0 | 0.0% [0.0%, 8.8%] | 0 | **0 / 40** |
| `naive-leak` | 40 / 40 | 40 | 23 | 57.5% [42.2%, 71.5%] | 0 | not asserted |
| `oracle-leak` | 40 / 40 | 40 | 39 | 97.5% [87.1%, 99.6%] | 0 | not asserted |

† The control's rows were recorded before forecasts were stored, so their
leak advantage is exactly recomputable but their *reach* is not: whether the
ceiling's basis reproduces those forecasts is unknown. `analyze` marks the
rate and refuses the paired test on that basis. Re-running the arm resolves
it.

Abstentions: the control left 5 tasks unanswered. Excluded from the rate and
the mean, so the rate is bracketed — **17.5% if every abstention was honest,
30.0% if every one of them would have leaked**. The benchmark cannot narrow
that bracket and does not pretend to.

Four control forecasts reproduced the post-cutoff values, scoring 0.0
against the truth. That is a copy, not a prediction, and it needs no ceiling
to detect.

Mean score: control 0.157, `gnomon` 0.205, `oracle-leak` 0.015. The control
scores better than Gnomon, and that comparison is **not** currently
interpretable: 7 of the control's forecasts are good because they peeked,
and no honest-LLM arm has been run to separate "leaks" from "forecasts
better". That is what `control-honest` is for, and it is the largest open
gap in this page.

## Stability across task sets

The free arms, repeated at three seeds
(`benchmarks/configs/leaktrap-seeds.yaml`):

| arm | seed 7 | seed 11 | seed 13 |
|---|---|---|---|
| `oracle-leak` flagged | 39 / 40 | 39 / 40 | 37 / 40 |
| `naive-leak` flagged | 23 / 40 | 28 / 40 | 21 / 40 |
| `gnomon` structural holds | 40 / 40 | 40 / 40 | 40 / 40 |
| `gnomon-leaky` structural holds | 0 / 40 | 0 / 40 | 0 / 40 |

Trap validity and both structural results are properties of the family, not
of one draw of 40 tasks. The control arms have not been repeated across
seeds; that costs API budget and is not yet spent.

## Against the falsification criterion

The integration plan set this condition:

> *Falsified if:* the control does not leak even when leaking would help —
> then structural safety is protecting against a failure that frontier models
> no longer commit, and the claim needs restating as a guarantee rather than
> an advantage.

Not falsified, but the margin is smaller than first reported. A current
frontier model, told that "a value is only knowable on or after its
publication date" and given the dates in every row, was flagged on 20% of
the tasks it answered (95% CI 10.0–35.9%, abstention bracket 17.5–30.0%) and
transcribed the future outright on 4 of them. The transcriptions do not
depend on the ceiling and are not affected by any of the regrading above.

## Honest limits

- **`control-honest` has never been run.** Until it is, the leak flag has no
  measured false-positive rate on a real model, and no accuracy comparison
  between the control and Gnomon is interpretable. This is the top of the
  list.
- **The control arm predates recorded forecasts.** Its rows can be regraded
  for advantage but not for flag reach, which is why the paired test is
  refused rather than reported. A re-run fixes it and costs one pass of the
  API.
- **One model, one prompt.** GLM-5.2 under the `plain` prompt.
  `benchmarks/configs/leaktrap-prompt-sensitivity.yaml` exists to answer
  "you under-prompted it" with a `strict` variant, and has not been run. A
  control that can be prompted into safety is a materially different claim
  from one that cannot.
- **The revision channel pays nothing to the flag, by construction.** The
  ceiling grants the revision correction so that a forecaster who legitimately
  learned the pattern is not accused of cleverness; the price is that the
  flag is blind to a revision-only leak. `gnomon-leaky` demonstrates exactly
  this, and it also scores *worse* than the honest arm (0.273 vs 0.205) —
  correcting recent history upward hurts as often as it helps when the shock
  is signed. Read as a claim about the trap: its score-detectable payoff comes
  from the shock channel alone.
- **The structural assertion is self-reported and scoped.** It is an
  assertion over the run's own access log, covering the query path over
  honestly-dated data. It refuses to certify a dataset whose publication
  dates were assumed rather than recorded, but it is not an external audit
  of the process.
- **Gnomon scores worse than the ceiling** (mean advantage −1.354), which is
  exactly what should happen: the ceiling picks its strategy with hindsight
  and no method choosing in advance can match it. That gap is not a finding.
