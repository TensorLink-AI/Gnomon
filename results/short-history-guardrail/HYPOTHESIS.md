# Pre-registered hypotheses: short-history guardrail + lead-time widening

Registered 2026-08-03 at the commit that implements the feature, **before**
the benchmark run. Restated verbatim from
`docs/design/news-regime.md` ("Pre-registered hypotheses for build
item 1"), where they were frozen at commit `fe39738` before any
implementation existed. The 50-task surrogate and scripts in
`results/news-regime-explore/` are the fixed benchmark
(`scripts/build_tasks.py` seed 20260803 +
`scripts/run_experiments.py`). Both mechanisms were *simulated* in the
exploration session (`results/news-regime-explore/iterate_analysis.json`:
guardrail 2.98 MSE / 1.69% MAPE; √step-anchored-1 widening 89.7%
pooled), so these are predictions that the integrated implementation
reproduces the simulation — an implementation that misses them differs
from the simulated mechanism in a way that must be explained, not
accepted.

> **H-G1 (guardrail closes the self-inflicted gap).** With the
> fold-starved selection guardrail on, gnomon-pure on the 50-task
> surrogate scores filtered mean MSE ≤ 3.0 and mean MAPE ≤ 1.8%
> (from 7.42 / 2.37%; floor is 2.56 / 1.61%), with abstention count
> unchanged (0) and wins-vs-floor not worse than 16/50.
> *Falsifier:* MSE > 3.0, or any new abstention, or a single task where
> the guarded forecast is > 10% worse MSE than the raw floor without a
> disclosed reason.
>
> **H-G2 (long history untouched).** On every series in the existing
> golden set with ≥ 4 rolling origins, artifacts are byte-identical
> (IDs and bodies).
> *Falsifier:* any diff.
>
> **H-G3 (intervals honest with lead).** With √step widening anchored
> at step 1 (the schedule selected by the second-pass rule search,
> which measured 89.7% pooled / 88% step-7 / 82% step-1 in simulation)
> active at short history, the implemented lane reproduces the
> simulation within noise: pooled q10–q90 coverage on the 50-task
> surrogate in **[74%, 93%]**, step-7 ≥ 60%, step-1 ≤ 88%.
> *Falsifier:* pooled < 74% (under-covers), pooled > 93% (blanket
> over-widening beyond the simulated schedule), step-7 < 60%, or
> step-1 > 88%.
>
> **H-G4 (disclosure names the power).** Every degraded-run artifact
> carries `selection_fold_count` and a `selection_underpowered` typed
> reason when the guardrail acted; `gnomon capabilities` states the
> guardrail's existence and default. *Falsifier:* any 30/7 artifact
> whose selection statistics appear without their n.

## Addendum: H-G5, registered after H-G1's falsification, before the fix

The first benchmark run (recorded in RESULTS.md) falsified H-G1: the
guarded run scored MSE 4.56 / MAPE 2.09% against thresholds 3.0 / 1.8%.
The isolating measurement found the cause **outside the simulated
mechanism**: the guardrail's point path scores MSE 2.995 ≈ the
simulation's 2.98, but the published q50 recentres every quantile on
the median of the pooled residuals — at 30/7 a location estimate from
14 selection-optimistic residuals, measured at mean |shift| 0.84
(≈ 1σ of the series' daily moves) in a coin-flip direction, making the
q50 path worse than the point path on 33 of 50 tasks. That is the same
defect class the guardrail exists for: an unevidenced location move at
fold-starved history, and E3's tilt geometry already showed ~1σ
coin-flip tilts destroy value.

The fix to be implemented after this registration: on **degraded runs
only**, quantiles are centred on the model's point path
(`point_bias_correction` becomes 0, disclosed); recentring is untouched
wherever separated folds exist. Predictions, frozen now:

> **H-G5a.** On the 50-task benchmark, the guarded run's q50 equals its
> point path on every task; filtered mean MSE lands in **[2.90, 3.05]**
> and mean MAPE **≤ 1.80%**, with 0 abstentions.
> *Falsifier:* any nonzero `point_bias_correction` on a degraded run,
> or MSE/MAPE outside those bounds.
>
> **H-G5b.** Task-level: every `last_value` selection is an exact tie
> with the raw floor; every non-tie traces to a `seasonal_naive`
> selection (a disclosed, fold-scored choice), with losses-vs-floor
> ≤ 10 of 50.
> *Falsifier:* a non-tie on a `last_value` task, or > 10 losses.
>
> **H-G5c.** Coverage stays inside H-G3's registered bands after the
> centring change: pooled in [74%, 93%], step-1 ≤ 88%, step-7 ≥ 60%.
> *Falsifier:* any band violated — in particular pooled < 74% would
> mean the recentring was load-bearing for coverage and the fix must be
> rethought (e.g. cap the shift rather than suppress it), not shipped.
>
> **H-G5d.** Series with ≥ 4 rolling origins remain byte-identical.
> *Falsifier:* any diff.

## Analysis plan

Identical to the exploration run: all 50 tasks, defaults, synthetic
daily axis, official metric block with the `mse > 100` filter reported
beside every mean; abstentions counted; per-task win/loss pairs beside
means. H-G2 is checked by the golden suite plus a direct pre/post
artifact byte-diff on a ≥ 4-origin series. RESULTS.md reports each
hypothesis against its frozen threshold, harm cases individually
(any task > 10% worse MSE than the raw floor), and nothing is
post-hoc filtered.
