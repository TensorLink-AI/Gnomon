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

## Analysis plan

Identical to the exploration run: all 50 tasks, defaults, synthetic
daily axis, official metric block with the `mse > 100` filter reported
beside every mean; abstentions counted; per-task win/loss pairs beside
means. H-G2 is checked by the golden suite plus a direct pre/post
artifact byte-diff on a ≥ 4-origin series. RESULTS.md reports each
hypothesis against its frozen threshold, harm cases individually
(any task > 10% worse MSE than the raw floor), and nothing is
post-hoc filtered.
