# Results: short-history guardrail (against HYPOTHESIS.md)

Run 2026-08-03, same session as the registrations. Fixed benchmark: the
50-task S&P surrogate from `results/news-regime-explore/` (30 daily
bars in / 7 out, seed 20260803), official metric block, no
official-filter failures, no abstentions under any condition. The build
converged over three registered iterations, each falsification
explained and the fix registered before implementation. Final state:

| | mean MSE | mean MAPE | pooled q10–q90 coverage |
|---|---|---|---|
| `last_value` floor | 2.56 | 1.61% | — |
| gnomon-pure **before** (commit `fe39738`) | 7.42 | 2.37% | 63.7% (per-step 82→44%) |
| gnomon-pure **after** (this build) | **2.995** | **1.71%** | **79.1%** vs 80% nominal (per-step 92, 88, 78, 80, 68, 78, 70) |

## Hypothesis outcomes

- **H-G1 — FALSIFIED, cause isolated.** The first implementation
  (guardrail + √step widening) scored MSE 4.56 / MAPE 2.09% against
  thresholds 3.0 / 1.8%. The guardrail itself reproduced the simulation
  (point-path MSE 2.995 vs simulated 2.98); the miss was entirely the
  median-residual recentring — mean |shift| 0.84 ≈ 1σ of the series'
  daily moves, in a coin-flip direction, q50 worse than the point path
  on 33 of 50 tasks. Exactly the defect class E3's tilt geometry
  predicts is destructive. The "wins-vs-floor ≥ 16" clause was also
  missed (15); those wins were recentring wobble, not skill, and
  disappear with it (final run: 1 win / 7 losses / 42 exact ties — see
  H-G5b).
- **H-G2 — HOLDS.** Direct pre/post byte-diff of a ≥ 4-origin artifact:
  identical bytes, identical forecast id. The `two_series_h5` golden is
  unchanged since before the build and byte-checked on Python 3.12.
- **H-G3 — HOLDS for the mechanism it registered, which was then
  removed on evidence.** The √step-anchored-1 widening measured 92%
  pooled on the first run (inside [74%, 93%]). After the recentring fix
  it measured 94.9% (H-G5c falsified, over the 93% cap), and the
  attribution became clean: the original 82→44% decay was mostly
  recentring wobble compounding with lead, and whole-horizon pooled
  residuals already contain multi-step dispersion, so any lead-growth
  multiplier double-counts (the caveat `interval_bounds`' docstring
  always carried). The widening was removed under H-G6.
- **H-G4 — HOLDS.** Degraded artifacts carry `selection_fold_count` in
  sensitivity and the `selection_underpowered` typed reason when the
  guardrail acts; `gnomon capabilities` states both behaviors under
  `short_history` (pinned by `tests/test_short_history_guardrail.py`).
- **H-G5a/b — HOLD.** With recentring suppressed on degraded runs:
  MSE 2.995 ∈ [2.90, 3.05], MAPE 1.71% ≤ 1.80%, 0 abstentions,
  `point_bias_correction` 0 on every row; all 8 non-ties are the 8
  `seasonal_naive` selections (1 win, 7 losses), every `last_value`
  task an exact tie with the floor.
- **H-G5c — FALSIFIED** (pooled 94.9% > 93%; step-1 92% > 88%), on the
  conservative side. Explained above; led to H-G6.
- **H-G5d / H-G6c / H-G7c — HOLD** (same byte-identity evidence as
  H-G2, re-verified after each iteration via the golden suite).
- **H-G6a — HOLDS.** Unwidened point-centred band: pooled 79.1% ∈
  [74%, 88%], minimum per-step 68% ≥ 60%. One deviation from the
  registered fix description: the double-count rationale went into the
  code comment beside `constant_interval_width` rather than into the
  disclosure message itself, because that message is shared with
  non-degraded mid-size series whose byte-identity H-G6c protects.
- **H-G6b — HOLDS.** Point-path metrics identical (2.995 / 1.71%).
- **H-G7a — HOLDS.** With the hard lock replaced by
  `SINGLE_FOLD_SELECTION_MARGIN = 0.75`: zero non-baseline selections
  on the benchmark, all other numbers unchanged.
- **H-G7b — HOLDS.** `daily_requests` (perfect +3/day trend, 3
  origins) selects `drift` again with measured test coverage 1.0, and
  `test_contracts_v2::test_forecast_emits_verified_lineage` passes
  unmodified. The artifact carries the new "Single-fold selection"
  warning naming the cleared margin.

## What shipped (all confined to degraded runs)

1. **Fold-count-scaled selection margin**: below 2 disjoint selection
   folds the margin rises to 75% single-fold improvement — measured to
   admit zero spurious wins on 50 near-martingales while a plain linear
   trend clears it easily. Dense (overlapping) origins do not lift the
   gate. The lightweight single-trailing-holdout path keeps a hard
   baseline lock (its holdout can be one observation). Candidate scores
   always survive in the artifact as evidence.
2. **Point-centred intervals**: the median-residual recentring is
   suppressed (`point_bias_correction` = 0, disclosed as
   `point_recentring_suppressed`); no lead-growth multiplier on the
   borrowed band. Coverage moved from 63.7% to 79.1% against 80%
   nominal from these two changes alone.
3. **Disclosure**: `selection_underpowered` typed reason,
   `selection_fold_count` in sensitivity, single-fold-selection warning
   when the escape hatch fires, capabilities entries.

Runs with ≥ 4 rolling origins are byte-identical throughout. The three
degraded goldens were deliberately refreshed (Python 3.12) and reviewed
at each iteration.

## Files

- `summary.json` — final benchmark aggregates (versioned).
- `raw/` — final per-task records (untracked, regenerable).
- History: first-iteration numbers are quoted above and in the
  HYPOTHESIS.md addenda; each iteration's registration precedes its
  run in the git history of this directory.
