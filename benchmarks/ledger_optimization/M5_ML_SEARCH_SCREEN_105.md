# Fixed M5 ML search screen — development diagnostic 105

Freeze this protocol before running any of its numerical forecasts. Diagnostic
104 found less than 8% hindsight headroom among the configurations explored in
the audited Favorita subset. Historical M5 screen 015 used a fixed StatsForecast
menu and different history/CV construction. Neither answers how much opportunity
exists in the current tunable ML task. This screen uses the already frozen M5
development jobs and current numerical implementation; it does not select new
cases or change the live agent workflow.

## Fixed inputs and model grid

Use all eight development series, all 26 origins, 730 observations, horizon 14,
and current CV endpoints 688, 702 and 716. Authenticate development jobs SHA-256
`dd608a2a0188cbe7e0e684127c4e5abbc77019982fdd14aabbb6e635064c164b`
and the original panel manifest through `m5_ml_development_contract`. Read no
sales archive or reserved numerical values. Use the unmodified `numerical.py`
identified by the seed-7 frozen M5 capsule, under its verified numerical runtime.
Keep its RF seed 17, clipping, features and preprocessing. No Gnomon version or
provider implementation change is permitted.

The eleven configurations, ordered for exact tie resolution, are:

1. Seasonal, period 7.
2. Seasonal, period 14.
3. Seasonal, period 28.
4. Ridge, window 90, lags 7, alpha 10.
5. Ridge, window 90, lags 28, alpha 100.
6. Ridge, window 365, lags 14, alpha 10.
7. Ridge, window 365, lags 28, alpha 10.
8. Ridge, window 730, lags 28, alpha 1.
9. Ridge, window 730, lags 56, alpha 1000.
10. Random forest, window 365, lags 14, depth 4.
11. Random forest, window 730, lags 28, depth 8.

Each task has 44 forecast calculations: three CV predictions and one production
prediction per configuration. This is below the existing 60 numerical-attempt
budget, but is a fixed numerical grid, not evidence that an agent can discover
and execute it within the interaction/time limits. Retain every attempt, elapsed
time and failure; a failure does not license replacing a configuration or series.
Use at most two CPU workers and single-thread numerical libraries. No Engy calls.

## Fixed descriptive comparisons

Recompute RMSLE from saved point/actual pairs. Compare these rules on all 208
cases, retaining all cold starts:

- Current CV: lowest arithmetic mean RMSLE across the three current folds.
- Recent history: lowest mean production RMSLE over the previous four fully
  matured origins; use current CV until four origins have matured.
- Lifetime history: lowest mean production RMSLE over all prior fully matured
  origins; use current CV until four origins have matured.
- Lifetime support: override current CV with the lifetime leader only after
  four matured origins, strictly lower mean historical RMSLE and at least half
  paired wins against the current CV choice. Otherwise retain current CV.
- Hindsight minimum: lowest production RMSLE among the eleven configurations.
  This last row is explicitly non-deployable and cannot support a ledger claim.

All historical comparisons use the same complete configuration cohort and the
task's source/recording visibility cutoffs. Production targets may be scored
only after predictions and choices have been fixed. They must never enter a
model request, current CV calculation or same-origin historical policy choice.
Period-end availability remains an assumption, not an observed publication
vintage. Unavailable promotion data remains explicitly unavailable.

Report per-case, per-series, per-origin and overall arithmetic means, cold
origins 0–3 versus subsequent origins, prediction uniqueness, failures, calls,
actual estimator fits and elapsed time. Do not tune these rules, grid or labels
after seeing results. Independently verify saved requests, predictions, metrics,
choice indices, temporal eligibility and source hashes without refitting.

## Interpretation and unchanged gates

This is a reused-development opportunity diagnostic, not an agent comparison,
an API evaluation, a new paid arm or an estimate of causal ledger value. A
hindsight gain shows only opportunity; it cannot prove that an agent or past-only
rule can realize it. The fixed current-CV comparator is also not the actual
no-ledger agent. Do not promote a rule by retrospectively choosing its best
series/window or infer that a weaker control would make 20% persuasive.

This screen does not change the frozen M5 paid development plan, candidate 100,
the final cohort, or the 20%/95% target. Any later candidate change requires its
own prospective freeze and fair development comparison. Preserve all scripts,
failures and costs on the development branch; keep main and PyPI unchanged.
