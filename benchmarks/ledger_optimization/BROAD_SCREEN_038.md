# Frozen hourly development screen 038

Run only the 16 development spans prepared in 037, SHA-256
`60c8db1de333290d7cc04ca3a52dc4c4d46d8b93867105c5cb9be03adb8de15f`.
Do not read the raw archives or reserved future values. This is a numerical
mechanism screen, not an agent comparison, a Gnomon treatment effect, or final
evidence for the 20% objective. Keep all previous negative screens.

For every series, forecast 24 hours at each of 26 weekly origins. Use the last
730 observations and three backtest folds ending at history positions 658,
682 and 706. Backtest predictions use only observations before their origin.
Calendar features use nominal source-hour labels; availability/recording remain
assumed at period end as in 037. No actual vintage or DST claim is made.

## Common fixed recipes and costs

All selectors receive identical complete cohorts, current CV errors and the
same arrived historical outcomes. Previous outcomes could also be calculated
by a no-ledger integration: this screen tests whether persistent matched evidence
is useful, not whether ledger storage uniquely creates information.

Freeze these six recipes before any performance calculation:

1. Daily seasonal naive, period 24.
2. Weekly seasonal naive, period 168.
3. Average of the same hour in the last three weeks, arithmetic count mean
   (all three are available even at the earliest CV origin).
4. Ridge, window 336, contiguous lags 48, alpha 10.
5. Ridge, window 730, contiguous lags 168, alpha 10.
6. Random forest, window 730, contiguous lags 48, depth 6, 80 trees,
   minimum leaf 3, seed 17, one job.

Regressions predict log1p counts recursively, standardized for Ridge. Features
are the contiguous lag vector, previous up-to-24/168-hour log means and source hour/
weekday sine/cosine values. Retain the previously disclosed guard: predictions
clipped to [0, log1p(max(100 * max(training counts), 1000))] before expm1.
Calendar variables are known, not observed future targets. These explicit
hourly recipes replace daily-unit defaults prospectively; there is no tuning
against results from this screen.

Each case charges all six recipes at three CV origins plus production: 24
forecast computations, including deterministic baselines. Total 9,984;
4,992 are estimator fits, the rest deterministic baseline computations. Equal
cost for each selector, within the earlier 60-fit per-task ceiling. Count cohort
maintenance, even when the selected recipe was already obvious. Save wall time,
CPU time, versions, source hashes and errors. API calls and LLM tokens are zero.

## Frozen selectors

- Control: lowest mean per-fold CV RMSLE; ties use recipe order above.
- Primary historical selector: lowest mean RMSLE over the latest four complete
  same-series production origins whose horizons closed and outcomes were recorded
  by now; require at least three. Otherwise use current CV. Ties use current CV,
  then recipe order. This is the same rule used in screen 032.
- Secondary diagnostic: minimize the equal-weight mean of current CV RMSLE and
  the historical mean above, using the same minimum history and tie rules.
  Do not replace the primary result with this diagnostic after seeing results.
- Hindsight diagnostic: lowest current actual RMSLE among the six executed
  recipes. It is nondeployable, not an estimate of ledger performance or a bound
  over all possible models.

Select before exposing current production targets to the selector. Do not pool
cross-series history. Report overall, per source and round >=10 means for all
selectors; equal numbers of cases give equal source weights. Report regret and
selection changes. Do not drop failed cases or silently resume overwritten runs.

## Decision gate

This screen can reject an unsuitable recipe/task portfolio. Passing requires
the primary historical selector to reduce mean RMSLE at least 20%, with positive
improvement in each source. Even a pass only authorizes developing/testing the
agent integration; it does not authorize a final superiority claim. Before
opening final outcomes, freeze the agent comparison, common budgets and an
uncertainty method respecting correlated assets and common calendar shocks.
Do not calculate a misleading independent-case confidence interval here.

If this screen fails, preserve all recipe errors and reconsider the mechanism
using development evidence only. Do not launch a paid confirmation merely to
chase a lucky seed. Future Gnomon comparisons use 1.2.0 and Engy
`deepseek-v4.1-flash`; main/PyPI remain unchanged.
