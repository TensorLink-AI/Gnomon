# Prospective development diagnostic: persistent comparison cohort

Freeze this diagnostic before calculating its forecasts. It uses only the same
104 completed development tasks, identified by SHA-256
`380261912295bbe21f3c998a8c6f7048cde1fb0869d37bb4db44ad3a72b28f25`.
No new series, final targets or agent calls are authorized by this diagnostic.

The infrastructure question is whether maintaining comparable outcomes for
recurring configurations gives a past-only selection rule useful evidence.
Keep three versioned recipes in the cohort: the existing numerical lab's default
seasonal (7), Ridge (window 365, lags 14, alpha 10), and random forest (window 365,
lags 14, depth 6). These are constructor defaults, not recipes selected by their
development scores. Pin the numerical implementation to SHA-256
`a71b75f06ad69fcca06275046c5e74565b15d7d6dd2538df0fe827c448fc0413`.

At every origin, independently execute each recipe on the original three rolling
folds and the production horizon. Count all **12 fits per origin**, including
unselected production forecasts. In any subsequent agent experiment this common
initial cohort must be identical in all arms and consume the existing 60-fit
budget. It must not provide free ledger-only exploration or change model recipes.

Compare these fixed diagnostic selectors:

1. Current CV: lowest mean three-fold RMSLE; exact ties follow the declared order
   seasonal, Ridge, random forest.
2. Past evidence: once at least three complete, commonly matched prior origins
   have matured, lowest mean RMSLE over the most recent four such origins.
   Exact historical ties use current CV, then declared order. Before sufficient
   history exists, use current CV. No threshold search or variant selection.
3. Hindsight: best current production forecast, explicitly non-deployable.

Finish both deployable selections before reading current targets for scoring.
Prior outcomes must have last target and source/recorded availability at or before
the current origin. Perturb future outcomes to check that they cannot change a
past selection. Preserve all cold starts and failed fits. A failed fit invalidates
this diagnostic, is recorded, and is not silently deleted or retried with another
recipe. Save all predictions, requests/fold endpoints, selections, metrics, fit
counts, source hashes and elapsed time in a fresh result directory.

This is an offline mechanism screen, not evidence that an agent or ledger API
outperformed a control. It deliberately holds the configuration cohort stable;
the completed agent run allowed arbitrary configurations and is not directly
comparable. A subsequent agent run requires a separate prospective protocol and
the existing development spending gates. If hindsight cannot reach the unchanged
20% target here, do not tune selectors repeatedly on this cohort. Main and PyPI
remain unchanged, using 1.2.0 for any later Gnomon comparison.
