# Development error-estimate calibration 039

Screen 038's recent-history selector was 4.98% worse than current CV. Freeze one
follow-up rule before computing its results. This uses the same already-seen
416 development cases; neither the later slice nor the complete panel is a new
holdout. Related calibration/retrieval rules already failed on the older retail
development set. This is a test on the broader panel, not a novel proven method.

For each current series/origin, consider only earlier same-series origins with
complete six-recipe CV and production scores, with production horizons closed
and outcomes recorded by the current origin. Retrieve the latest eight. Require
at least four; otherwise retain current-CV selection.

For recipe m, estimate its CV bias as the mean of
`production_RMSLE(m) - contemporaneous_CV_RMSLE(m)` over those origins. Shrink
the correction toward zero with weight `n / (n + 4)`; four zero-bias prior
observations reduce small-sample overreaction. Estimate current production loss
as `max(0, current_CV_RMSLE(m) + weight * bias(m))`. Select its minimum, breaking
ties by current CV then the original recipe order. No context, window, shrinkage,
threshold or recipe is selected using the resulting scores.

This corrects backtest estimates using accumulated prediction errors instead of
substituting old absolute error levels for current evidence. Forecasts, inputs,
underlying fitting costs and all original selections remain immutable. Expose
the retrieved origins, bias, shrinkage and corrected estimates on every case.
It is an integration candidate for evidence presentation, not a shipped ledger
change and not unique information unavailable to the control.

Read only files pinned by the 038 receipt. Select before passing current actual
scores to reporting. Save all cases, overall and per-domain metrics, rounds
0–17 and 18–25 (both development), and round >=10 diagnostics. Also summarize
038 historical overrides: counts helping/hurting/tying current CV, sum of gains
and losses, and per-recipe CV versus production error estimates. These diagnostic
strata must not be used to substitute a winning variant after execution.

Promotion gate remains at least 20% overall reduction against current CV and
positive reduction in both domains. No significance claim from this reused
development set. No new forecasts, estimator fits, API calls or reserved future
reads. Charge the already-incurred cohort construction costs and record this
analysis's additional runtime. Preserve failures and all earlier negative work.
