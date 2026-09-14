# Diagnostic075: can the frozen strength family reach the goal?

Freeze this diagnostic before computing its summaries. Read only complete audited
074development artifacts. No new forecasts, fits, searches, API calls, policy
selection or validation/final access. This is descriptive analysis of previously
observed development outcomes, never a deployable decision rule.

For each of416scored tasks, independently recompute RMSLE for all five recorded
ledger candidate outputs. Preserve candidate IDs, source file hashes, historical
risk means and originally selected strength. Report every fixed-strength mean,
adaptive selected mean, fixed.5incumbent mean, matched-control and strong050mean.
Compute the per-case minimum over the five already executed candidates and its
mean. This perfect-hindsight finite-set bound is not attainable forecast skill.
If even this minimum cannot reach20%over the controls, no selector restricted to
these exact candidate outputs can reach that target on this development cohort.
It is not a bound on other datasets, new fits or interpolation between outputs.

Measure transfer without inventing a new policy: for each of the ten unordered
strength pairs, compare signs of historical risk difference and realized current
risk difference. Use absolute1e-12tie tolerance for diagnostic counts only; the
frozen074selector keeps exact ties. Report concordant, discordant, historical-only
tie, realized-only tie and both-tied counts, plus concordance among pairs untied
on both sides. Comparisons within a case are dependent; no significance claim.
For each case, recompute the historical selected-versus-.5advantage, realized
selected-versus-.5advantage, selected-to-hindsight regret and .5-to-hindsight
regret. Report means and realized improved/worse/tied counts. Negative values
are retained. Historical advantage is not an expected causal gain estimate.

Report whole cohort, both domains, early0-7/later8-25, and strength-choice groups
with denominators. No tuning of thresholds, neighbor rules or selector from this
diagnostic. Separate script tests on synthetic scores and independent recheck
of source hashes, candidate risks, all pair classifications and aggregate bounds.
Keep all costs and failures. Main/PyPI and untouched reserves unchanged.
