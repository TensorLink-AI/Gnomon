# Context support in the current ML ledger — diagnostic 103

Freeze this diagnostic before computing its counts. Use only the 30 matched
candidate-100 cases retained in history-progress-002 and selection-coverage-001
(pilot plus audited continuation batches 001–003). Later completed cases are
excluded by this fixed source boundary, not by their scores.

For each latest displayed configuration-pair comparison, use the returned
lifetime matched-origin evidence and its exact versioned executions. Read their
origin-bounded requests from authenticated, immutable SQLite copies. Reconstruct
current context from the current task's production request. Require the two
historical requests to agree on context, series, unit, horizon and future dates.
Require full historical horizons to end by the current origin. Preserve the
upstream source/recording visibility contract; no new pages or actual revisions.

Reuse the four labels and filter order from CONTEXT_PROTOCOL.md: last-28 zero
fraction, adjacent 14-observation trend, last-28 coefficient of variation, and
known future promotion plan. The labels describe observations, not stockouts or
causes. Missing promotion remains unknown. Do not apply the Favorita promotion
assumption to M5's unavailable promotion channel.

Report every nested filter's matched-origin count, mean per-origin RMSLE, ties,
and disagreement with unfiltered history/current CV. Select the first filter
with at least four matched origins, using counts alone. This is the old fixed
eligibility rule, not statistical confidence. Retain insufficient pairs/cold
cases and unreturned-page status. Never globally rank recipes scored on different
cohorts. Counts repeat evidence across pairs/sessions and are not independent.

This tests whether context filtering has usable support and changes the evidence
an agent could see. It does not choose a forecast, score a counterfactual policy,
establish that the agent read the comparison, or measure accuracy improvement.
Latest displayed comparisons can postdate an earlier selection. Read no grade
files, host targets or reserved final data. Make no provider/Engy calls, fits,
ledger mutations or changes to the live run. Preserve sources, hashes, all
failures and a compact receipt. Keep the prototype undeployed pending results.

## Result: supported filtering adds no different winner in this snapshot

Protocol freeze `d6663478` preceded the diagnostic counts. All 30 fixed sessions
were retained, including four cold starts. Their latest displayed comparisons
contain 56 pairs: 43 with a returned historical card, nine with no historical
contrast, and four whose configurations were absent from the historical catalog.
These last two statuses remain unknown; no missing evidence page was fetched.

Of the 43 returned pairs, 22 have at least four unfiltered origins. The fixed
count-only retrieval chooses all four labels for two pairs, sparsity/trend for
two, sparsity alone for ten, and unfiltered history for eight. The other 21
pairs remain insufficient. Thus 14 pairs across ten sessions support a filtered
cohort. **No eligible selected cohort changes the lowest-error provider(s)
relative to unfiltered history.** Four eligible selected cohorts disagree with
current CV, but unfiltered history already contains that disagreement.

This is support/disagreement evidence, not a forecast-accuracy experiment. It
does not rule out other context definitions, later history or another dataset.
It supplies no reason to deploy this filter as a way to change current model
choices. Keep it undeployed; do not spend a new agent trial solely on this
unchanged ranking or describe finer filters as beneficial without evidence.

The new pure renderer passed six tests covering maturity, query visibility,
task/provider revisions, exact shared contexts, empty cohorts, ties, count-only
eligibility and input preservation. Across real requests, 378 context checks
agree with the original four-label implementation. Independent reconstruction
of counts, means, winners and input hashes passed 1,552 checks. All 30 source
databases remained byte-identical. No provider calls, fits or Engy calls ran.

The first driver failed before counts because it expected hashes directly in a
receipt that instead references audit-003. Original driver/error/exit status are
retained; the corrected driver authenticates against that committed audit
receipt. Raw evidence, reproduction scripts and source hashes live under
`results/context-support-103-offline-001`; its compact receipt is
`evidence/context-support-103-offline-001.json`. Current experiment 100 and
main/PyPI were not changed; final targets remain unopened.
