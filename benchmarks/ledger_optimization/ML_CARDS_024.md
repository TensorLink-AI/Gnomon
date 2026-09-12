# Dynamic-configuration incumbent adapter, before agent inference

This preparation does not launch a trial or change a finished experiment. The
incumbent is the published 1.1.9 build, not a deterministic selector relabeled
as an agent. Both ledger arms will execute the common numerical models, retain
every eligible production execution, and expose the same raw evidence. The
adapter only organizes public ledger evidence; it never fits a provider.

Preserve the incumbent's MAE cards for last-four, last-twelve and lifetime
production-origin windows. Windows use the global eligible past-origin sequence,
not the last N successes of a particular model. Within a comparison, use exactly
matched origins for both configurations. Do not require every configuration ever
tried to overlap: compare pairs and explicitly label each pair's own cohort.
Never combine pairwise ranks into a global ranking on unequal samples.

Provider name, declared revision and configuration identity remain bound. Only
prior production forecasts with closed horizons enter the catalog; historical
backtests stay separately available in the raw logs and current fold results.
The ledger applies actual source/recording cutoffs at the current task origin,
exact series/unit identity, history timestamps and ex-ante recording eligibility.
No scored execution or actual is invented to fill a missing comparison.

Default pair page size is 12. Order pairs by the latest shared production origin,
then canonical configuration IDs, never by observed errors or a hindsight rank.
Pairs without a shared origin follow pairs with overlap. Return the complete
configuration index, total pair count and exact next offset. Callers can request
the remaining pages or an explicit pair with the same normal agent time budget.
This shared retrieval arrangement applies to both ledger presentations; do not
restrict the incumbent to an arbitrarily unfavorable configuration subset.

For each selected pair, call public compare_history once with the lifetime
range and explicit cutoffs. Construct the three windows from its complete
matched-origin records, retaining execution/actual references and exclusions.
Verify against separate public per-window queries in both pinned runtimes.
Save full cards before printing a compact view: retain exact metrics, counts,
window definitions and exclusions counts, with file/JSON-pointer references to
each complete matched-origin and exclusion list. The agent can inspect those
references using its existing file/Python tools; no evidence is silently lost.
The current task's actuals and future records must have no effect. Public reads
must start zero provider calls and preserve the ledger bytes and stored scores.

The baseline output remains MAE, matching the frozen incumbent evidence design.
The eventual development presentation may add the objective-aligned RMSLE,
calculated within-cohort ranks/differences and recent/lifetime disagreement from
the same referenced executions. That treatment must be separately frozen and
tested before a matched paid comparison. Raw predictions and actuals remain
equally accessible to controls. No comparative performance claim follows from
adapter compatibility or these synthetic checks.
