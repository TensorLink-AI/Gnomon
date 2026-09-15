# Historical support at the final selection — diagnostic 101

This is a read-only development diagnostic, frozen before calculating its counts.
It changes neither the running 097 experiment nor the staged candidate 100.

Use every ledger session in the independently audited 097 pilot and incremental
batches 001–008: 67 sessions. Preserve cold starts and all available sessions;
do not select using outcomes, successful completion, or favorable comparisons.
The source set is fixed even while later sessions finish.

At each session's final selected checkpoint, reconstruct the completed current
three-fold configurations from its experiment-log prefix. Independently recompute
their RMSLE from the logged CV pairs. This is the set of completed tests, not a
claim that the agent read every fold or used it in its reasoning.

Read only successful review responses delivered before that exact selection in
the boundary log. Authenticate their full-evidence hashes and validate their
pair/window aggregates using the existing paired-consistency validator. Full
files verify the summaries; reading them here does not imply the agent fetched
their detailed contents. Union the pairs actually returned across pages, reject
contradictory duplicates, and retain the last review's catalog and pagination.
Unreturned pairs are unknown, not zero history. Do not query additional pages.

For each pair of completed current configurations, report whether it was in a
returned page, whether both identities were in the latest returned catalog,
and its last-four, last-twelve and lifetime matched-origin counts when known.
Report newest/oldest origin ages, current-CV versus historical winner disagreement,
and whether a disagreeing winner rests on one origin or at least two origins.
Two origins is a descriptive count threshold, not a significance or independence
claim. Compare winners only within their respective cohorts; do not compare CV
and production error magnitudes directly.

Aggregate support both per pair and per session, keeping denominators explicit.
Include selected-configuration pairs separately. Report reviewed-page coverage,
new identities absent from the catalog, and cold starts. This distinguishes absent
matching evidence from evidence not retrieved; it cannot establish agent misuse,
causal benefit, or which model would have forecast better.

Do not read grade files, host targets, final/protected data, or future task scores.
No counterfactual scoring, model fits, API requests, new ledger query, selection
policy, or accuracy-based candidate amendment is permitted. Hash inputs before
and after the scan. Retain the plan, driver, raw pair/session counts, all failures,
and summarized findings. Prior run costs are unchanged.
