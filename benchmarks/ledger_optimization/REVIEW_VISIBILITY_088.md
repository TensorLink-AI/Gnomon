# Development088: recording visibility at the review boundary

Hypothesis, before synthetic execution: the existing catalogue filters forecast
origins and closed horizons, but not the execution's local recording time.
Although public compare_history filters scored evidence, later-recorded events
may change configuration discovery, pair ordering or global-origin windows of
an earlier review. This is not an established defect in published Gnomon.

Use the pinned published 1.2.0 runtime and synthetic providers only. Reuse087's
fresh synthetic fixture generator, then append two newly registered providers'
forecasts with past origins but recording times after the query. Query the same
origin with limit1 before and after. Preserve full evidence, ledger hashes,
execution envelopes, exact runtime identity, costs and both old/new outputs.
The existing evidence must remain immutable. No original030 database mutations,
API calls, numerical selection changes or protected data access are permitted.

If catalogue contamination reproduces, add a separate development boundary
filter using public ledger.execution to validate recording visibility and
execution identity before catalogue construction. Preserve the old adapter for
historical reproduction. Unknown/unverifiable execution references must fail
closed; they must not be silently trusted. Count all extra read calls.

Acceptance: previously visible cards, order and window membership are unchanged
after future-recorded insertions; hidden configuration names do not appear in
the brief view. Full operator evidence may describe excluded references, clearly
separated from usable query evidence. At the exact inclusive recording cutoff
an execution becomes visible, but ex-ante scoring eligibility remains governed
by compare_history. Test duplicate/tampered envelopes and task identities, and
that queries mutate no ledger and execute no providers. This is an infrastructure
gate, not evidence of accuracy improvement or permission for final confirmation.
