# v0.7 loop I012: crash-safe agent-response economy

Decision: **retain the response-checklist candidate as a no-build, close Q5,
and continue to the final release gate.**

The frozen four-case ContextBench development shard completed in both cold and
warm receipt lanes with exact semantic, source, gate, relationship, primary,
and automation-boundary preservation. Each case used exactly one Gnomon
product call. The cold lane used nine final-agent requests and one repair-only
turn; the warm lane used ten requests and two repair-only turns, with no
compiler calls. Since eight final-agent requests is the two-turn-per-case lower
bound, even eliminating every repair could reduce requests by only 11.1% cold
and 20% warm. The frozen 25% request gate therefore cannot be met, and cold
prompt tokens cannot fall 25% by removing its sole repair. No production
checklist was implemented and the untouched seed-2026083103 confirmation
corpus was not consumed.

Crash safety was independently exercised. Resuming the completed warm run
executed zero rows and made zero model calls. A deterministic cross-invocation
fault test proves that only a failed row reruns, the completed row remains
byte-identical, append-only attempts remain `[1, 1, 2]`, cumulative usage is
three requests, and resumed usage counts only the one new request.
ContextCacheBench retained 120/120 forecast and artifact parity, 120/120
receipt and assessment hits, 90.9% compiler-request reduction, and 95.4%
context-argument byte reduction. Focused validation passed 183 tests.

Actual cold and warm final responses were inspected. All four preserved exact
validated source references after any bounded repair, rejection gate codes,
primary authority, context automation false, and applicable conditional
consequences. The structural-relationship case preserved the typed
`no_distinct_numeric_path` meaning. Raw rows, attempts, observations, receipts,
and cache artifacts remain in the local paths named by `orchestration.json`.
The external-evaluation intake is unchanged, and
`docs/astrid-btc-agent-plan.md` remains untracked and excluded.
