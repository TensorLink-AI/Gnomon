# v0.6 loop I005: point-in-time champion/challenger routing

Decision: **promote.** Gnomon now converts paired realised adapter outcomes
into a bounded candidate-pool recommendation. It requires a pinned revision,
an exact temporal regime, eight outcomes, at least 5% mean relative gain, and a
95% Wilson win-rate lower bound above chance. A four-outcome recent window
rolls back to the champion on degradation.

The route is intentionally not a forecast decision. Every response says
`automatic_promotion=false`, `automation_eligible=false`,
`routing_authority=candidate_pool_only`, and
`job_local_admission_required=true`. The next forecast must still admit the
challenger against mandatory baselines. Agents get an executable pool; humans
get the cohort, cutoff, sample, evidence, decision, and rollback condition.

The frozen synthetic replay completed 60/60 prospective decisions. Stable
challenger performance reduced routed error 18% versus always-champion. After
drift, the policy rolled back after two harmful observations and finished 0.4
error units better than always-champion. A mixed control never routed, regimes
did not pool, future outcomes did not alter replay, and unpinned candidates
were blocked.

The final local suite passed 2,557 tests with 11 skips. Raw resumable artifacts
remain under `results/v06-p5-i005-*`. The competition-specific
`docs/astrid-btc-agent-plan.md` remains untracked and excluded.
