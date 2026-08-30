# v0.7 loop I005: claim and support coherence

Decision: **promote and continue Q1 seasonal-admission validation.**

The exact current-head reproduction showed two general boundary defects. A
supported `historical_mean` baseline with zero measured improvement was called
“High-confidence” and said to have beaten itself by 0%. Separately, a degraded
30-row, 12-step threshold forecast with no final-test coverage published
`probability_any_breach: 0.671414` and
`breach_more_likely_than_not: true`, even while its reasons said the estimate
could not govern an action.

On exact commit `198ca4c`, all nine frozen ClaimBench gates pass. Baseline
retention is rendered as an evaluated baseline with no measured uplift. Every
sub-supported threshold result is routed to the existing bounded assessment:
`probability_status: unavailable_uncalibrated`, no probability vector or
horizon-event probability, and `automation_eligible: false`. Supported
probability-bearing forecasts retain their estimator and have a separate
runtime quantile-consistency test.

Summary Markdown, full tool output, and brief tool output agree exactly in all
four cases. The exact final isolated suite passed **2,609 tests with 11
skipped**. No forecast number, selected model, support status, or supported
threshold probability was changed by headline rendering.

Raw resumable checkpoints and exploratory runs remain preserved locally. The
external-evaluation intake remains the controlling issue map, and
`docs/astrid-btc-agent-plan.md` remains untracked and excluded.

