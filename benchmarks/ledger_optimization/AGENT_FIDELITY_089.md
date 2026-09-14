# Development089: prospective agent evidence-reading comparison

This is an auxiliary interface experiment, not an accuracy trial, a Hermes ML
workflow, a new agent forecasting result or final confirmation. The original030
forecast-selection hindsight ceiling is only2.2%; do not spend on reselecting
those forecasts to pursue a20%claim. Test the narrower unresolved087hypothesis:
does the smaller faithful review actually reduce model tokens without increasing
evidence-reading errors? A positive result permits using the format in a future
prospectively frozen development workflow, not bypassing accuracy gates.

Before API inference, hash/freeze code, all input packets, expected answers and
scheduled calls. Use only the already-inspected030development reviews with
original inventorySHA722c105f0466ff56a87d9a90776739a4832871abdd50ac73700da2ae7a42902c.
Take every one of the four series at rounds1,9,17,25, without score-based
selection. Round0has no saved full review; this is a16review interface cohort,
not a forecast cohort that removes cold-start failures.

Compare the existing compact_cards response with087brief on exactly the same
full query. Use two fixed seeds7and19 and modeldeepseek-v4.1-flash through Engy.
This is32matched review/seed pairs,64independent sessions. Randomize dispatch
order with fixed shuffle17, single worker. No history or previous answer is
shared between sessions; both formats have identical questions, facts, tools,
temperature0.2, max_tokens2048 and at most2upstream requests. One correction is
allowed only for malformed submission syntax; never supply the expected facts.
No provider tools, forecast computations, source mutations or new ledger writes.

Ask for the first and last returned pair (deduplicated), last4andlifetime matched
origin counts, n, date bounds, both RMSLEscores, complete exact-tie winner sets,
recent/lifetime disagreement; and page coverage/next offset, provider calls and
whether a global ranking is supported. Require submit_evidence with a typed
object. No prose scraping or silent alias repair. A schema-valid wrong answer
is a factual failure and does not receive a correctness hint. Different pairs'
cohorts never justify a global ranking. No forecast selection is scored here.

Primary interface endpoint: fraction of sessions with every requested fact
correct. Also report field accuracy, schema completion, syntax corrections,
per-question errors, reported prompt/completion tokens, latency and service
errors. Score IDs/counts/ties/booleans exactly and numerical RMSLE within1e-9
absolute tolerance. Missing answers count as unsuccessful interface completion;
service failures are separately labeled. Do not treat retry requests as free.
Use paired review/seed differences and descriptive per-review grouping; no
claim about a population based on four reused series. Actual usage is required
for token comparisons; bytes are not a substitute, and unknown cost is not zero.

Stop new calls after3consecutive service failures. Save each request before
sending, each response/error immediately, and a terminal session result.
At most128upstream requests; no readiness calls or automatic run restart.
A timeout is a charged attempt, not permission to restart the experiment.
Resume only pending jobs with no request-start evidence; uncertain in-flight
requests require reconciliation and cannot be silently retried.

Adoption screen, not statistical superiority: all64sessions terminal, no source
or scoring-integrity failures, brief all-facts-correct rate at least the original
rate, and at least25%lower paired mean prompt-token use with usage present for
every session. Preserve negative results without changing cases/prompts/seeds.
No numeric accuracy threshold is tested here. Target20%/95%on untouched final
data remains unchanged; final data stay closed and main/PyPI unchanged.
