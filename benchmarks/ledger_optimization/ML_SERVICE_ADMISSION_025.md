# Common service admission, prospective development policy

Pilot v3 remains failed: no cases are regraded or retried. Six incomplete sessions
received only service errors, and two received one response before service errors.
Before a fresh run, add this identical admission step for every arm/session.

Before starting the agent, its 480-second task clock, or any task numerical work,
send one fixed, task-free canary to the same Engy model. A successful response
must be an actual completion object with text or reasoning content and no error.
HTTP 200 error envelopes are errors. The canary receives no series, observations,
candidate scores, memory, arm prompt or future target. All arms use the same
payload and settings; its text is not shown to the agent.

Only known service errors (HTTP 408/429/5xx, upstream/rate-limit error categories)
permit waiting. Retry at a fixed 60-second interval, at most ten probes, with
30-second HTTP timeouts. Unknown contract/authorization failures stop admission.
Exhaustion stops the trial with an explicit infrastructure-incomplete state;
unstarted sessions are not declared successful or silently dropped. Preserve
completed sessions and all probes. A new start requires a separately recorded
decision; this policy is not an automatic reroll of failed agent sessions.

Record every probe's exact task-free payload, redacted full response, HTTP status,
elapsed time, token usage when supplied and decision. Probe costs are part of
total experiment cost, separately labeled from agent requests. Missing billing
is unknown. Never include real authorization headers or keys in evidence.

Admission checks are serialized by the runner before each session. After a
healthy admission, keep the existing common agent rules: 16 forwarded requests,
60 numerical attempts, 480 seconds, two bounded task corrections, fixed fallback.
No within-session service failure gets free requests, a reset clock, or a rerun.
Existing valid checkpoints survive later service failures. A canary cannot
guarantee continued availability or represent the latency of a long-context
request. Report all such failures and preserve intent-to-treat scoring.

This change targets avoidable starts during a known outage. It does not increase
the ledger arm's forecasting information, model capabilities or execution budget.
Completion/integrity gates remain separate from the requested final accuracy
and uncertainty target. Freeze integration and test it before paid dispatch.
