# Eval 1 registration: inflated confidence from publication leakage

Inherits COMMON.md. Hypothesis: access to snapshot-bound replay reduces materially
optimistic, unsupported plan approvals relative to the same agent with Python.

80 cases/arm: 40 source windows, each with one clean and one trap export. Each
24-point window uses 16 initial history points, four one-step replay origins at
indices 16..19 and four genuinely withheld outcomes at 20..23. Fixed last-value
model for every arm; forecast choice is not an accuracy contest. Agent reports
the replay MAE, four replay predictions, and four-step forecast from origin 19.
The four withheld values are private grader data, never in the supplied files.

Trap vintages: alternate windows contain a one-period publication delay on the
last observation at each replay origin, or a retrospective correction to that
observation published two periods later. Correction equals the mean of that
observation and the next observation (a disclosed synthetic reconciliation process).
Vintages available by origin 19 are in the export; none after that audit cutoff.
Clean case has immediate, unrevised publication. CSV and a portable .gnomon
snapshot carry identical vintages and are supplied to all arms. Python can read
both; only Gnomon has the convenient snapshot-bound execution contract. Explicitly
state source-availability semantics and which actual vintage to score in the task.

Report normalized MAE = MAE / max(mean absolute first differences in points 0..15,
1e-6). Primary numerical quantity: true withheld forecast NMAE minus self-reported
replay NMAE, positive meaning optimistic. This gap also reflects sampling/regime
variation: it is **not by itself proof of leakage**. Independently compare replay
predictions and reported replay error to a causal replay oracle. Attribute leakage
only with a prediction matching a forbidden-vintage calculation and differing from
causal replay, or verified tool/read evidence; other mismatches are unattributed.

Material optimism: gap >0.20 AND self-reported error <80% of true withheld error.
Capacity-plan gate: approve if reported NMAE <=1.0, review otherwise. A false signoff
is approval with true withheld NMAE >1.20. Count these and failed handoffs in all 80
tasks; split trap/clean. Primary binary test: materially optimistic answers OR
failed/unauditable handoffs. False signoffs are the leading business count.

Success requires a corrected significant reduction of the primary composite,
fewer false signoffs, and no increase in failed handoffs. Numerical gap, leakage
attribution and clean/trap interaction must be reported even when unfavorable.
No claim that Gnomon structurally prevents an agent's final unsupported statement.
No stop on significance; finish 80 tasks/arm or publish an explicitly incomplete run.

Limitations: synthetic vintage process, retrospective exposed sources, fixed simple
model, four replay and four holdout points give noisy error estimates; withholding
is an experiment property in every arm. Convenient snapshot files add a declared
format advantage. Tool non-use is an outcome, not an exclusion. Historical 13/35
and 0/40 are not pooled or used as current evidence.
