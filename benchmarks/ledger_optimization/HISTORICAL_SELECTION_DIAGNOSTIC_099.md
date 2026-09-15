# Historical selection diagnostic 099 — development only

Freeze this diagnostic before computing its policy scores. It is a retrospective
analysis of already observed development sessions, not a prospective policy trial
or a final-evaluation result. No live worker, model, budget or outcome is changed.

Use every independently audited session in the 097 pilot and continuation
batches 001–003: 68 sessions. Report all-three-arm comparisons on their common
22 cases, and preserve the other sessions without mixing unequal case sets.
Do not select sessions using success or accuracy. The final holdout stays closed.

For each session, consider only configurations with complete current three-fold
backtests and a successfully collected current forecast before the submitted
checkpoint's selection. Use the exact executed predictions, not new fits. Extract
historical production outcomes only from that session's visible event history:
same series/unit/horizon/versioned configuration, origin strictly before the task,
complete target timestamps, and recording availability at or before the task.
Recompute historical and CV RMSLE from their scored pairs.

Evaluate these fixed rules, without tuning thresholds or picking a winning rule:

1. Current three-fold CV mean, with configuration-ID tie breaking.
2. Minimum historical mean over the last four global visible production origins.
3. Minimum historical mean over the last twelve global visible production origins.
4. Minimum historical mean over all visible production origins.

Each historical rule requires at least two origins shared by **every** currently
eligible configuration within its window. Otherwise it falls back to rule 1.
Use the same matched origins for all candidate means. Do not rank configurations
using different cohorts. Ties use configuration ID. Return evidence counts,
origins, selected identity, historical means and explicit fallback reason.

Write and hash every selection before loading host scoring targets. Then score
the frozen selected execution against the already completed task's host actuals.
Keep agent-submitted results alongside these rule results. Report coverage,
changed selections, wins/losses/ties and mean per-case RMSLE. No efficacy interval
or target claim from this partial reused development cohort.

This holds the agent's observed exploration and future history fixed. Different
earlier choices could change later agent behavior, so the replay is not a causal
estimate of deploying these rules. It can reveal whether straightforward use of
available matched history merits a prospective experiment or whether more work
is needed on exploration/context retrieval. Controls have the same raw facts;
compute the diagnostic separately within each arm, without pooling executions.

This diagnostic adds zero agent calls and zero fits. Existing run costs remain
unchanged; future scoring data must never be used as a selection input.
