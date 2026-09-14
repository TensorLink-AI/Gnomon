# Conditional development continuation of guarded pilot 093

Recorded while the 36-session pilot is live. This is a prospective continuation
plan, not authorization to bypass its completion/integrity gate or a claim that
the dispatch implementation has passed verification. No extra run is launched
by this document. Pilot accuracy is not a promotion criterion.

## Purpose and fixed cohort

The pilot's first three origins cannot measure a long accumulation period.
If its frozen gate passes, continue the same four development series through
all 26 origins already in the pinned source manifest. That yields 104 matched
tasks and 312 sessions across plain Hermes, Hermes + Gnomon, and Hermes + Gnomon
+ ledger. Preserve the existing 36 pilot sessions and execute only the remaining
276 sessions. Do not rerun or replace any pilot failure or baseline-only result.

The source manifest remains SHA-256
`cf7bdd21e216e84809edb8653710e0c0755402864201400c5749a8dbff00f561`.
Its four series each span origins 2016-08-16 through 2017-08-01 with 730 history
observations and 14 future steps. This cohort is previously inspected development
data. It cannot establish the untouched-final objective.

## Gate and preservation requirements

Before continuation, independently verify all 36 sessions, at least 11/12 full
workflows in every arm, zero integrity failures, and the terminal identity of the
original controller and pilot processes. Verify runtime inventories and all
24 frozen source hashes against launch commit `26ceaff` and the accepted preflight.
A missing, malformed, or false gate refuses dispatch before loading credentials.

Create a separate continuation root. Keep the original pilot and its archive
immutable. Copy every original session plus each arm/series's terminal work and
Hermes home state, recording relative paths and SHA-256 hashes before and after
copying. Preserve the raw experiment logs, SQLite ledger, memory, text skills,
notes and checkpoint history. Do not reconstruct agent memory from a human
summary or inject these post-hoc analyses into agent prompts.

Reconstruct the host's prior-outcome list only from the preserved pilot grades
and matching frozen host jobs. The next origin may receive only fully matured
outcomes. No cross-arm histories, later targets, or scores from other experiments
may enter a workspace. Use each series's original index plus its absolute origin
number to retain the rotating arm order. Keep two series chains concurrent.

Reuse the exact frozen model-fitting, ledger-review, worker, tool schemas and
budget enforcement modules. Freeze and test any added host continuation code
separately; do not change the running v6 package. Test state-copy identity,
origin/arm order, chronological maturation, rejection before credential access,
and a synthetic resumed session in every arm before paid dispatch.

## Analysis and costs

Primary development estimates include all 104 scheduled cases, including the
pilot cases, under the existing fallback and workflow-completion rules. Report
the additional-origin subset separately and label its boundary; do not select
between the two summaries by whichever gives a larger improvement. Retain the
existing cold (origins 0–3), mature (>=10), and late (>=22) descriptions.

Report valid forecasts separately from full workflows, per-case RMSLE, matched
arm contrasts, numerical attempts, requests, tokens and unknown usage. Retain
the exploratory series-cluster uncertainty warning: four reused series and one
requested seed are not confirmatory. Native memory availability does not prove
usage; report observed memory/skill tool calls separately from ledger retrieval.

Copied pilot requests are references to already incurred costs, not new calls.
Report pilot cost, incremental continuation cost, and their deduplicated total.
Preserve failed admissions and interrupted sessions. No selective retries and
no accuracy-based early stop. Any integrity failure pauses new dispatch and is
retained with its consequences for interpretation.

The unchanged target remains >=20% lower mean per-case RMSLE versus the matched
no-ledger agent on untouched final data, with a paired 95% interval excluding
zero improvement. This development continuation does not open the reserved M5
or protected validation targets. Main and PyPI remain unchanged.
