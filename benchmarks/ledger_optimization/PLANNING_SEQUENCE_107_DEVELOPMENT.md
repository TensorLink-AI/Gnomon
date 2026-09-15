# Prospective development comparison: candidate 107

This protocol is frozen before any paid candidate-107 requests. It defines the
next development comparison, not a final test or permission to bypass launch
checks. Candidate 100 remains unchanged while it finishes. Its original failure
records, if any, must be retained and independently reconciled before admitting
a successor on the same host.

## Treatment and matched controls

Use three arms: Hermes alone, Hermes with published Gnomon 1.2.0 without ledger,
and Hermes with Gnomon 1.2.0 plus development ledger evidence and candidate-107
recipe plans. Keep the candidate-100 common CV table and evidence comparison.
The sole agent-visible treatment addition is the optional historical recipe
plan validated in `PLANNING_SEQUENCE_107_WORKER_RESULTS.md`.

Keep DeepSeek v4.1 Flash on Engy, temperature 0.2, seed 7 and 3,072 output tokens.
Keep the existing 16-request/60-numerical-attempt limits, exploration/selection
reserve, 480-second agent and 520-second host limits, native-memory access,
candidate implementations, configuration space, request identity, 730-row
history, three 14-step CV folds and 14-step forecast horizon. All controls keep
identical current data, numerical tools and completion/fallback rules.

History contains only source- and recording-visible outcomes from forecasts
made before their targets. Every arm finishes the current origin before the
next origin is exposed. No extra historical query or model execution is allowed
inside the annotation hook. Recipe support is descriptive, not an accuracy
ranking or a forecast selection. The actual worker enforces admission and the
agent must explicitly select a tested execution after comparison.

## Cohort, pilot and continuation

Use the exact four-series, 26-origin development jobs of candidate-100 plan 003,
with the same task-source hash and job identities. Start fresh arm-specific
state; never reuse candidate-100 forecasts as candidate-107 outcomes.
The full comparison is 312 sessions, 104 matched cases per arm.

The pilot is the first six sequential origins for all four series: 72 sessions.
This includes two origins after the four-origin recipe-support threshold; a
three-origin pilot would never test a mature recipe suggestion. Keep two series
parallel, serial origins and the original arm rotation within each origin.

Continuation adds the remaining 240 sessions without rerunning the pilot,
resetting memory, dropping failures, changing settings, or regenerating outcomes.
Pilot acceptance requires all 72 retained grades, all forecasts structurally
valid, at least 22/24 complete workflows per arm, no integrity/audit failures,
complete shutdown/usage accounting, and a passing independent recipe audit.
Whether the agent chooses to query history or follow a suggestion is measured,
not a gate. Accuracy is not used to accept the pilot or decide which pilot cases
to retain. If an integrity gate fails, retain the failed run and fix the cause
under a new prospective version; do not silently continue changed code.

## Analysis and promotion

Primary: arithmetic mean per-case RMSLE across all 104 cases, including the
unchanged failure fallback. Compare ledger with no-ledger Gnomon and Hermes;
retain all failures, costs and seeds exactly as executed. Use the existing
series/block uncertainty method without treating repeated origins as independent
series. Report the small four-series development scope explicitly.

Secondary: completion; cold origins 0–3; mature origins 10–25; late origins
22–25; historical queries; plan availability; proposed versus executed recipes;
distinct current configurations; current-CV versus final selection; fit/request
counts; native-memory use; tokens; missing usage and any API errors. Recipe
availability and adoption do not establish that history caused better accuracy.

The existing 20% target is unchanged. A development result below it does not
open held-out data. Any promoted candidate requires an unchanged prospective
multi-seed development confirmation and a frozen untouched-final evaluation.
The known negative 104/105 search-ceiling screens remain part of the record;
this run cannot erase them or establish general ledger value.

## Dispatch prerequisites still outstanding at this freeze

Bind an exact capsule, runtime inventory, this protocol, development jobs, both
independent audits and host/controller sources into a machine-readable plan.
Test the 72-session pilot and non-rerunning continuation paths with scripted
upstream replies, including archive/terminal accounting. Require a fresh output
and exclusive host reservation, verify predecessor process identity and terminal
evidence, and preserve unknown costs rather than reporting them as zero. No
production release, held-out data access or automatic dispatch follows from
this document alone.
