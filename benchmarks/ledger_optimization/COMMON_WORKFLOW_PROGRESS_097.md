# Common workflow progress 097 — pilot running

The separately frozen 36-session pilot launched at **2026-09-15 00:33:52 UTC**.
At 00:35:01 UTC, the controller and first forwarded/returned requests were
verified; the accepted manifest matches the tested source and package inventory.
No completed outcome was available at that initial verification.

- Controller PID `4183838`, start ticks `1557684005`.
- Worker PID `4183869`, start ticks `1557684021`.
- Boot ID `998193f3-2771-4162-80e8-1a4887370f60`.
- Run: `/root/gnomon-ledger-ml-v3/code/results/workflow-097-pilot-001`.
- Controller/evidence: sibling `workflow-097-pilot-launch-001`.
- Frozen bundle: sibling `workflow-097-pilot-bundle-001/payload`.
- Bundle SHA-256 `8139f24620af984cb8bd3e71805426debebea1bceb7e83684ddf8b0fe9353158`.
- Runtime: published Gnomon `1.2.0+ga38cd0cad353.s9723394ccb6d`; Engy
  `deepseek-v4.1-flash`, requested seed 7.

Launch receipt: `evidence/workflow-097-pilot-launch-001.json`. Monitor the same
process, without restarting, using:

```sh
python3 -m benchmarks.ledger_optimization.pod_guarded_093 \
  --run workflow-097-pilot-001 --launch workflow-097-pilot-launch-001
```

The gate remains 12 valid forecasts and at least 11 full workflows per arm,
with zero audit failures. No accuracy threshold determines continuation. The
controller never automatically launches a continuation or final evaluation.

The conditional continuation has been checked locally against this exact capsule:
12 synthetic Hermes sessions, 96 numerical fits, preserved three-origin prefixes,
and a fourth origin resumed for each arm. `continue_collection_096.py` is reused
with the new capsule and its new exact-source proof; its filename does not select
the worker version. `control_continuation_097.py` preserves the process, archives
evidence and counts all requests once, including failures. It records the 36
retained and 276 new sessions separately. Its archive uses the existing helper's
`pilot/` container name for the entire run; the manifest identifies both stages.
See `evidence/workflow-097-continuation-preflight-001.json`. No continuation has
been dispatched; the paid pilot must pass and finish before its state can be copied.

The completed 096 pilot failed its predeclared full-workflow threshold: two plain
Hermes sessions spent their exploration requests inspecting data/source files,
then could only retain the baseline. All failures stay in the completed pilot;
the 276-session continuation is not authorized. This is a workflow problem,
not evidence that the ledger's forecast calculations failed.

The proposed next experiment keeps 096's prospective collection, published
Gnomon 1.2.0, Engy deepseek-v4.1-flash, identical model/configuration space, raw
information, memory availability, and numerical/request/time limits in all arms.
It adds an identical progress reminder at requests **4, 8 and 11**, while more
than 90 seconds remain. The protected phase and 16-request cap stay unchanged.

The reminder reads only agent-visible task, history header and execution records.
It reports remaining requests, complete current-origin configurations, whether
an ML comparison exists, and whether an explicit selection after comparison is
recorded. It points to the next unfinished workflow step but chooses no model,
hyperparameters or forecast. It performs no fits and returns no scores or future
outcomes. This helps test ledger evidence use after reducing procedural failures;
it does not isolate a ledger-only infrastructure change.

`workflow_progress_097.py` is the proposed function. `workflow_capsule_097.py`
builds a separate immutable capsule from the exact 096 source. The first offline
prototype changed transport, task instructions and protocol text; it remains
retained in `results/workflow-097-offline-001/capsule` and was never dispatched.
The tested successor `results/workflow-097-offline-002/capsule` also adds the
independent analyzer checks. Original and forwarded requests, each reminder,
the visible evidence prefix length/hash, and remaining-time observations are
retained. The auditor independently reconstructs counts from the hashed prefix
without calling the reminder generator. It rejects extra outcome fields, changed
original messages, fabricated completion, shifted budgets and orphan reminders.

Local tests verify incomplete/wrong-origin folds do not count, prior-origin
records do not establish current completion, actuals/predictions/metrics are not
included, state is unchanged, and the HTTP transport keeps the original model,
seed, request ceiling and protected phase. Test traffic uses scripted upstream
responses with no Engy access. The whole targeted check run passed 29 tests,
including continuation gates and state-transfer regressions; see
`results/continuation-workflow-local-checks-001`. The subsequent exact-source
launch suite passed 48 tests in `results/workflow-097-launch-checks-001`.

The actual Hermes worker integration passed six synthetic workflows, 48 fits,
30 scripted responses, 143 probe checks and 649 independent audit checks.
All six request-4 reminders were audited; separate HTTP tests exercise requests
8/11, protected-phase behavior and request exhaustion. The two-origin worker
test also verifies collection, explicit selection, maturation and isolated
native memory. It used no Engy calls and makes no agent-efficacy claim. The
575-file archive and hashes are recorded in `evidence/workflow-097-worker-001.json`.

The prospective plan is frozen in `results/workflow-097-prospective-plan-001/plan.json`.
It preserves 096's cases, budgets, seed, model, options and gate byte-for-byte as
JSON values; the new launcher checks this equality. Complete 096 evidence was
preserved, and the remote read-only prerequisites passed. The separate 34-file
bundle is verified in `evidence/workflow-097-dispatch-ready-001.json`; no paid
097 call was made during preparation. Use fresh state for all three arms and
compare only matched arms within that run. Do not mix
096 controls with 097 ledger results, rescue selective failures, loosen the old
gate, or choose continuation based on accuracy. The final holdout remains closed
and the 20% objective remains unmet.

## First live independent audit

The immutable snapshot in `results/workflow-097-live-audit-001` captured all 17 sessions with final grade and memory receipts, without success or accuracy filtering. All 1,717 files matched source hashes before and after copying and survived archive verification. The frozen 097 auditor passed 3,705 checks with zero integrity or shutdown failures. All 17 workflows completed; 166 requests had returned usage (2,132,732 tokens), with no API errors or missing usage. The receipt records common-case scores separately from unequal completed-arm counts. This snapshot is a subset of the ongoing pilot, not an additional experiment or cost total.

The conditional continuation bundle is built locally at `results/workflow-097-continuation-bundle-001`, with source/proof hashes and isolated command-import checks. It has not been deployed or launched. Terminal pilot gate verification and independent complete evidence audit remain prerequisites. The final evaluation stays closed and the 20% objective remains unmet.

## Staged continuation and decision diagnostic

The 37-file continuation payload is now staged in a separate pod directory. All files, exact worker sources, synthetic continuation proof, runtime inventory, and installed 1.2.0 build matched. No continuation or additional Engy call was started by staging. The terminal gate is deliberately still pending; staging verification does not authorize dispatch.

A read-only diagnostic recomputed three-fold RMSLE from each audited execution before its selected checkpoint. All 17 finished decisions (6 plain, 5 Gnomon, 6 ledger) selected a current-CV minimum among their tested configurations. This does not rule out historical evidence influencing exploration. It does mean these cold-start decisions alone do not demonstrate a selection advantage beyond current CV. The diagnostic includes every audited session and adds no forecasts. Receipts retain exact event-log hashes and configurations.

## Terminal pilot and live continuation

The pilot finished at 2026-09-15 01:07:15 UTC with 36/36 valid forecasts and 36/36 full workflows. All 4,237 archived files were verified; the independent auditor passed 8,135 checks, with zero integrity or shutdown failures. The gate passed on completion and integrity, independently of accuracy. Final mean RMSLE was 0.503288 plain, 0.501086 Gnomon, and 0.498361 ledger. The ledger reduction versus Gnomon was 0.54%, with exploratory 95% interval [-2.10%, 4.43%]; this does not establish a benefit. See `COMMON_WORKFLOW_PROGRESS_097_RESULTS.md` and the final receipt.

After independent local and repeated pod gate checks, the additional 276-session continuation launched at 01:08:30 UTC: controller PID 8358, child PID 8389, with boot/start identities in the launch receipt. Both were confirmed live; all 36 retained sessions and their 3,702 files were unchanged. The run is `workflow-097-development-001`; its controller is `workflow-097-continuation-launch-001`. Combined totals include the pilot once. No final evaluation or release was started.

## First resumed-session audit

The first six completed continuation sessions (round 3, two series, all three arms) passed 1,937 independent checks with zero integrity or shutdown failures. Their 656 copied files matched before/after hashes and the archive hash. Each restored three prior outcomes; the six histories contained 53 matured production executions, 35 unselected. The auditor matched every matured forecast to its original request, point values and recording-visible outcome and verified that host synchronization started zero numerical calls. This confirms evidence carry-forward, not accuracy improvement.

All six completed full workflows, and all arms selected identical forecasts on the two newly matched cases. Together with the pilot, 42 unique sessions were independently audited. The new snapshot is disjoint from the pilot; its costs must not be added again when reporting the complete continuation. See `evidence/workflow-097-development-audit-001.json`.

## Continuation audit 002

The next 15 finished sessions passed 5,925 checks with zero integrity or shutdown failures. All 1,690 snapshot files verified; all 15 completed full workflows. The batch is disjoint from the pilot and batch 001, bringing independently audited coverage to 57 unique sessions. Only the 18 completed three-arm cases are used for the combined score comparison.

An independent reconstruction of the available CV results before selection found 4/5 current-CV-minimum choices in each arm. All three exceptions concerned the same round-4 task: Ridge mean 0.892492 versus seasonal 0.891061. Plain and ledger rationales explicitly preferred Ridge on two fold wins and the most recent fold despite the near tie. Gnomon-without-ledger incorrectly described the seasonal mean as about 0.972 and Ridge as the minimum. The selected forecast was identical across arms; this is evidence about explanation quality and deliberate tradeoffs, not a demonstrated ledger accuracy benefit. Exact rationales, hashes and before-selection calculations are retained.

## Continuation audit 003 and origin tracing

Another 11 completed sessions passed 5,180 independent checks with zero audit failures or shutdown gaps. All 1,264 copied files and the archive verified. The snapshot is disjoint from the pilot and previous two continuation batches, bringing independently audited coverage to 68 sessions. All completed full workflows. Every new decision selected its minimum tested current-CV mean. The first analyzer invocation omitted the repository import path and failed before analysis; its error is retained, and the corrected invocation passed. No experiment was rerun.

On the same 22 cases completed by all three arms, the verified totals are:

| Arm | Mean RMSLE | Reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.492300 | 3,204,965 | 221 | 276 |
| Hermes + Gnomon | 0.492123 | 3,175,883 | 224 | 252 |
| Hermes + Gnomon + ledger | 0.493495 | 3,001,201 | 213 | 264 |

Ledger is 0.28% worse in RMSLE with 5.50% fewer tokens than Gnomon without ledger at this development snapshot. Neither these partial scores nor token savings establish the 20% accuracy target. Unequal completed-arm counts are excluded from this matched comparison. The live run continues with its frozen settings; the final holdout stays closed.

A separate read-only audit traced the round-4 no-ledger explanation above through the actual boundary responses. All 32 returned review/backtest records had correct origins and metrics, independently reproduced from their scored pairs. The current baseline folds are August 30 (0.697400), September 13 (0.934102), and September 27 (1.041682), mean 0.891061. The decision instead attributes an August-16 baseline value (1.178013) to September 13. Substituting that older origin reproduces its claimed approximate mean of 0.972. This supports an origin-mixing explanation, not an incorrect returned score, and does not expose the model's internal reasoning.

The interface adds avoidable reconstruction effort: `start` computes the baseline folds but returns only checkpoint identity, while subsequent backtest calls return fold scores and their mean. The agent requested two historical review pages that did not contain the current baseline results. It also read only the first 4,096 of 5,932 characters of `previous_runs.json`; the next offset and total were correctly disclosed. A future common-arm variant could return the already computed baseline scores directly. That proposal adds no new evidence or numerical calls and must be tested prospectively; the live experiment has not been modified. All three arms selected the same forecast on this case, so correcting the explanation would not itself change its measured accuracy.

Receipts: `evidence/workflow-097-development-audit-003.json` and `evidence/workflow-097-origin-diagnostic-001.json`. Raw evidence and executable diagnostic are retained under their named `results/` directories. These snapshots are subsets of the eventual full-run costs, not additional paid experiments.

## Continuation audit 004

The next 21 completed sessions (seven per arm) passed 10,548 checks with zero
integrity failures or shutdown gaps. All 2,535 files matched the source and
archive hashes. This snapshot is disjoint from the pilot and batches 001–003,
bringing independently verified coverage to 89 sessions, all full workflows.

| Arm | Matched mean RMSLE | Matched tokens | Matched API requests | Matched fits |
|---|---:|---:|---:|---:|
| Hermes | 0.475830 | 4,303,635 | 293 | 356 |
| Hermes + Gnomon | 0.475792 | 4,283,124 | 295 | 328 |
| Hermes + Gnomon + ledger | 0.478044 | 4,071,541 | 278 | 336 |

These totals use only the same 29 cases completed by all three arms. Ledger has
0.47% higher error with 4.94% fewer tokens than Gnomon without ledger. This is
ongoing development evidence, not an established benefit or final result.

Before-selection CV reconstruction finds minimum-tested-CV choices in 5/7 plain,
7/7 Gnomon and 6/7 ledger sessions in this batch. Nonminimum choices are retained
as observations, not treated automatically as errors; their rationales have not
yet been audited. The receipt lists exact identities and scores.

A separate fixed-rule replay on the earlier 68-session snapshot found that
simple matched historical averages did not improve over actual agent decisions.
See `HISTORICAL_SELECTION_DIAGNOSTIC_099_RESULTS.md`. It neither changes these
live scores nor establishes that richer context retrieval cannot help. No new
fits were made by that diagnostic, and the final holdout remains closed.

## Continuation audit 005

The next 21 newly completed sessions passed 12,361 independent checks with no
audit failures or shutdown gaps. The copied archive and all 2,711 files matched
their source hashes. This disjoint snapshot brings audited coverage to 110
sessions: plain 36, Gnomon 37, ledger 37, all valid and full workflows.

| Arm | Matched mean RMSLE | Matched tokens | Matched API requests | Matched fits |
|---|---:|---:|---:|---:|
| Hermes | 0.476991 | 5,341,362 | 362 | 440 |
| Hermes + Gnomon | 0.480212 | 5,312,360 | 362 | 412 |
| Hermes + Gnomon + ledger | 0.481070 | 5,219,411 | 345 | 408 |

These figures cover the same 36 cases completed by all three arms. Ledger has
0.18% higher RMSLE and 1.75% fewer reported tokens than Gnomon without ledger.
This remains incomplete development evidence, with no established accuracy
benefit. The new batch incurred 202 model requests/responses and 3,242,256
reported tokens, plus 21 readiness requests/294 tokens; no usage gaps or API
errors. These costs are already part of the running experiment, not new trials.

Current-CV-minimum choices were 6/6 plain, 7/8 Gnomon and 7/7 ledger in the new
batch. The Gnomon exception was item 1047756/store 23, round 11: the same Ridge
configuration chosen by the ledger agent in the previous audit. Both compared
Ridge CV RMSLE 0.789591 with random forest 0.742632, with random forest winning
all three current folds. The no-ledger decision also says random forest "lost
fold 2" and immediately notes that 1.022 versus 1.057 is a win. It cites older
matured Ridge evidence as its reason for selection. The numerical evidence was
correct; that sentence is internally contradictory. The decision file is retained
and hashed in the receipt.

This is not evidence that the ledger uniquely caused a bad decision. The same
forecast choice and explanation inconsistency occur without it, and overriding
current CV is not inherently incorrect. Candidate 100's offline contrast may
make the cohorts easier to compare, but its replay does not show behavioral or
accuracy improvement. A prospective design should expose current-CV summaries
equally across arms and isolate the additional historical contrast as the ledger
treatment. The live 097 experiment has not been changed.

Receipt: `evidence/workflow-097-development-audit-005.json`. At the subsequent
02:27:17 UTC poll the original controller remained live with 112/312 completed;
the independent snapshot above contains 110. Final/protected data remain closed.

## Continuation audit 006

The next 32 newly finished sessions passed 20,957 independent checks with no
failures or shutdown gaps. The 71,708,080-byte archive and all 4,501 files were
verified. Combined disjoint coverage is 142 sessions, all valid and full workflows:
47 plain, 48 Gnomon, 47 ledger. Scores use only the same 47 cases in all three arms.

| Arm | Matched mean RMSLE | Reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.472702 | 7,033,815 | 475 | 568 |
| Hermes + Gnomon | 0.471936 | 7,189,919 | 474 | 536 |
| Hermes + Gnomon + ledger | 0.473114 | 6,834,997 | 445 | 524 |

Ledger has 0.25% higher RMSLE and 4.94% fewer reported tokens than Gnomon without
ledger. One API attempt has unknown usage, so these are reported totals, not an
assertion of complete billing. The new batch contains 318 forwarded/returned
requests, 5,049,288 reported tokens and one HTTP 502; readiness adds 32 requests
and 448 tokens. The affected ledger session (item 1047756/store 23, round 18)
completed validly without forecast fallback: 12 numerical successes, three
backtested configurations and an explicit post-comparison selection. The 502
receipt remains retained and usage_complete remains false. No rerun was needed.

Current-CV-minimum choices were 10/11 plain, 11/11 Gnomon and 10/10 ledger in this
batch. The plain exception is descriptive, not proof of an irrational selection;
current CV and future performance are different evidence. This batch adds no
claim of an accuracy benefit or causal evidence for candidate 100.

Receipt: `evidence/workflow-097-development-audit-006.json`. Subsequent live poll
at 03:14:33 UTC confirmed the original controller running with 149/312 completed,
all valid/full, and 49 matched cases. The 142 above are independently audited
records, not that later monitoring snapshot. Final/protected data remain closed.

## Continuation audit 007

The next 36 newly finished sessions passed 27,045 independent checks with no
audit failures or shutdown gaps. The 92,351,368-byte archive and all 5,223 files
were verified. Disjoint audited coverage is now 178 sessions: plain 59, Gnomon 60,
ledger 59, all valid and full workflows. Scores below use the same 59 matched cases.

| Arm | Matched mean RMSLE | Reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.469486 | 8,991,710 | 599 | 692 |
| Hermes + Gnomon | 0.470629 | 9,109,644 | 591 | 672 |
| Hermes + Gnomon + ledger | 0.470142 | 8,738,587 | 556 | 656 |

Ledger has 0.10% lower RMSLE and 4.07% fewer reported tokens than Gnomon without
ledger. This small interim difference is not a demonstrated 20% advantage. The
combined totals retain the earlier unknown-usage attempt. The new batch itself
contains 353 forwarded/returned requests, 5,811,487 reported tokens, no API errors
or usage gaps, plus 36 readiness requests/504 tokens. These are costs already
incurred in the same experiment, not new trials.

All 12 plain and all 12 ledger decisions selected a current-CV minimum among
tested configurations. Gnomon did so in 11/12; its exception was item 1047756/store
23, round 25. These are descriptive diagnostics, not evidence of causation or a
rule that overriding CV is wrong. The candidate-100 worker and trial plan remain
unchanged and no candidate-100 paid session has started.

A local diagnostic was accidentally invoked before the asynchronous analyzer
had written report.json. It failed with FileNotFoundError and was retained;
the diagnostic reran successfully after the audit completed. No forecast or
agent session was repeated. Both attempts are hashed in the receipt.

Receipt: `evidence/workflow-097-development-audit-007.json`. Final/protected data
remain unopened. The separately added four-arm synthetic final analysis closes
the prior-ledger comparison requirement; it does not change this live experiment.


## Continuation audit 008 and protocol correction

The next 24 completed sessions passed 11,776 independent checks with no audit
failures or shutdown gaps. The 31,058,894-byte snapshot archive and all 2,845 files
were verified before analysis. Combined disjoint coverage is 202 sessions:
67 plain, 68 Gnomon, 67 ledger; every one is valid and a full workflow. The table
uses the same 66 matched cases in all three arms, including every outcome.

| Arm | Matched mean RMSLE | Reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.474953 | 10,142,510 | 674 | 780 |
| Hermes + Gnomon | 0.477184 | 10,312,001 | 662 | 744 |
| Hermes + Gnomon + ledger | 0.481002 | 9,880,019 | 624 | 740 |

Ledger has 0.80% higher error and 4.19% fewer reported tokens than Gnomon without
ledger. The historical unknown-usage attempt remains in the combined evidence;
reported tokens are not complete billing. This new batch contains 244 forwarded
and returned requests, 3,956,644 reported tokens, no API errors or missing usage,
plus 24 readiness requests and 336 tokens. These are already-incurred costs of
the same run, not new experiments.

Current-CV-minimum selections were 6/8 plain, 7/8 Gnomon and 7/8 ledger. All
exceptions are retained with their tested configurations. They occur in both
ledger and control arms; departing from current CV alone does not demonstrate
bad reasoning or an infrastructure defect. The candidate-100 worker is unchanged.

Receipt: `evidence/workflow-097-development-audit-008.json`. Candidate-100 plan
003/bundle 002 restores the agreed three-arm Gnomon 1.2.0 protocol. The four-arm
planning statement at the end of audit 007 was mistaken and is superseded; its
original evidence remains retained. See PLAN.md and
`evidence/contrast-100-three-arm-correction-001.json`. No paid candidate-100 run
has started. The original 097 controller remains responsible for finishing its
312 sessions; it has not been restarted. Final/protected data remain unopened,
the 20% objective remains unestablished, and main/PyPI are unchanged.


## Continuation audit 009

The next 25 newly completed sessions passed 15,085 independent checks with no
failures or shutdown gaps. The 36,908,182-byte archive and all 3,087 files were
verified. Combined disjoint coverage is 227 sessions: plain 76, Gnomon 75, ledger
76; every session is valid and a full workflow. Comparisons below use the same
75 matched cases in all three arms, with no success or accuracy filtering.

| Arm | Matched mean RMSLE | Reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.466765 | 11,533,733 | 765 | 892 |
| Hermes + Gnomon | 0.466477 | 11,867,389 | 755 | 852 |
| Hermes + Gnomon + ledger | 0.470588 | 11,459,545 | 716 | 848 |

Ledger has 0.88% higher error and 3.44% fewer reported tokens than Gnomon without
ledger. The earlier missing-usage attempt remains in the combined evidence, so
the token column does not establish complete billing. The new batch contains
252 forwarded/returned requests and 4,120,547 reported tokens, no API errors or
usage gaps, plus 25 readiness requests and 350 tokens. No sessions were rerun.

Current-CV-minimum selections were 9/9 plain, 5/7 Gnomon and 7/9 ledger. The
exceptions and all tested configurations remain in the receipt. These are
behavioral descriptions; overriding CV is not itself an error or proof of a
historical-evidence benefit. There is still no established accuracy improvement.

Receipt: `evidence/workflow-097-development-audit-009.json`. The original run
continues toward 312 sessions. The separately tested waiter is armed for the
frozen candidate-100 pilot after successful predecessor completion and full
independent audit; it cannot restart either trial or open final data. This audit
changes no runtime, worker source, budget, cohort, prompt or candidate. Main/PyPI
and the final/protected-data gate remain unchanged.

## Continuation audit 010

The next 24 newly completed sessions passed 14,887 independent checks with no
audit failures or shutdown gaps. The 44,288,171-byte archive and all 3,131 files
were verified. Combined disjoint coverage is 251 sessions: plain 84, Gnomon 84,
ledger 83; all are valid and full workflows. The comparison uses the same 83
matched cases in every arm, without success or accuracy filtering.

| Arm | Matched mean RMSLE | Reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.467539 | 12,768,684 | 843 | 996 |
| Hermes + Gnomon | 0.470665 | 13,160,409 | 835 | 940 |
| Hermes + Gnomon + ledger | 0.471187 | 12,641,928 | 789 | 928 |

Ledger has 0.11% higher error and 3.94% fewer reported tokens than Gnomon without
ledger. The historical missing-usage attempt remains in the combined evidence;
reported tokens are not complete billing. This batch contains 234 forwarded and
returned requests, 3,758,161 reported tokens, no API errors or usage gaps, plus
24 readiness requests and 336 tokens. No forecast or agent session was rerun.

All new decisions selected a current-CV minimum: 8/8 plain, 9/9 Gnomon, 7/7
ledger. This describes the explored configurations and is not evidence of an
accuracy gain or a causal explanation. The independent audit and diagnostic
each completed on their first execution.

Receipt: `evidence/workflow-097-development-audit-010.json`. At the separate live
monitor observation on 2026-09-15 at 05:10:11 UTC, the original controller was live
at 255/312 completed sessions; all completed sessions were valid/full workflows.
The waiter observation at 05:09:46 UTC verified both predecessor processes and
the waiter live, with no candidate-100 output or controller yet created. These
live observations are distinct from the immutable 251-session audited subset.
The candidate remains frozen, final/protected data remain unopened, and the
20% objective remains unestablished. Main/PyPI are unchanged.

## Executed-forecast opportunity check on 83 audited matched cases

Independently recomputed RMSLE for 716 already-executed production forecasts
from the pilot and disjoint audits 001–010. Request histories, timestamps,
forecast horizons, series, units and selected execution identities were checked
against the development host jobs and audited selected forecasts. All 354
repeated configuration predictions agreed within 1e-12. Source files were
hashed before analysis and verified unchanged afterward. No new fits, API calls
or final-data reads were made.

| Arm | Selected mean RMSLE | Future-aware best within its executed set |
|---|---:|---:|
| Hermes | 0.467539 | 0.434848 |
| Hermes + Gnomon | 0.470665 | 0.438738 |
| Hermes + Gnomon + ledger | 0.471187 | 0.440151 |

The future-aware best choice from the union of all three arms' executed
forecasts has mean RMSLE 0.427823: only 9.10% lower than the matched no-ledger
Gnomon result. Thus selection-only changes among these already-produced
forecasts cannot yield the 20% target on this subset. This is an optimistic,
non-executable diagnostic using future outcomes and unequally explored
configurations; it is not a fair treatment, deployable policy, bound on the
whole permitted model space, or conclusion about the unfinished run.

The implication for development is to measure whether accumulated evidence
also improves exploration within the unchanged common models and budgets.
Candidate 100 already exposes current-versus-historical contrast during the
workflow; its frozen intervention and pilot admission rules are unchanged by
this post-hoc diagnostic. The observed bound is not used to select cases,
remove failures, change tools or access the final holdout.

The first diagnostic attempt incorrectly required Gnomon's top-level status
field on plain Hermes execution records. That attempt and its source/output
are retained. Correcting the surface-specific check allowed the second
analysis to complete; no agent session or forecast was repeated. Full execution
references remain in the hashed original report, with compact per-case
summaries in `evidence/workflow-097-executed-opportunity-001.json`.

## Continuation audit 011

The next 18 completed sessions passed 13,552 independent checks with no audit
failures or shutdown gaps. The 41,611,160-byte archive and all 2,550 files were
verified. Combined disjoint coverage is 269 sessions: 90 plain, 90 Gnomon, 89
ledger; all are valid and full workflows. The table uses the same 89 matched
cases in every arm, with no success or accuracy filtering.

| Arm | Matched mean RMSLE | Reported tokens | API requests | Fits |
|---|---:|---:|---:|---:|
| Hermes | 0.471258 | 13,834,926 | 906 | 1,084 |
| Hermes + Gnomon | 0.474632 | 14,129,651 | 894 | 996 |
| Hermes + Gnomon + ledger | 0.474280 | 13,682,510 | 849 | 1,000 |

Ledger has 0.07% lower error and 3.16% fewer reported tokens than Gnomon without
ledger: essentially tied on accuracy, not evidence for the 20% target. The
earlier missing-usage attempt remains in the combined evidence, so reported
tokens do not establish complete billing. This batch contains 182 forwarded
and returned requests, 3,096,968 reported tokens, no API errors or usage gaps,
plus 18 readiness requests and 252 tokens. These are retained costs of the
original run; no agent session or forecast was repeated.

All 18 new decisions selected a current-CV minimum (6/6 per arm). This is a
description of the explored configurations, not evidence of historical-data
influence or a causal explanation. Audit and diagnostic each completed on
their first execution. Receipt: `evidence/workflow-097-development-audit-011.json`.

The independent live monitor at 2026-09-15 05:30:36 UTC showed the original
controller live at 270/312, all completed sessions valid/full. This live count
is separate from the immutable 269-session audited subset. Candidate 100 remains
frozen behind the predecessor completion/audit gates. Final/protected data,
main and PyPI remain unchanged; the objective is unestablished.
