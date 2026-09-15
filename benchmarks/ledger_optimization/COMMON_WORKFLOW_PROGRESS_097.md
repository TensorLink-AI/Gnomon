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
