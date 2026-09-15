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
