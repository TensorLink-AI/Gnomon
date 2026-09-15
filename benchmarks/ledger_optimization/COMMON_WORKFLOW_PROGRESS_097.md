# Common workflow progress 097 — tested; separate pilot ready

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
