# Common workflow progress 097 — offline prototype

The 096 pilot cannot reach its predeclared full-workflow threshold: two plain
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
builds a separate immutable capsule from the exact 096 source, changing only
transport, task instructions and protocol text. Original and forwarded requests
and each reminder are retained. The draft capsule is
`results/workflow-097-offline-001/capsule`. It is **not dispatch-ready**.

Local tests verify incomplete/wrong-origin folds do not count, prior-origin
records do not establish current completion, actuals/predictions/metrics are not
included, state is unchanged, and the HTTP transport keeps the original model,
seed, request ceiling and protected phase. Test traffic uses scripted upstream
responses with no Engy access. The whole targeted check run passed 29 tests,
including continuation gates and state-transfer regressions; see
`results/continuation-workflow-local-checks-001`.

Before any paid 097 run, finish preserving 096, add independent reminder
verification to the analyzer, exercise the exact capsule with actual Hermes
workers, and freeze a new prospective plan and completion gate. Use fresh state
for all three arms and compare only matched arms within that run. Do not mix
096 controls with 097 ledger results, rescue selective failures, loosen the old
gate, or choose continuation based on accuracy. The final holdout remains closed
and the 20% objective remains unmet.
