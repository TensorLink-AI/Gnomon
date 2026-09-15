# Candidate-100 terminal audit reconciliation

The frozen candidate-100 annotation auditor can reject correct replies after
the worker reloads a canonically serialized review cache. The independent
auditor now reproduces that serialization boundary, without sorting response
arrays or accepting changed facts. Earlier retained batches already reproduced
the old error and passed the corrected audit. The live worker remains frozen.

`reconcile_contrast_100.py` is an evidence-only path for the completed run. It
must not be invoked while the controller or child process is live. It neither
restarts forecasts nor changes `INCOMPLETE.json`, an original report, or an
original exit status. It does not authorize dispatch or final evaluation.

## Required evidence

The helper requires the exact candidate-100 plan, original capsule, all 312
sessions, 36 retained pilot sessions, 276 new sessions, original completion
and runtime records, complete costs, and the controller's finished archive.
It checks every archived member against its inventory and original files.
Process termination must be checked on the original machine and boot. A copied
remote process receipt checked against local `/proc` is insufficient; a reboot
requires separately authenticated host evidence outside this narrow procedure.

First rerun the original auditor, against saved executions and into a separate
directory. It must reproduce precisely:

`Annotation audit: presentation matches verified inputs and latest requested history`

Then replace only its annotation-audit callable in memory and rerun the complete
independent analysis. Restore the original callable afterward, including when
analysis fails. Every expected series/origin/arm must be present, with no audit
failures or shutdown gaps. Accuracy and agent success are not admission filters;
failed agent sessions and unknown API usage remain in the evidence and costs.
Recomputed cost facts must match the controller's records; only the inspection
timestamp and local evidence root may differ.

The new destination preserves the original failure, corrected report and costs,
input/source hashes, and a reconciliation receipt. Original evidence is hashed
again after replay. Any failure is retained in `FAILED.json`; there is no retry.

## Invocation after terminal archival

From the original pod's code directory, using a fresh destination:

```sh
python3 -m benchmarks.ledger_optimization.reconcile_contrast_100 \
  --output results/contrast-100-development-001 \
  --launch results/contrast-100-continuation-launch-001 \
  --capsule results/contrast-100-dispatch-bundle-002/payload/capsule \
  --runtime RUNTIME_FROM_ORIGINAL_COMMAND \
  --plan PLAN_FROM_ORIGINAL_COMMAND \
  --destination results/contrast-100-terminal-reconciliation-001
```

This is a template: verify the capsule, runtime and plan locations against the
retained original command before using it. Runtime configuration does not call
an agent or fit a model during audit replay. No credential path is accepted.

## Validation and remaining work

Unit checks cover missing/duplicate cases, wrong origin/version/runtime/source,
unrelated failures, remaining audit failures, partial cost coverage, live jobs,
foreign-host process receipts, existing destinations and restoring the original
auditor after a corrected-audit failure. Retained-session replay checks both
auditors against 19 actual sessions and compares the corrected result exactly
with the previous independent audit, retaining all original files.

The first replay driver failed to import the benchmark package because its
script directory replaced the repository import path. That failed process is
retained as `results/contrast-100-reconciliation-replay-001`; later drivers use
`runpy` from the repository root. This was a local probe-launch problem, not a
forecast or evidence failure.

The complete 312-session terminal reconciliation has **not** yet run. The M5
launcher still rejects an incomplete predecessor. A separately verified admission
amendment will be needed if the full terminal reconciliation passes. The live
M5 integration sources are unchanged; its current proof must not be silently
relabelled as testing a future launcher amendment.
