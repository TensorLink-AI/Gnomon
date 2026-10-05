# CPU evaluation status — 2026-10-05

Built on main after Gnomon 1.4.0 and the README update. No runtime/default change.

- Implemented six datasets, nine CPU model configurations and six selector arms.
- Prepared 143 scored series; the six pilot series are reserved separately.
- Added immutable source/input/dependency identities, per-call deadlines and
  receipts, declared fallback forecasts, resume validation and a pilot gate.
- Fixed-model selection and its normalisation use only observations available
  before the earliest scored decision across the series pool.
- Prepared 12 synthetic agent workflow tasks and ordinary/lean/full templates
  using the existing driver; lean actually disables the optional ledger/time tools.
- Benchmark suite: **311 passed, 23 optional-environment skips**. New CPU module:
  **14 passed**. Ruff and whitespace checks passed.
- Corrected local feasibility pilot: **972 candidate calls, zero failures**,
  complete scored coverage on electricity and taxi (six scored folds each), all
  six selector arms replayed. See [machine-readable evidence](evidence/local-pilot.json).
  These two-origin pilot scores do not establish a forecasting improvement.

The first pilot was followed by a review of cross-series validation normalisation.
A regression test now ensures a series' later history cannot affect the fixed
model chosen at the pool's earlier scored start. The attached evidence is the
new pilot rerun after this correction, not a mixture of implementations.

## Remaining execution

Targon SSH to the recorded existing Gnomon pod returned `Permission denied
(publickey)` using its recorded key. No remote code was deployed, no Targon
benchmark was run and no new instance was provisioned. Updated connection
information has been requested. The scored suite requires a successful pilot
on the same host/allocation; the local pilot cannot authorise the Targon stage.

The 143-series scored evaluation is not run. Real agent runs are also not run;
model/endpoint, image identities and a spending limit remain unconfigured. The
prepared 108-episode synthetic cohort and existing 33-episode retrospective pilot
are launchable through the existing matched driver once configured. Answer
oracles alone do not attest tool use or persistent storage; trace review is needed.

Pretrained foundation models and covariate-aware evaluation are deferred from
this CPU-first scope. This is GIFT-Eval-derived online routing, not an official
GIFT-Eval submission or a full FASE reproduction. Existing defaults stay unchanged.
