# Prospective collection experiment — not launched

The 095 renderer could identify parameter neighbors but found no supported
one-setting comparisons on its supplied pages. The following audit tests a
possible cause without fitting new models or reading future target values.

## Observed collection gap on 65 matched development cases

| Arm | Completed backtest config/origins | Produced forecast config/origins | Missing forecasts | Original numerical attempts |
|---|---:|---:|---:|---:|
| Hermes | 263 | 127 | 136 | 916 |
| Gnomon | 284 | 126 | 158 | 978 |
| Ledger | 291 | 129 | 162 | 1002 |

These are distinct configurations at an origin, not numerical execution counts.
Every listed backtest configuration completed the three current folds. Producing
a current-origin forecast for each remaining configuration would have required
136/158/162 additional fits, respectively, assuming the same original searches.
No session's original attempts plus this hypothetical cost exceeds 60.
This is an arithmetic headroom check, not proof the extra fits would finish
within the time limit or leave agent behavior unchanged.

Ledger produced 64 configuration-pair/origin comparisons. Had every completed
backtest configuration also produced a forecast, it would have had 586 potential
pairs. Series/configuration pairs recurring at least three origins would rise
from 7 to 47. These are not independent observations, do not imply outcomes were
already mature at any particular query, and contain no hypothetical scores.
The equivalent control counts are retained in the receipt. No missing historical
forecast may be retroactively manufactured and called ex-ante evidence.

Receipt: `evidence/guarded-agent-093-collection-opportunity-001.json`. Exact
source hashes, configurations, costs and per-case counts are retained there.

## Candidate change and fairness constraints

An offline capsule can be prepared while 093 continues. After the existing run
and queued seed integration finish, investigate actual-worker integration and
a new frozen common lab variant. A successful three-fold backtest would also
produce and retain one current-origin forecast for that configuration. This is
an evidence-collection change, not a new model or a ledger-only free execution.

- Apply the identical collection rule, fit cost, admission and failure handling
  to all three arms. Retain the same 60-fit, model-request and wall-clock caps.
- Admit the full batch before beginning it and retain a final-selection reserve.
  A failed fit is charged and recorded. Do not silently fit in a retrieval call.
- Store an unselected forecast; do not automatically change the typed checkpoint
  or call the task complete. Explicit agent selection remains authoritative.
- Reuse a verified existing current-origin execution of that exact configuration
  instead of refitting it. Validate task identity, provider revision and request.
- Do not discard successful backtests if the production fit fails. Expose partial
  batch state and a precise retry/selection path, with every attempt metered.
- Raw production predictions and later matured outcomes remain available in
  controls. Ledger organization is the treatment; neither cross-arm forecasts
  nor future outcomes may be exposed to an arm's decision.
- Score an unselected forecast only after its complete actual horizon is visible
  under source and recording cutoffs. Preserve unsuccessful predictions too.
- Leave current metric, targets, series/origin manifests, model families and
  completion requirements unchanged within the new prospective comparison.

This change alters the search cost relative to original 093. Freeze it as a
separate experiment, not a patch to the running result. A matched new control is
required; do not compare the new ledger arm with old 093 controls and attribute
the difference to ledger. Do not bundle 094/095 displays into this first test.

## Evidence required before any paid dispatch

Build a separate capsule without editing the 24 frozen 093 files. Test actual
workers using synthetic data and scripted replies through the guarded boundary:

1. Exactly three backtests plus one production fit per fresh configuration, with
   every attempt reconciled and failures charged.
2. Exact repetition reuses all complete evidence; conflicting identity rejects.
3. Insufficient batch budget starts zero fits; deadline expiry preserves prior
   checkpoint and any completed evidence without declaring the batch complete.
4. Production fit failure after successful CV preserves the CV results and old
   checkpoint, and a bounded explicit recovery cannot duplicate executions.
5. Automatic collection never selects a forecast; explicit commit can reuse it.
6. Across successive synthetic origins, unselected forecasts mature identically
   in raw controls and ledger; no premature actual or recording visibility.
7. All-arm source manifests differ only where the declared treatment permits;
   agent/model settings, numerical budgets and model implementations match.
8. Independent analysis reconciles original and extra fits, checkpoints, stored
   outcomes, API usage and paired coverage. Report wall time as well as fit count.

Only then consider a bounded prospective development trial. The coverage audit
does not establish accuracy improvement. If the added collection consumes the
budget without useful evidence or harms completion, retain that result instead
of enlarging the ledger-only budget. The 20% final objective remains unchanged
and the final holdout stays closed until a defensible development candidate exists.

## Offline capsule prepared

`collection_capsule_096.build` verifies the complete original 24-file 093 source
inventory before creating a fresh directory. It replaces only `lab.backtest`
using `collection_backtest_096.py`; all original files stay unchanged. The
manifest explicitly says `offline_prototype_not_dispatch_ready`. It retains
the old task descriptions and analyzer, so it must not be used for paid dispatch
without completing those outstanding changes and integration checks.

The replacement validates exact task/configuration/provider revision identity
on both new and reused executions, retains successful partial CV fits, and
produces an unselected current-origin forecast. A complete repeat fits nothing.
A failed fourth fit leaves three reusable CV results; an explicit retry during
the same admissible exploration period fits only the missing forecast. Initial
baseline collection may consume its four reserved fits; other batches retain
one final-fit reserve. A deadline is checked before every new fit. No code
extends deadlines or request/fit budgets. Explicit commit reuses the resulting
execution; collection alone does not publish or replace a checkpoint.

Nine tests load the generated lab with a synthetic core and controlled budgets.
They cover full/reused collection, failed CV/production fits, budget/deadline
stops, identity/future-actual rejection, invalid new forecasts, explicit commit
reuse and immutable source inventory. A synthetic deadline-resume test resets
its fake clock solely to exercise reuse; actual expired sessions are not allowed
to reset time. These are state-machine tests, not actual numerical execution,
filesystem checkpoint durability, guarded-worker or maturation validation.

The retained capsule and receipt are in `results/collection-capsule-096-offline-001`.
Committed receipt: `evidence/collection-capsule-096-offline-001.json`. No paid
trial, actual model fit, package release or final-data access occurred.
