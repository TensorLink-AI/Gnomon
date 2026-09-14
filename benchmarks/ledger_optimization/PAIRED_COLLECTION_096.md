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

## Initial offline capsule (preserved)

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

## Local real-model integration completed

`probe_collection_local_096.py` now runs scripted guarded lab subprocesses on
two synthetic origins for all three arms, using separate local plain/Gnomon
venvs. Plain has no Gnomon installation; all other local package versions match.
NumPy 2.5.3, SciPy 1.18.1 and scikit-learn 1.9.1 match the frozen pod numerical
versions. This runs locally, without modifying or competing with the pod runtime.

The public index initially offered Gnomon only through 1.1.9. No older version
was substituted. The exact pod 1.2.0 wheel was copied read-only and verified
against SHA-256 `030a5cd063c424482bebdc2a522aa98a8f4bf88285bab31c4e47ddc9afa04f2f`.
The local installed build matches source SHA-256
`9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e`.

The first integration exposed a prototype bug: an in-memory typed forecast
point tuple was rejected because the new validator required a list. Both are
valid public representations. The corrected capsule accepts both and includes
a tuple-to-persisted-list regression. Ten local unit tests now pass. Original
capsule and failed integration (17 actual fits) remain preserved.

The corrected integration completed 48 real fits: eight per origin in each arm.
175 integration assertions passed. Collection did not select a forecast; exact
repeats and explicit commits reused executions. At the second origin, both the
selected baseline and unselected Ridge forecast matured, with unchanged points
and matched ledger evidence. Predictions agree across all three arms.

A separate verifier passed 213 checks, recomputing 42 metric records, checking
source and recorded cutoff visibility, confirming read-only ledger queries
left the database bytes unchanged, and comparing 46 installed files to the
verified wheel. Its first attempt mislocated wheel data entries under
site-packages; that verifier failure is retained, with the corrected relocation
check. No fits were rerun for that correction.

Total real fits retained across failed/successful integration: 65. Engy calls:
zero. No Hermes agent or API transport was exercised, no predictive-performance
claim follows, and the full pod worker/dispatch gates remain outstanding. The
live 093 run and queued seed integration are unchanged. The evidence root is
`results/collection-096-local-integration-002`; receipt:
`evidence/collection-096-local-integration-001.json`.

## Updated budget descriptions and admission audit

Capsule 003 changes five declared files in its generated copy: lab.py, TASK.md,
boundary_schemas_093.py, PROTOCOL.md and analyze.py. The original 24-file
inventory remains unchanged. Task and tool descriptions now disclose the four
fresh fits, unselected collection, reuse and reserve; remaining-batch capacity
uses four fits. The generated analyzer retains the corrected local 093 audit
and adds sequential collection admission checks.

The additional audit reconciles admitted missing origins against actual fits,
charges unsuccessful attempts, rejects duplicate successful executions and
checks phase, remaining budget, final-fit reserve and task/configuration identity.
The parent analyzer remains responsible for guarded-call, selection, numerical
and visibility checks. Six audit tests and ten capsule tests passed.

The updated capsule passed 187 guarded integration assertions with another
48 real fits across two synthetic origins and all three arms. Its separate
verifier passed 213 checks, recomputed 42 metric records and verified 46 installed
wheel files. Both selected and unselected forecasts matured; original execution
points survived, and current forecasts remained unscored. All 113 fits across
the original failed and two successful integrations are retained. No Engy calls
or Hermes agent sessions were made.

Receipt: `evidence/collection-096-local-integration-002.json`; raw evidence:
`results/collection-096-local-integration-003`. Actual Hermes worker/API transport
and a frozen prospective dispatch manifest remain outstanding. The candidate
remains undeployed, the running 093 experiment is unchanged, and the final
holdout remains closed.

## Hermes transport probe preparation (historical)

`probe_collection_worker_096.py` uses the same frozen run/worker interface with
scripted upstream replies, no real Engy calls and no credentials. Across two
synthetic origins and three arms it requests review/start, Ridge backtesting,
exact repeat, explicit baseline commit, and a prose final. It expects 30 scripted
responses and 48 fits, own-arm native memory persistence, no cross-arm memory,
and a complete independent audit. Initial API requests must contain the updated
four-fit task instructions and collection tool schema.

The project validator was checked on all six retained local integration projects
(39 assertions). Five corrupted copies—missing unselected forecast, premature
actual, duplicate fit, unintended Ridge selection and altered matured points—
were rejected. These checks validate the checker, not Hermes transport. The
actual worker run remains gated on the existing controller and queued seed test
finishing. Receipt: `evidence/collection-096-worker-preparation-001.json`.

## Prospective development schedule frozen

`collection_plan_096.freeze` verifies the exact original development source and
the capsule's complete frozen base, declared changes and file hashes before
writing a fresh plan. Plan-002 adds explicit protected selection budgets and
stronger inventory validation; the earlier unlaunched plan remains preserved.
Five tests cover the matched schedule, source/capsule drift, missing source
entries and overwrite rejection.

The schedule contains all four reused Favorita development series and 26 origins,
with three freshly initialized arms and seed 7. First three origins: 36 sessions;
conditional remainder: 276. Pilot continuation requires all 12 forecasts valid
and at least 11/12 full workflows per arm, plus a clean independent audit.
There is no pilot accuracy threshold. Retain pilot results exactly once and
include every failure; never substitute old 093 controls or old arm memory.
The collection rule, model families and 60-fit/16-request/480-second limits are
identical across arms. Compare accuracy within this new experiment only.

This is a development plan, not dispatch authorization or final efficacy evidence.
The current paid controller, queued seed integration, exact-source Hermes probe,
runtime verification and fresh-state checks must finish before paid dispatch.
Receipt: `evidence/collection-096-prospective-plan-001.json`. The final holdout
remains closed and still requires the separately frozen multi-seed protocol.

## Exact local Hermes worker integration passed

The earlier plan anticipated testing on an idle pod. Instead, an isolated local
runtime was reproduced from the frozen Hermes archive and the live pod package
inventory. This clears the actual-worker checks without modifying or competing
with the running pod. The paid launch prerequisite remains unchanged: finish
and preserve the current run and queued seed integration first.

Hermes archive SHA-256:
`e91f16fa0791926a91324814454d2530785100ab2793c62aa9dc7c84d1f03924`.
All 5,815 archived Python files match the local source. The 93 plain and 94
Gnomon package versions match the pod exactly; Gnomon 1.2.0 is the sole package
difference between local arms. Four additional live dependency pins were applied
after detecting their absence from the older asset requirements. Commands,
installation outputs and hash-verified transfer receipts are retained.

The real local Hermes worker, guarded tool boundary and proxy transport completed
six synthetic workflows across two origins and all arms. There were 30 scripted
upstream responses, 48 real fits, 142 integration assertions and 385 independent
audit checks, all passing. Agents received the updated task and tool descriptions.
Repeated backtests and commits added no duplicate fits; collection left the
explicit baseline selected; selected and unselected forecasts both matured at
the next origin. Native memory persisted within an arm and stayed absent from
other arms. All 161 actual fits across the collection integration attempts remain
accounted for, including the original failed attempt.

No Engy calls, real-data forecasts or learned agent selection were tested here.
This is worker reliability evidence, not an accuracy or token-efficiency result.
A 542-file evidence archive and per-session wall times are retained under
`results/collection-096-local-worker-001`. Receipt:
`evidence/collection-096-local-worker-001.json`. Pre-dispatch runtime/source checks,
fresh arm state, the original terminal audit and queued seed test remain required.

## Pilot preflight compatibility

`collection_preflight_096.accept` verifies the saved worker proof against its
committed receipt, all 542 raw evidence files, the current capsule, all six
matched grades and the complete independent audit. It emits the `tested_sources`
receipt consumed by the frozen pilot runner; the stronger two-origin integration
is reused, rather than fitting models again just to rename a result field.
Five altered-evidence/source/overwrite probes were rejected. No new fits or
Engy calls were made. Tested runtime inventory and build are retained for
pre-dispatch comparison. This receipt explicitly does not authorize dispatch;
predecessor terminal, fresh-state and runtime checks still apply.
Receipt: `evidence/collection-096-preflight-001.json`.

## Pilot launcher prepared and dry checked

`launch_collection_096.py` checks terminal identities for the paid controller,
its child, the seed controller and both seed children before accepting their
completion/exit/audit receipts. It then binds the plan and worker preflight to
current source and task hashes and compares runtime package versions. Credentials
are read only after these checks and the frozen runner's published-build check.
It records plan, source and predecessor references before the first API call.
It runs only the 36-session pilot and its independent audit; it never starts the
276-session continuation or final holdout automatically.

An actual check-only invocation exposed a launcher import-path bug: running the
launcher as a module had already loaded the repository's regular benchmarks
parent, so child imports ignored the new capsule path. The location check
rejected this before a run directory or credential access. The corrected launcher
pins subsequent child imports to the verified capsule. Nine tests now pass,
including this import-path regression and live controller/child, failed or changed
evidence, source/task mismatch and PID-reuse checks.

The corrected check-only command passed against real local runtimes, the actual
collection capsule and proof, and synthetic terminal predecessor metadata. A
separate read-only check against the real pod correctly rejected launch because
the original controller remains live. Both failed and passing checks are kept.
No additional fits or Engy calls occurred. Receipt:
`evidence/collection-096-launch-checks-001.json`. Actual paid launch remains pending.

## Pilot process and archive controller

`control_collection_096.py` wraps the gated launcher in one tracked subprocess.
It records controller and child PID/start-time/boot identities, exact argv, source
hash, stdout/stderr and exit status. It never restarts the child, increases its
budgets, copies credentials or starts the continuation. Launch and pilot paths
must be separate fresh directories. The child retains all existing predecessor,
source, runtime and build checks before credential access.

After child exit, the controller requires matching runner, audit and gate
receipts. A fully audited pilot that fails its workflow-completion gate remains
a completed experiment with `continuation_gate_passed=false`. A process failure,
missing receipt or audit failure produces `INCOMPLETE.json`. Both completed and
incomplete runs are archived with individual file hashes and an archive hash;
`FINISHED.json` describes archival completion and must be read with its
`complete` field and `INCOMPLETE.json`, not treated as automatic success.
External symlinks reject archiving instead of following runtime paths.

Eight subprocess/archive tests and the nine existing launcher tests passed.
Tests verify archived bytes, failing exits despite successful-looking receipts,
missing/audit-failed evidence, spawn failures, fresh paths and symlink rejection.
They used zero Engy calls and zero numerical fits. This is local lifecycle
verification; no paid pilot has launched. Receipt:
`evidence/collection-096-controller-checks-001.json`.

A follow-up handoff review found the original paid controller's pre-archive
credential scan was missing from the new wrapper. The wrapper now performs that
scan only after an accepted-launch receipt establishes that the child passed
its pre-credential gates. Rejected launches do not read credentials. Detected
credentials or an unavailable scan withhold the archive and `FINISHED.json`,
while retaining raw evidence locally. Four additional synthetic-key tests passed,
bringing the combined controller/launcher suite to 21 tests. No real keys, Engy
calls or numerical fits were used. Receipt:
`evidence/collection-096-controller-checks-002.json`.

## Relocatable dispatch bundle prepared locally

`results/collection-096-dispatch-bundle-001/dispatch.tar.gz` contains 32 exact
files: the tested capsule, frozen plan, preflight receipt, development task
manifest and controller/launcher package. It contains no credentials, mutable
arm state or final holdout. Archive SHA-256:
`e8dff0dc8a2f9de2428d347396e6ddec6c66a3e10c4c0c3ff88bc617947ff254`.

An independently extracted copy passed every file hash and the launcher's
check-only path with the real local runtimes and synthetic terminal predecessor
metadata. No output run was created and no fits or Engy calls occurred. The
bundle has not been uploaded or launched. Before using it, recheck the real
predecessor completion receipts and the remote runtime/source identities;
synthetic terminal metadata is solely a local relocation test. Receipt:
`evidence/collection-096-dispatch-bundle-001.json`.
