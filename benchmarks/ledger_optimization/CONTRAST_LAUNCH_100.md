# Candidate 100 gated launch integration

The launcher is separate from the frozen worker and does not alter the running
097 experiment. The candidate remains a development trial, with no final-set
access or automatic continuation.

`launch_contrast_100.py` admits only the exact frozen candidate-100 plan and
synthetic proof. Its preflight adapter renames `sources` to `tested_sources` and
adds the already-verified runtime inventory expected by the existing worker;
it does not invent additional checks. The adapter's complete contents are checked
against the original proof and plan before launch.

Before credential access the launcher verifies both predecessor process identities,
terminal receipts, all inventoried evidence, exact cohort and cost coverage. It
then invokes `recheck_predecessor_100.py` in a fresh process with the original 097
capsule. That original analyzer recomputes the full report and must agree with
the retained rows and arm aggregates. A second inventory check ensures the old
evidence remained unchanged. Candidate runtime/package and source identities are
checked before the worker reaches the credential callback. The worker itself
rechecks the pinned 1.2.0 build and input/source hashes before that callback.

`--check-only` performs these prerequisite checks and retains the independent
re-audit, but creates no pilot, reads no credentials and sends no Engy requests.
It is not a read-without-writes operation: its separate audit output is intentional.
The pilot and audit directories must be fresh and must not overlap old evidence,
source directories or each other. Observation timeouts are not restart authority.

The credential callback retains the accepted plan/proof hashes and launcher source,
then reads the explicit credential file without copying its contents to evidence.
The pilot uses the unchanged 36-session first-three-origin design. The gate checks
the exact cohort, all 12 valid cases per arm, at least 11 complete workflows per
arm, and a clean audit. It never reads RMSLE to decide continuation. Exceptions
produce a failed runner receipt; the launcher does not retry or continue.

`control_contrast_100.py` uses the existing non-restarting archival supervisor.
The predecessor-audit output must be inside the fresh controller directory (for
example `LAUNCH/predecessor-audit`) so its command, stdout/stderr, exit status and
full rechecked report enter the same archive. The controller records process
identity, exit status, complete or incomplete status, and a credential-scanned
inventory/archive. It never launches continuation or the final test.

## Validation

Nineteen unit tests passed across plan, predecessor, launcher and controller:
source/proof/runtime tampering; preservation of budgets/cohort; live-process
rejection before credential or process access; wrong cohort; audit failures;
accuracy-independent gate boundaries; retained failed runs without retry;
accepted-plan retention; and archive inclusion of the predecessor recheck.
Tests of the launch callback use a synthetic driver, not a real paid session.

The retained completed 097 pilot was also independently re-audited through its
original capsule: 36 sessions, 8,135 checks, identical rows and arm aggregates,
zero provider/Engy calls. It does not establish that the still-live full run has
finished. Full terminal admission remains pending.

Evidence: `results/contrast-100-launch-integration-001/`, including normalized
preflight JSON, help output, test stdout/stderr, and the independently rechecked
pilot report. The candidate's real six-session Hermes integration remains in
`results/contrast-capsule-100-worker-003/`. Neither scripted integration nor
launch correctness establishes an accuracy gain.

## Continuation integration result

The unchanged `probe_continuation_096.py` was run against the exact candidate-100
capsule. Nine synthetic sessions established three origins, then the prefix was
copied and three fourth-origin sessions completed. All 12 were valid/full;
59 probe checks and 6,715 independent resumed-report audit checks passed. The
new annotation audit was active inside that analyzer. Source inventory and native
memory isolation survived the copy. Both selected and unselected forecasts
matured from earlier origins; current forecasts still had no current outcomes.

Cost: 96 real local numerical fits, 48 scripted model responses and zero Engy
requests. Copied pilot sessions are not extra executions. Together with the three
earlier candidate worker probes, total integration cost is 240 local fits and
138 scripted responses. Independent re-audits add no numerical/provider calls.
These are engineering checks, not forecasting accuracy replicates.

Evidence: `results/contrast-100-continuation-preflight-001/` and the launch
integration directory's `continuation.stdout`/`continuation.stderr`. The passed
proof binds the exact worker and continuation helper sources. The existing
continuation/controller implementation can be reused only with this proof and a
new candidate pilot that passes its completion gate. The original 097 live run
has not been changed, retried or stopped. Its final archive, re-audit and runtime
verification must precede actual candidate admission.
