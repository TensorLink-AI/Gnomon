# M5 development archival controller

`m5_ml_controller.supervise` supervises one separately admitted child command
and archives its original process status, stdout/stderr, session files and costs.
It never retries the child, launches a continuation, selects on forecast error,
or opens the final evaluation. The old running Favorita controller is unchanged.

The controller authenticates the fixed development input before spawning. Its
completion check requires all 72 pilot or 624 full-stage tasks, exact original
jobs, explicit requested seed, independently audited grades and recomputed
per-case RMSLE. The complete-stage receipts must distinguish 72 retained and
552 new sessions. All fallback and failed-agent sessions remain in the evidence.
The existing cost collector counts each canonical session once; copied pilot
requests are not counted again. Missing billing/usage remains unknown.

Evidence completion is separate from pilot quality. A complete archived pilot
with fewer than 24 valid predictions or 22 full workflows in any arm returns
`complete:true, continuation_gate_passed:false`. Its observed outcome is kept;
the terminal-prefix admission check refuses continuation. Missing sessions,
incorrect scores, failed child processes and inconsistent gate receipts return
`complete:false` and retain an `INCOMPLETE.json` marker. Neither outcome causes
an automatic retry. Identical existing output/controller roots are rejected.

## Verification

Twenty-six tests passed across the controller, terminal/archive and stage checks.
Six new controller tests cover full 624-session accounting with a failed-agent
fallback, pilot-quality failure versus evidence completeness, altered scores,
real child success/failure, unknown usage after a failed request, and rejection
before subprocess creation for unauthenticated inputs.

A durable local probe launched two actual child processes, once each. They
copied synthetic 72-session receipts and exited 0 and 9 respectively. The first
was archived complete; the second retained its failed status and one request
without reported usage. Both archives were checked byte-for-byte against the
retained state. All 289 files, original commands and process records are kept in
`results/m5-ml-controller-001`. There were no model fits or Engy calls.

These are host-controller tests, not 72 actual agent workflows. Synthetic
cohort fixtures mock only the fixed production-hash authenticator; the real
cohort/score/archive/process logic runs. Existing real-worker seed probes remain
the separate execution evidence. Do not combine them into a claim that the
complete paid launch path has run.

## Remaining launch work

Connect this controller to the cohort-aware pilot/continuation launcher and its
frozen plan, runtime/build/source checks, preflight proofs and one-shot launch
reservation. The frozen worker's generated manifest also needs the launcher's
explicit requested-seed metadata. Recheck the copied prefix before credentials
and exercise multi-series execution through that integrated path. This controller
accepts an already admitted command; it does not perform those missing checks.

No paid M5 run has started or been admitted by this work. Main/PyPI and live
candidate-100 sources are unchanged. The final partition remains unopened and
the 20% objective is unproven. Receipt: `evidence/m5-ml-controller-001.json`.
