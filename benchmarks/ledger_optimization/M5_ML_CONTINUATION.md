# M5 development continuation preparation

The separate M5 worker passed a two-series synthetic state-transfer test using
published Gnomon 1.2.0 and the matched Hermes runtime. This adds continuation
evidence to the earlier single-series/two-origin worker test. It does not admit
a paid M5 run, open final targets, or establish a forecasting benefit.

`probe_m5_continuation.py` executes three origins in each of two distinct
synthetic series across all three arms, audits all 18 sessions, freezes their
file inventory, copies the prefix, then executes only the six fourth-origin
sessions. All 24 forecasts and workflows passed. The probe preserves every file
in the original pilot, checks copied historical session files, verifies restored
native memory and checks for cross-arm/cross-series memory contamination. Every
resumed session has six matured configuration forecasts, three selected, with
unchanged predictions; its current target remains unavailable.

The test used 192 real numerical fits and 96 scripted model responses with
requested seed 7. All 224 integration assertions and 13,430 independent audit
checks passed. No Engy calls or real M5 numerical inputs were used. Both series
used the exact frozen M5 worker. This does not test simultaneous series threads,
a second seed, or the entire paid dispatch/controller path. All source, runtime
and original-pilot identity checks passed.

`m5_ml_stage_checks.check_development_stage` authenticates the previously prepared
development inputs before checking stage receipts. It requires exactly 72 pilot
or 624 complete-development sessions with matching series/origin identities and
recomputes every case RMSLE from the retained points and fixed actuals. Raw
grades, audited rows and per-arm report counts must agree. Missing, duplicated,
replaced or invalid rows reject. Audit failures and shutdown gaps cannot be
overridden by a passing completion flag.

The M5 pilot requires all 24 forecasts per arm to be valid and at least 22 full
workflows, preserving the existing pilot's 11/12 completion proportion. Poor but
correctly scored forecasts can pass this infrastructure check; no accuracy
promotion condition is introduced. Complete development retains failed forecasts
and their fallback scores in the full denominator. The 72 retained / 552 new
counts cannot be replaced by the old 36 / 276 counts.

Eight synthetic stage tests passed, including score tampering, matching fake
report changes, incomplete grids, threshold boundaries, poor but correctly scored
results, and failure retention. These are receipt checks, not an authorization
mechanism: terminal-process identities, archived hashes, source/runtime identity,
prospective plans and one-shot launch/continuation admission remain required.
Every returned summary leaves execution unauthorized and the final gate closed.

Evidence: `evidence/m5-ml-continuation-offline-001.json`, with raw outputs and
both immutable pilot and resumed work under
`results/m5-ml-continuation-offline-001/`. Stage test attempts are retained under
`results/m5-ml-stage-checks-001/`. The active Favorita trial remains unchanged;
the 20% final objective is unmet.
