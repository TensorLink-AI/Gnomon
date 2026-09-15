# M5 development worker integration

`m5_ml_capsule.build` creates a separate worker from the authenticated
candidate-100 parent and the existing eight-series M5 development export. It
does not select candidate 100 for final confirmation or admit a paid run.
Candidate 100's Favorita comparison remains active and has not established the
20% target. Reserved M5 targets remain closed.

The build changes six files only:

- `run.py` binds the exact development-job hash, changes the fallback task-source
  path, labels source/recording assumptions and missing promotions, and records
  the dataset/cohort identity. Its task count already follows the supplied grid.
- `TASK.md` explains the same M5 semantics to all three arms. The zero promotion
  column means unavailable data, not observed absence. Calendar features refer
  to the day preceding the period-end timestamp.
- `PROTOCOL.md` records the development cohort and temporal assumptions.
- `contrast_audit_100.py` includes the verified cache-order replay correction.
- `m5_ml_development_contrast.py` calculates descriptive matched differences,
  including fallback scores, without treating eight items in two stores as
  independent bootstrap clusters.
- `analyze.py` uses that descriptive comparison. All numerical session checks
  and forecast-score calculations remain inherited from the parent.

Model implementations, tool boundary, fit/request budgets, native memory,
outcome maturation, execution binding and transport are unchanged. The generated
source inventory records both changed and preserved file hashes. The original
capsule is never rewritten. Source files compile before a new output is created;
altered parent or task bytes reject before output creation.

The cohort contract gives 208 host tasks: eight series times 26 origins.
Each seed has 72 pilot decisions and 552 later decisions across three arms.
The old 36/276 launch and continuation gates cannot admit this worker unchanged.
Those gates, full multi-series resumed-state verification, and a prospective
runtime/seed/budget plan remain required. This build has no final-target exporter
and does not bundle any sales values.

Initial checks in `results/m5-ml-capsule-offline-001/` cover reproducible builds,
unchanged forecasting/budget modules, corrupted-source rejection, zero-control
contrasts and mismatched/duplicate pairs. A real-development preparation check
executes the new host preparation for first/last origins of all eight series and
all arms, then perturbs future targets: 96 preparations and 1,264 checks passed.
Arm-visible numerical inputs remain equal and unaffected by target perturbation.
These preparation checks intentionally supply empty prior state; they do not
verify resumed memory or accumulated outcomes.

The first full-worker probe stopped before any provider or Engy call because
the previously used `/tmp` runtime was absent. That failure is retained. Runtime
restoration under `results/m5-ml-runtime-restore-001/` uses the authenticated
Hermes archive and original package pins; its verifier confirms all 5,815 Python
files and exact 93/94 package inventories. Gnomon 1.2.0 is the sole arm package
difference. Subsequent worker integration receipts must be checked separately;
restoring a runtime does not prove that integration passed.

The second probe then passed using the restored runtime and the exact new
capsule: six complete synthetic Hermes workflows across three arms and two
origins, 30 scripted model responses, 48 real numerical fits and 3,859 independent
audit checks. Historical evidence and native memory carried to the second
origin, common CV tables matched across arms, and the ledger-only historical
view remained conditional on a prior review. No Engy calls were made. This
single-series test still does not replace the required multi-series resumed
dispatch test. The initial missing-runtime failure remains alongside the
successful attempt; no numerical attempt was lost or silently repeated.

Evidence: `evidence/m5-ml-capsule-offline-001.json`. Capsule SHA-256:
`e4a51304b46c87c2f83546eae07d983747ef9139db4975064098ebda88fc1f81`.
