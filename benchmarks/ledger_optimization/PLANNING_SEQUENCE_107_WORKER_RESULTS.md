# Recipe-plan worker integration: synthetic result

The isolated candidate-107 worker completed 15/15 scripted Hermes sessions,
covering five sequential synthetic origins in each of the three arms. All used
the published Gnomon 1.2.0 runtime where applicable. This is an integration
result, not an agent accuracy experiment or evidence for the 20% objective.

The ledger arm received plans from its latest explicitly requested review.
After four matured origins, the first next action was the common seasonal-7
checkpoint. Once that checkpoint actually existed, the next action became the
historically supported Ridge backtest. Executing that backtest removed it from
the untried recipes. Selection remained an explicit subsequent commit. Controls
received the same current CV table, with no historical recipe plan. Native
memory remained isolated by arm and persisted across origins.

## Evidence and limits

- Worker preflight: 384 assertions, 15 complete workflows, 90 scripted model
  responses, 120 actual numerical attempts, zero Engy calls. No paid accuracy
  result exists for this candidate.
- Original complete workflow audit: 8,745 checks and zero failures. Re-running
  it to a new output directory left all 1,711 original files unchanged.
- Independent recipe audit: 1,891 checks across all 15 sessions, 20 annotations,
  16 plans with a historical catalog, and two actionable next calls. It also
  invokes 6,443 existing CV audit checks; these overlap with the complete
  workflow audit and must not be counted as additional independent evidence.
- The independent auditor does not import the recipe generator, sequence
  planner, or compact renderer. It reconstructs support unions, prerequisites,
  calls, budgets and eligibility from retained replies and original artifacts.
  Historical source/recording visibility still depends on the original complete
  workflow audit, which passed separately.
- Eight deliberately corrupted copies were rejected: inflated support, invented
  budget, lost checkpoint, forgotten executed configuration, premature admission,
  automatic selection, unrequested review and an extra historical query. Each
  copy had its artifact address, hash, retained annotation and published view
  updated together. This tests semantic consistency beyond digest verification.

The first new audit failed because it incorrectly compared a checkpoint's
formatted file bytes with its canonical-JSON identity digest. The auditor was
corrected to the actual frozen identity contract. That failed attempt remains
under `results/planning-audit-107-launch-001`; no forecasts or original evidence
were regenerated. The corrected audit is `planning-audit-107-synthetic-002`.

The successful worker execution is `results/planning-worker-107-synthetic-001`;
its exact command and exit 0 are retained in `planning-worker-107-launch-001`.
The corruption copies and results are `planning-audit-107-faults-001`.
The original workflow re-audit is `planning-audit-107-workflow-001`.

The verified archive `results/planning-worker-107-archive-001/evidence.tar.gz`
contains 2,815 original files plus its inventory. Size: 17,024,809 bytes.
SHA-256: `ede23a4f6c8bd055055d7abfcab48d6f98a0b6e2b4e50e3d7d0cdc0607d36397`.
Inventory SHA-256: `6bcc036471475520784bdf1c1bb8e80521511cbbfede3670ea10765abd19ca58`.
Every archived member was read back and checked; original files were unchanged.

The prototype capsule remains marked not dispatch-ready. A prospective plan,
validated host launcher, terminal predecessor check and paid-run artifact
binding remain required. Main, PyPI and held-out data are unchanged.
