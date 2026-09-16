# Development evidence

Implemented on `dev/ledger-optimization`, September 16, 2026 (Brisbane).
Main and the published package are unchanged.

- Prepared the frozen training-selected cohort: 48 products, 16 per stratum.
- Materialized development sales only, through June 5, 2011. Validation and
  final target periods remain unopened.
- Passed 19 tests using the installed Gnomon 1.2.0 wheel, including data
  visibility, source overlap, all ten models, direct/Gnomon parity, execution
  resolution, missing-case scoring and ledger replay across a DST boundary.
- Completed the final-code smoke: 12 cases, 13 methods, zero model fallbacks.
- Completed 325 additional checks on the real smoke outputs: independent
  RMSLE/MAE calculations, direct/Gnomon parity, retained missing-case denominator
  and Gnomon ledger RMSLE agreement with the original matured predictions.
- Completed the full 624-case development baseline run in
  `results/online-retail-ii-baselines-001`: 7,680 numerical computations, zero
  model fallbacks, 1,276.8 seconds. Mean RMSLE: Ridge 0.64513, CV ensemble
  0.65694, recent-four historical selector 0.66910, rolling CV selection
  0.67980, seasonal naive 0.78483. These are numerical controls, not agent arms.

## Small smoke scores, not the agent experiment

| Method | Cases | Mean RMSLE |
|---|---:|---:|
| Histogram gradient boosting, log sales | 12 | 1.00560 |
| Ridge, log sales | 12 | 1.00723 |
| Rolling CV selection | 12 | 1.11395 |
| CV top-three ensemble | 12 | 1.12683 |
| Weekly seasonal naive | 12 | 1.28046 |

All 13 methods are in [the retained report](evidence/smoke-report.json). This
six-product, two-origin smoke validates execution; it is too small to establish
which method is strongest. The historical selector equals CV during these cold
starts. No Hermes, paid Engy or three-arm causal comparison has run here. No
ledger performance target has been achieved by these checks.

## Evidence locations

Compact committed evidence is under `evidence/`. Complete local artifacts:

- `results/online-retail-ii-protocol-001`: prospectively frozen protocol.
- `results/online-retail-ii-env-001`: isolated environment, install attempts and
  test logs. The first PyPI install failed; the cached exact 1.2.0 wheel succeeded.
- `results/online-retail-ii-development-001`: host panel and filtering manifest.
- `results/online-retail-ii-smoke-002`: final-code plan, cases, forecasts, scores.
- `results/online-retail-ii-ledger-smoke-002`: replayed SQLite ledger and public
  paired comparison reports.
- `results/online-retail-ii-verification-002`: direct/Gnomon executions, an
  intentional one-submission/twelve-case scoring check and verification receipt.
- `results/online-retail-ii-baselines-001`: full development baseline attempt.

Earlier smoke and failing integration-test attempts were preserved rather than
overwritten. They are not the acceptance evidence for the final code. The raw
ZIP, customer-level transactions, environment and full host outputs are not
committed. The README describes rerunning from the supplied archive.

## Agent comparison

The separate [guarded Hermes runner](agent_eval/RUN.md) now implements matched
tools, information, memory and call/time budgets. Its local suite passes 22
checks; the real pinned Hermes runtime passed a six-session, two-origin
synthetic-transport preflight with native memory and direct/Gnomon parity.
That preflight made no Engy calls and is not agent performance evidence.

The paid run is supervised on Targon under `/root/online-retail-agent-001`:
`paid-001` contains the 3,744-session plan and eventual scores, `launch-001`
contains its controller admission/process/exit receipts. It begins with the
36-session quality gate described in RUN.md, retaining those cases in the full
denominator. Inspect `progress.json`, `pilot-gate.json` and `FINISHED.json` for
actual completion; a launch is not a completed comparison. The deployed bundle
SHA-256 is `e2eb66f0da4f0f9925f43c9b9436310dfc8dfc69f0da737ca3caf33f8cd304a2`.

Validation, final selection, clustered uncertainty analysis and the 20% target
remain uncompleted. The generic imported-submission scorer still does not
certify ledger use or budgets; this controlled runner retains the actual tool
and API evidence needed to audit those claims.
