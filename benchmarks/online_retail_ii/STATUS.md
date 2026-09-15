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
- Started the full 624-case development baseline run in
  `results/online-retail-ii-baselines-001`. Its `report.json` is the completion
  receipt; partial `host-scores.jsonl` is not a completed result.

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

## Remaining experiment work

Freeze and audit the actual Hermes runner's equal tools, information, native
memory, numerical/token/time budgets and filesystem isolation. Run the three
arms against these cases and score their host-owned typed executions. Freeze
the selected implementation and automatic control on validation before the
single final evaluation and clustered uncertainty analysis. The current scorer
explicitly does not certify ledger use or agent budget compliance.
