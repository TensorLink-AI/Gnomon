# M5 ML host integration: remaining work

## Verified development input contract

`m5_ml_development_contract.authenticated_contract` now authenticates the exact
original manifest and the previously prepared development jobs before parsing
them. It verifies the eight selected series, all 26 origins, overlapping history,
actuals, covariates and visibility cutoffs against the shared constructor. It
rejects missing/replaced series and altered task identities and returns the
72-pilot / 552-continuation / 624-total session counts per seed. It performs no
file reads, dispatch or provider calls itself.

The real existing development bytes passed this check. Twenty-four synthetic
contract/adapter/panel tests also finished with `OK`, including mutations of
history, targets, visibility and cohort identity. The original test tool's exit
metadata was lost to output truncation; its complete terminal unittest summary
was retained. The separate real-input verification exited 0. Receipt:
`evidence/m5-ml-development-contract-001.json`.

This completes input authentication, not the operational integration below.
The output explicitly leaves execution unauthorized and the final gate closed.
No archive or reserved numerical targets were opened by this verification.

## Dispatch integration still required

An isolated M5 worker builder now implements the development source binding,
common temporal/promotion disclosures and descriptive two-store comparison.
See `M5_ML_WORKER.md`. The launch/continuation and full resumed-state requirements
below still apply; the old helpers have not been silently relaxed or repurposed.

Two-series state-copy/continuation and fixed full-cohort stage-count checks now
pass separately; see `M5_ML_CONTINUATION.md`. These verify 18 retained plus six
new synthetic workflows and the 72/552/624 stage boundaries. They do not yet
cover the full dispatch/controller path or a second requested seed.

The separate seed-19 full-worker integration now also passes; see
`M5_ML_SEEDS.md`. It verifies six synthetic sessions, original and forwarded
seed settings, unchanged model outputs, runtime equality, and wrong-seed audit
rejection. The two-series resumed-state test above remains seed 7 only;
seed-specific continuation admission still needs verification.

This is a code inspection of the current candidate-100 capsule and its existing
launch helpers, together with the retained development-preparation receipt. It
does not grant final access, launch M5 sessions, inspect reserved targets, or
change the running Favorita comparison. Candidate 100 remains under evaluation;
it has not met the accuracy target or been selected for final confirmation.

The shared worker already prepares 730-row histories and 14-step forecasts.
The M5 adapter's first/last development-origin checks previously established
byte-identical numerical inputs across arms and exclusion of perturbed future
targets. The worker's `chain` and `main` iterate supplied series, and its manifest
counts jobs dynamically. Those facts reduce integration work; they do not prove
an end-to-end M5 run or resumed-state correctness.

The complete dispatch path still requires adaptation:

- The frozen worker authenticates the Favorita task-source hash. An M5 capsule
  must explicitly bind the already-prepared development jobs and preserve its
  source/manifest lineage. The prepared job hash is
  `dd608a2a0188cbe7e0e684127c4e5abbc77019982fdd14aabbb6e635064c164b`.
- `continue_guarded_093.jobs_from_source` requires four series and 26 origins.
  `continue_collection_096.verify_gate` and the archival controller require
  36 retained, 276 new and 312 total sessions. An eight-series dataset cannot
  be admitted through those checks unchanged. Do not bypass them or claim a
  36-session gate proves the complete M5 prefix.
- The existing exploratory analyzer computes series-cluster uncertainty and
  describes four reused series. M5 development has two stores; its items must
  not be presented as eight independent store clusters. The final calculation
  is separately specified in `M5_ML_ANALYSIS.md`; no development CI substitutes
  for its store-cluster and shared time-block calculation.
- A new exact-source synthetic integration must exercise multiple series and
  resumed origins through the complete dispatch path, retaining isolated arm,
  series and seed memories, prior numerical evidence, failed attempts and costs.
  Preparation-only checks with empty prior state do not cover this requirement.
- Runtime, prompts, cohort, seeds and shared budgets must be frozen before an
  admitted run. Source and recording assumptions remain period-end visibility;
  unavailable promotions remain explicitly unavailable, not observed zeros.

The already selected panel fixes these dimensions:

| Panel | Series | Stores | Origins | Decisions per seed, three arms |
|---|---:|---:|---:|---:|
| Development | 8 | 2 | 26 | 624 |
| Reserved final | 24 | 8 | 26 | 1,872 |

Three initial origins would contain 72 development decisions per seed, not 36.
The specified two-seed final comparison contains 3,744 decisions. Any staged
development completion gate must bind its actual cohort and be fixed before
dispatch; these arithmetic counts do not authorize execution or choose seeds.

Final access also requires the eligible complete development result, immutable
selected worker, authenticated final-reader gate, and one-shot final dispatch
described in `M5_ML_PANEL.md`. The required 20% development point improvement has
not been established. The eight prepared development series and 24 reserved
series must not be replaced after observing model performance.

Inspected sources: candidate-100 `run.py` and `analyze.py` in
`results/contrast-capsule-100-offline-003/capsule/`,
`continue_guarded_093.py`, `continue_collection_096.py`, and
`control_continuation_097.py`. Existing preparation evidence is
`evidence/m5-ml-development-prepare-001.json`. This inspection adds no test,
forecast, API request or final-data observation.
