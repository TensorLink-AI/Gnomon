# M5 worker seed replication

`m5_ml_seed_capsule.build` creates isolated seed-7 or seed-19 copies of the exact
verified M5 development capsule. It authenticates the parent manifest, every
worker file and the cohort contract before creating output. It rejects unknown
seeds, altered sources, symlinks, nested output and overwrites. No dataset values,
credentials, network or final targets are read by the builder.

For seed 19, only `worker.py`, `transport.py`, the matching agent-request audit
expectation in `analyze.py`, and a protocol note change. Numerical model random
states, model space, source binding, prompts, memory rules and budgets remain
unchanged. The service-readiness canary stays seed 7, temperature zero, 16 tokens;
it is not an agent decision. A seed-7 copy preserves all worker source bytes.
Three builder tests cover the exact change set, reproducibility, parent
preservation and rejection before output creation.

## Seed-19 integration result

The exact seed-19 capsule passed six synthetic Hermes workflows across all
three arms and two sequential origins in the isolated local runtime. It made
48 real numerical fits and 30 scripted model responses, with zero Engy calls.
All 187 integration assertions and 3,859 independent numerical/evidence/budget
checks passed. Memory carried forward only within its own arm and requested
seed. All arms received identical common CV tables; historical contrast appeared
for ledger after review/maturation. Full evidence artifacts were read through
Hermes tool dispatch and their hashes verified.

Additional checks confirmed seed 19 in all 30 original worker requests and all
30 forwarded requests. On a separate copy, changing one forwarded request to
seed 7 caused the seed-19 auditor to reject. The original probe stayed unchanged.
All six scripted selections and forecasts exactly match the earlier seed-7
synthetic integration, as expected when the same scripted tool choices use
unchanged deterministic model implementations. Runtime package inventories and
the pinned published Gnomon 1.2.0 build also match the seed-7 probe exactly.

These checks verify requested seed propagation, not whether Engy/DeepSeek honors
seeds or provides reproducible stochastic sampling. The live paid experiment was
not modified. The generalized contrast probe defaults to seed 7 for historical
capsules without a requested-seed field and accepts only seeds 7 or 19.

## Remaining admission work

This removes the missing seed-19 full-worker integration prerequisite. It does
not establish a second paid-seed result, full eight-series execution, concurrent
series safety, or seed-specific resumed-state/controller admission. The existing
two-series continuation probe covers seed 7; seed-specific source/runtime/plan
checks must prevent a seed-7 prefix from being resumed under seed 19.

Every paid seed needs separate process, project, ledger, native-memory and result
roots and an authenticated requested-seed manifest. The 72-pilot / 552-new / 624
total stage counts apply per seed. A frozen prospective plan and full cohort-aware
launch/continuation gates are still required. No candidate has met the 20%
accuracy requirement; final data remain closed. Main and PyPI are unchanged.

Evidence: `evidence/m5-ml-seed-19-offline-001.json`. Exact source capsule,
scripted wire records, audits, runtime comparison, negative audit copy and
command stdout/stderr/exit records are retained in
`results/m5-ml-seed-19-offline-001/`. These synthetic responses have zero paid
tokens; the 48 real local fits are explicitly retained as test work.
