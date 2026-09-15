# Frozen M5 development launch inputs

`m5_ml_launch_inputs` authenticates the original M5 manifest, already prepared
development jobs, frozen three-arm candidate-100 budget plan, exact seed-specific
capsule and recorded six-session worker preflight. It does not import the worker,
read credentials, inspect the reserved source archive or execute a forecast.

Plans for requested seeds 7 and 19 now bind the same eight development series,
208 series/origin tasks, 72 pilot decisions and 624 complete decisions per seed.
They preserve Gnomon 1.2.0, DeepSeek v4.1 Flash, the common model/request/fit/time
budgets and separate native memories. Runtime inventories and build identity
match across seeds. Development comprises two stores, not eight independent
store clusters. The final partition remains closed.

| Requested seed | Plan SHA-256 |
|---|---|
| 7 | `2fbfc7716c3aaa027d9265a28e2d801c1c17423cbdd06644b8682aab3569e76a` |
| 19 | `643ba1061dffcd2076cd409a311da3441edb52dd57444d1a85c7dc4d10686325` |

Raw plans: `results/m5-ml-launch-inputs-001/plan-seed-7.json` and
`plan-seed-19.json`. Their statuses explicitly deny dispatch authorization.
They freeze inputs/budgets before any paid M5 result; the remaining launcher
source and operational preflight must be independently bound before execution.

The reservation helper uses exclusive file creation in one declared host
registry. Its key binds dataset, capsule, requested seed and stage, independently
of output-directory names. A competing launch or renamed output cannot obtain
a second reservation. A reservation is never automatically released after a
spawn failure or interruption. The launcher must first verify its predecessor,
runtime and applicable prefix; reservation alone proves none of those facts.

Six tests passed, including concurrent attempts for different output paths,
separate pilot/continuation stages, changed budgets/proofs/sources, unknown seeds,
plan changes before reservation and rejection of reused paths. The real-artifact
freeze and verification passed for both seeds; no production reservation was
created. Tests create reservations only in temporary synthetic directories.

Remaining: wire the frozen inputs, actual host runtime/build inspection, stopped
predecessor and copied-state checks into the pilot/continuation launcher and
controller; bind those source files and exercise the integrated multiple-series
path before credentials/API calls. No new efficacy evidence or final admission
follows from these input checks. Receipt:
`evidence/m5-ml-launch-inputs-001.json`.
