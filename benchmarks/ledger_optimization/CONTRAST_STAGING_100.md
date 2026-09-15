# Current staging: corrected three-arm bundle 002

The active candidate uses `results/contrast-100-dispatch-bundle-002` locally and
on the pod, with frozen plan 003. It restores the user-corrected three-arm 1.2.0
protocol; the old 1.1.9/four-arm requirement was accidentally reintroduced as a
future prerequisite and is not active. No candidate-100 paid session was launched
with either plan. Bundle 001 is retained as superseded staging evidence.

Bundle 002 contains the same 79-file layout and unchanged worker capsule. Archive
SHA-256: `a18612bcd7ed3b3dc58b8803cb2471654916cfef7d05f2f7c3b352f29f40a052`.
Plan SHA-256: `0d6e759270e23db8d54f5cc7a5a0009a8af366e8d9ace23cf66beb648b76bc27`.
The pod verified its files and input contract. The launch check rejected the live
predecessor before creating outputs or reading credentials. See
`evidence/contrast-100-three-arm-correction-001.json` for the correction record.

A separate one-shot waiter was armed on 2026-09-15 after its 12 tests passed.
It waits for the real predecessor process identities to terminate successfully,
then invokes the unchanged controller and full pre-provider audit gates. It does
not launch continuation. See `CONTRAST_WAIT_100.md` and
`evidence/contrast-100-waiter-001.json`. The historical no-waiter statement below
describes the earlier bundle-001 staging observation, not the current state.

## Candidate pilot admitted on 2026-09-15

The one-shot waiter dispatched the controller at 06:52:08 UTC after the original
097 run terminated successfully. Before candidate execution, the launcher
verified all 41,608 inventoried predecessor files and independently re-audited
all 312 sessions. The 185,457 checks passed; the recomputed report hash exactly
matched the original. That recheck made zero provider or Engy calls.

At 07:02:50 UTC, the accepted-launch receipt identified plan 003, the frozen
candidate sources, DeepSeek v4.1 Flash and Gnomon 1.2.0. Both controller and
worker identities were live. The 36-session pilot had 28 forwarded requests,
26 returned responses and two grades. These are live monitoring counts, not
audited efficacy results. The preceding statements about no paid candidate
sessions describe staging before this admission.

The candidate gives all three arms a computed current-CV table. Only the ledger
arm also receives a contrast with the historical review it already requested,
including counts, age and disagreement. No extra query, model family or fit
budget is introduced. The pilot's completion gate is independent of accuracy;
continuation requires a separate verified admission. No automatic continuation,
final-data access, merge or release occurred. The 20% objective is unestablished.

Receipt: `evidence/contrast-100-live-001.json`. Controller PID 125645 and worker
PID 125676 are bound to their boot and start identities in that receipt.

The first six completed sessions subsequently passed 5,391 independent checks,
with no audit failures or shutdown gaps. All 622 copied files and the archive
hash verified. The audit independently reconstructs the current CV table and
ranks from visible history, checks the available execution-log prefix, and checks
the comparison against the last requested historical review. All six forecasts
and workflows are valid. These are two matched cold-start cases with no matured
history, so they establish interface correctness on this subset, not ledger value.

Mean RMSLE is 0.417695 plain, 0.415972 Gnomon, and 0.417695 ledger. The snapshot
contains 53 agent requests and 655,068 reported tokens, plus six readiness
requests and 84 tokens; no errors or missing usage were observed. These requests
belong to the running pilot and must not be counted again as audit costs. The
collector and analyzer each passed on their first execution. Receipt:
`evidence/contrast-100-pilot-audit-001.json`. The pilot remains incomplete and its
completion gate is not yet established.

## Historical staging receipt for bundle 001


The isolated pod bundle contains 79 files: candidate and original-predecessor
capsules, frozen plans, development tasks, worker proofs, the continuation proof,
and the exact launch/continuation helpers. It is separate from the live run and
shared runtime. The compressed archive is 465,793 bytes, SHA-256
`873121d610b98adcc7e284ce88bb01634d131057641904ed9bd2050c284ee2af`.

`build_contrast_bundle_100.py` checks the candidate plan/source/proof identities,
the continuation's exact helper and worker hashes, and imports all five command
entrypoints in the isolated copy. It also invokes the actual input validator from
that copied code. It creates neither a pilot nor an API request.

The bundle was copied to the fresh pod directory
`/root/gnomon-ledger-ml-v3/code/results/contrast-100-dispatch-bundle-001`.
The pod verified the archive and all 79 file hashes. The copied validator passed.
The actual launcher was then called with `--check-only` and no credential file.
It rejected the still-live 097 predecessor with the exact live-process error,
before creating either proposed output directory. This deliberate admission probe
is not a failed forecasting session and is not scored in either experiment.

A separate no-dispatch runtime check loaded the copied candidate and confirmed
that the pod's package inventory matches the inventory in the frozen plan and
synthetic preflight. It made zero provider or Engy calls. The real pilot will
repeat its runtime/build and source checks at admission; staging does not waive
those requirements. No waiter or automatic launch process was started.

Evidence: `results/contrast-100-dispatch-bundle-001/`, including builder receipt,
archive, payload inventory, deployment script/stdout/stderr/status and runtime
check script/stdout/stderr/status. The pod retains its own staged/check receipts.
The committed receipt identifies these artifacts by hash.

Remaining: the original 097 controller and worker must terminate with a complete
archive, then the full independent re-audit and admission checks must pass before
a fresh candidate-100 pilot. No accuracy-dependent continuation or final-data
access is allowed. The final 20% objective remains unestablished. Main and PyPI
are unchanged.
