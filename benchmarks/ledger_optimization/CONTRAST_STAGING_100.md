# Candidate 100 staged without dispatch

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
