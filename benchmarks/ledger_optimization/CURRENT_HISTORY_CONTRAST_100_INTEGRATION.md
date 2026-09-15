# Candidate 100: trusted records and evidence-read boundary checks

Status: still offline, not added to a worker capsule or a paid experiment.

## Trusted current inputs

`contrast_records_100.current_runs` reads the host's actual experiment-log prefix
and the exact three expected requests/actual arrays reconstructed from visible
input. It requires an explicit arm and the frozen lab configuration validator.
It checks canonical configurations, provider revisions, exact request equality,
execution identity, forecast/result agreement, target availability and observed
actual equality, and independently recomputes RMSLE. Duplicate executions or
origins fail closed. Incomplete groups are explicitly excluded with missing
origins; their history is never manufactured. Production forecasts and other-task
records are skipped before reading prediction or actual fields.

Plain Hermes's frozen execution wrapper does not emit Gnomon's typed status,
timestamps, unit or series fields. The reader permits their absence only when
the host explicitly identifies the plain arm, uses the exact authenticated
request for identity, and still rejects contradictory fields if present. Gnomon
and ledger executions require the typed metadata. This is a known wrapper
distinction, not an inferred provider success from an agent's text.

The reader is lab-specific. The caller must provide the real log prefix and
independently reconstructed visible inputs; supplying both a fabricated log and
fabricated expected data would not establish authenticity. It is not a public
general-purpose validation API and does not decide which model to execute.

## Completed-log replay

All 110 independently audited sessions from the pilot and continuation batches
001–005 were checked: 36 plain, 37 Gnomon, 37 ledger. The replay verified 319
complete current configurations and 957 CV fold executions; zero incomplete
configuration groups were present. All requests and actuals matched the visible
CSV reconstruction, and all scores matched recomputation. Source bytes remained
unchanged. No model was imported or fit; no Engy call or final target was read.

Expected requests were reconstructed with the exact pure `request_at` function
from the frozen 097 capsule, and canonical configuration validation used its
exact `configuration` function. The driver verifies those source hashes and
executes only the extracted pure functions, not model-importing modules.

This replay uses each session's completed-log state. It does not establish that
a particular configuration had already run at an earlier tool boundary. The
future integration must read only the prefix existing when the response is
generated and retain that boundary/hash for independent auditing.

## Real evidence boundary

The 46 artifacts (23 common, 23 ledger) from the compact-view replay were stored
in a fresh temporary project and retrieved through the exact frozen 097
`LabBoundary.dispatch` / `evidence_read` implementation. Across 189 pages,
reassembled bytes, declared length, SHA-256 and parsed JSON matched every
artifact. All 46 agent attempts to overwrite an artifact via `notes_write` were
rejected; source/artifact bytes remained unchanged. No subprocess was invoked.

This tests actual evidence dispatch and pagination, not the Hermes agent loop.
It does not yet prove that generated views are persisted before a real tool
response, that Hermes follows their references, or that any forecast improves.
Those remain integration/preflight gates before a prospective development run.

## Tests and retained evidence

Nine new trusted-record tests cover score recomputation, immutable inputs,
production/other-task exclusion, partial groups, duplicate IDs/origins, exact
identity, mismatched visible actuals, invalid folds, and plain versus typed
wrappers and explicit identity fields on unitless typed results. Together with
the existing contrast/view/history tests, 33 tests pass.

Raw replay artifacts and executable drivers:

- `results/current-history-contrast-100-records-001/`
- `results/current-history-contrast-100-records-002/` (final reader)
- `results/current-history-contrast-100-boundary-001/`

The second records replay repeats all 110 sessions after requiring explicit
identity fields even on unitless typed results. Both passes agree on all 957
fold executions; the original reader and its matching hash are retained in the
first directory. These are read-only validations, not additional agent trials.

To reproduce, copy a driver into a fresh results directory and run from the
repository root with `PYTHONPATH=.`. The preserved source snapshots and compact
artifacts must be present. Output reports are created exclusively and original
evidence is not overwritten. The committed receipt hashes drivers, output and
test logs. This work has made no change to the live 097 worker, main or PyPI;
the final holdout remains closed and the 20% objective remains unestablished.
