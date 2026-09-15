# Candidate 100: synthetic worker integration results

The second isolated capsule passed the real Hermes worker probe on published
Gnomon 1.2.0 with scripted DeepSeek-shaped replies. It is not deployed to the
running 097 experiment and has not received a paid efficacy trial.

| Probe | Sessions | Real numerical fits | Scripted responses | Probe checks | Existing workflow audit checks |
|---|---:|---:|---:|---:|---:|
| Initial capsule / existing 097 workflow probe | 6 | 48 | 30 | 143 | 685 |
| Revised capsule / contrast and retrieval probe | 6 | 48 | 30 | 187 | 685 |

Every session completed its full workflow. Both probes used two synthetic
origins and all three arms, with identical data and scripted model choices.
The total development cost here is 96 local numerical fits and 60 scripted
responses, **zero Engy requests**. Synthetic zero token usage is not a measurement
of what a real agent/API would cost. The two probes are retained separately;
they are not independent forecasting accuracy evidence.

The revised probe checks that the baseline's three CV scores are returned before
exploration; all three arms receive identical current-CV tables; the ledger
overlay is absent when no catalog was returned and present after outcomes mature;
and Hermes follows each full-evidence reference through its actual tool dispatch.
Returned artifact bytes, SHA-256, task identity and six execution references are
verified. A repeated backtest still performs no new fits. Explicitly committing
the baseline after exploring Ridge still works, and both selected/unselected
forecasts mature at the next origin. Native memory remains isolated by arm.

The first capsule called a completed cold review `not_requested`; the second
corrects this to `last_review_had_no_historical_catalog`. It also measures the
returned remaining budget after annotation work and rejects unmanifested
symlinks in the frozen source. Original capsules, source fragments, probe outputs
and hashes are preserved. No failed paid session was rerun or removed.

## Cost and limitations

Across the revised probe's 30 successful agent-visible lab responses, annotations
add 38,889 compact UTF-8 bytes: mean 1,296.3 and maximum 2,197 per response.
This is 95.09% relative to those same responses with `evidence_summary` removed.
It excludes envelope escaping, instructions, artifact reads, host final checks
and repeated API context. The compact pair view is smaller than its detailed
predecessor, but the integrated current table and references still add material
payload. Prospective token/time costs must be measured rather than assumed.

Forty-five unit tests pass, including capsule reproducibility/protected-source
coverage and tamper rejection, annotation persistence and current-prefix binding,
common-arm equality, baseline exposure, explicit pair focus, stale/cold review
handling, and the previously validated records, view and historical calculations.

The 685 existing audit checks validate original workflow invariants; they do
**not** independently reconstruct every new annotation. An independent annotation
audit against each actual boundary and execution prefix remains required before
paid dispatch. The probe demonstrates real worker/transport plumbing with
scripted actions, not whether an unfamiliar model will use the view correctly.
The running 097 experiment must still complete and pass its terminal audit,
followed by a fresh frozen candidate plan and prospective development gate.

## Evidence

- `results/contrast-capsule-100-offline-001/`: original capsule and original source
  fragments, plus its probe stdout/stderr.
- `results/contrast-capsule-100-offline-002/`: revised capsule, tests and probe logs.
- `results/contrast-capsule-100-worker-001/` and `...-002/`: every synthetic
  workspace, request/response, execution, memory, artifact and audit record.
- `probe_contrast_worker_100.py`: executable revised probe using the pinned local
  Hermes/Gnomon runtimes. It refuses real credentials and intercepts model traffic.

The committed receipt inventories the evidence and identifies both capsule
hashes. Main, PyPI, the live worker and the untouched final set remain unchanged.
The requested 20% accuracy advantage is still unestablished.
