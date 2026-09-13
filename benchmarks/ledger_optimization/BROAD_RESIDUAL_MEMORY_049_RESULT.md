# Shared residual-memory correction 049: gate failed

Adding context-matched historical signed errors improved RMSLE **0.85%** versus
the same correction tool trained only on current CV. It improved only **0.21%**
versus the stronger uncorrected CV ensemble. Neither meets the 20% development
gate. All 416 cases completed; the previous 1.90% ensemble rule remains better
on these development cases, without establishing held-out superiority.

| Domain | Cases | Uncorrected CV ensemble | CV-only correction | Ledger correction |
| --- | ---: | ---: | ---: | ---: |
| Electricity | 208 | 0.111226402 | 0.110525188 | 0.110363517 |
| Pedestrian counts | 208 | 0.410036795 | 0.414080781 | 0.409797440 |
| Overall | 416 | 0.260631598 | 0.262302984 | 0.260080479 |

CV-only correction worsened overall error by 0.64%. Reporting only the 0.85%
ledger gain would hide that control degradation. The prespecified guard against
the unchanged stronger control prevents it from being mistaken for a substantial
improvement. Within electricity the ledger gain was 0.15% versus corrected CV
and 0.78% versus uncorrected; within pedestrians it was 1.03% and 0.06%.

Both methods used identical 045 CV ensemble weights, forecast inputs and a common
24-lead signed-log-error correction primitive. The control trained corrections
on three current CV folds; the ledger used half current CV and half the sixteen
past contexts already selected by 047. Unit ridge shrinkage was identical and
fixed before scoring. Neither method used 047's fitted ledger weights. Lead-hour
alignment, source availability and recording visibility were checked explicitly.

The fitted objective is a squared-log surrogate with a zero-correction penalty,
not the mean per-case RMSLE evaluation metric. Corrections were fitted and
applied before current actuals entered scoring. All transformed forecasts are
labelled derived; original provider forecasts and task identity were retained.
The common nonnegative floor affected ten CV-control leads and six ledger leads,
out of 9,984 leads per method; every affected lead is recorded.

Protocol and code were frozen at `515a3de`. Four synthetic tests passed before
execution. Independent audit passed **38,434 checks**, covering all 832 vector
fits, residuals, normal equations, objectives, stationarity, positive curvature,
clipping, forecasts, scores, source hashes, lead alignment and temporal bounds.
No new provider fits or API calls were made. Processing took 1.3831 seconds wall
and 1.3327 CPU, excluding input load/audit. The inherited common forecast cost
remains 12,984 computations; previously incurred 047 weight-fit costs remain in
its original report and were not erased because this method did not need them.

Raw evidence, corrections, input hashes, per-case forecasts and rerun instructions
are retained in `results/broad-residual-memory-049-001/`; [the receipt](evidence/broad-residual-memory-049.json)
records the verified archive and all file hashes. Period-end availability is
still an assumption inherited from 042/043. No reserved future data or hindsight
046 weights were read. No paid confirmation, main merge or PyPI release occurred.

This rules out a large benefit for this particular fixed correction on these
development cases. It does not establish that all historical correction is
useless. The broader goal remains unproven; another development rule must still
beat fair controls before it can justify an untouched matched-agent evaluation.
