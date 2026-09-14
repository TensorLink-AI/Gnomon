# Intraday model mixtures 050: small improvement, gate failed

Ledger-trained intraday weights reduced mean RMSLE **2.04%** versus the same
intraday tool trained on current CV only. They improved **2.77%** versus the
older global CV ensemble. All 416 development tasks completed, but neither
comparison reaches the required 20%. No final observations were accessed and
no paid confirmation was launched.

| Domain | Cases | Global CV ensemble | Intraday CV | Intraday ledger | Ledger gain vs intraday CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| Electricity | 208 | 0.111226402 | 0.107486486 | 0.107343934 | 0.13% |
| Pedestrian counts | 208 | 0.410036795 | 0.409935646 | 0.399505518 | 2.54% |
| Overall | 416 | 0.260631598 | 0.258711066 | 0.253424726 | 2.04% |

The ledger-specific gain is primarily in pedestrian counts. In electricity,
nearly all improvement over the older global mixture also appears in the
current-CV control. Giving both arms the same four six-hour mixtures is essential
to avoid attributing that shared tool improvement to the ledger.

Both methods used the same six archived predictions and a common anchor: the
045 global CV weights. Current CV trained on three folds; ledger training gave
half the mass to those folds and half to the sixteen temporally available
contexts already selected by 047. Neither method used 047 ledger weights or
046 future-aware diagnostics. Task identities, forecasts, actuals and lead-hour
alignment were preserved. New combinations are explicitly derived forecasts.

Training uses a convex smoothed RMSLE objective plus the same fixed 0.01 anchor
penalty in both arms. Smoothing raises each case loss by at most one millionth;
all reported scores use unsmoothed RMSLE without the penalty. Before freezing,
a synthetic exact-fit case exposed a loose convergence certificate. The bounded
numerical correction is preserved in [the notes](INTRADAY_050_SYNTHETIC_NOTES.md).
The acceptance threshold was not relaxed. All real-data fits passed without
requiring that additional refinement; it has synthetic coverage only.

Protocol/code frozen at `11af992`. Four synthetic tests passed. An independent
scalar audit passed **60,485 checks**, including every fitted objective and
convex bound, smoothing bound, mixture, forecast, score, source hash, temporal
constraint and both promotion comparators. All 832 weight fits completed with
45,874 SLSQP iterations, taking 6.2507 seconds wall and 6.0623 CPU in the measured
processing loop, excluding source loading and audit. Zero new provider fits or
API calls; inherited common cost remains 12,984 forecast computations. All prior
experiment costs remain retained.

The numerical development result is stronger than earlier tested rules, but
repeatedly improving on these known cases is not held-out evidence. It does not
prove that an agent benefits, that the effect generalizes, or that a production
ledger change should ship. The original >=20% matched-agent objective remains
unmet; main/PyPI and reserved future data remain unchanged.

Full fits, forecasts, source references, manifest, audit and rerun instructions
are retained in `results/broad-intraday-050-001/`. [The receipt](evidence/broad-intraday-050.json)
records the verified archive and per-file hashes.
