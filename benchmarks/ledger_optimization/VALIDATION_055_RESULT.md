# New-series numerical validation: 5.17% improvement, 20% gate unmet

All416 tasks across sixteen new series completed after a disclosed numerical
solver correction. The ledger-assisted mixture reduced mean per-case RMSLE by
**5.17%** versus the matched current-backtest mixture. The paired95% interval
is **+1.32% to+8.80%**. Both domains improved on that comparison. This supports
a modest numerical effect on this panel; it does not meet the20% target and is
not a new Hermes/Engy agent evaluation or the untouched final evaluation.

| Scope | Tasks | Global CV guard | Matched intraday CV | Ledger intraday | Gain vs matched CV |
|---|---:|---:|---:|---:|---:|
| Overall |416|0.311943348486|0.310565542297|0.294511200398|5.1694%|
| Electricity |208|0.310533664665|0.296618879924|0.276935958833|6.6358%|
| Pedestrian counts |208|0.313353032307|0.324512204671|0.312086441963|3.8291%|

Against the global-CV guard the overall reduction is5.59%, with95% interval
**−1.67% to+11.24%**, crossing zero. The validation gate therefore fails both the
20% point target and the guard's uncertainty requirement. The matched-control
interval alone must not be presented as satisfying the full gate.

## What was fixed before evaluation

Protocol052 froze selection of eight additional series per domain,26 weekly
origins,24-hour targets,730-hour inputs, candidate recipes, temporal retrieval,
control capabilities, cold-start handling and uncertainty resampling. The
sixteen series are disjoint from original development and all32 final-reserved
series. Only the initial eligibility prefixes had previously been examined.
Final-reserved observations remain unopened. Sixteen series and two domains
limit generalization;416 tasks are not416 independent series.

The historical-evidence rule uses the sixteen nearest visible prior contexts
from the latest eight origin instants. Both arms use the same current forecasts
and backtest inputs; history changes the mixture-fitting evidence. Nominal
period-end recording/source availability and UTC-surrogate hourly coordinates
are assumed, not measured real-world release times. The method is a development
numerical prototype, not a released Gnomon capability.

## Numerical failure and uniform correction

The frozen054 run completed all533 raw forecast cases (117 warm-up plus416
scored) but stopped after165 mixture cases. SLSQP said success for a near-exact
CV fit, while its conservative suboptimality certificate failed. Preserve that
failure in [VALIDATION_054_FAILURE.md](VALIDATION_054_FAILURE.md).

Amendment055 was frozen at5aab39a after synthetic tests and before recomputing
validation fits. It changes numerical search and the certificate at exact fits,
retaining the exact unsmoothed objective, original regularizer and1e-5 exact
suboptimality bound. All416 mixtures were uniformly refitted, including both
controls, from byte-identical saved forecasts. No task was removed, no fallback
was substituted, and no forecast was regenerated. The amendment is disclosed;
this is not an uninterrupted preregistered run.

The independent audit passed296,346 arithmetic/identity/temporal/bootstrap
checks, plus1,248 amendment checks. It verifies all derived forecasts and
scores, training-input hashes, retrieved cohorts, certificates and all10,000
paired resamples. It independently recalculates simple forecasts but does not
refit every Ridge/Forest estimator; their frozen causal implementation and exact
history hashes were checked. Assertions are verification work, not statistical
sample size. No certificate refinements were needed for block fits.

## Costs and retained evidence

- Shared original forecasts:12,792 computations;6,396 estimator fits.
- Warm-up:117 usable cases;11 unavailable, no imputation/replacement.
- Failed054:495 completed weight fits and one failed fit;1090.37s total run.
- Corrected055:1,248 weight fits,50,637 SLSQP iterations;8.23s refit/analysis.
- Additional forecast computations for055:0. API calls and LLM tokens:0.
- Earlier adapter regression:6 weight fits on old development cases;0.265s,
  retained separately in validation-adapter-054-regression.json.
- No cold-start fallbacks occurred on this scored panel; the fallback path has
  synthetic tests, not additional empirical coverage here.

[Full receipt, scores, hashes and audit](evidence/broad-validation-055.json).
Archive: `results/broad-validation-055-001.tar.gz`.
The prior failed archive is retained separately; its costs are not erased.

Rerun055 from retained054 raw evidence using the command in
[VALIDATION_REFIT_055.md](VALIDATION_REFIT_055.md), choosing a fresh output folder.
Audit with `python3 -m benchmarks.ledger_optimization.validation_refit_verify`
followed by the source053 directory and the new run directory. No network/API
access is needed. Preserve the pinned numerical package versions in the manifest.

## Decision

No promotion, paid confirmation, main merge or PyPI release. Do not tune this
validation panel until it wins. Any further mechanism development must use the
original development panel and a separately frozen hypothesis. The best actual
Hermes comparison remains the earlier1.14% gain with an interval crossing zero;
the new5.17% result cannot replace that agent result. The20% final matched-agent
objective remains unachieved.
