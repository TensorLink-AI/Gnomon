# Should selection be decided on a distributional loss?

**Verdict: the seam is worth having; changing the default is not supported by
the evidence.** `selection_loss="pinball"` exists and is documented; the
default stays `"wape"`.

Companion to [`fold-stride-measurement-2026-08.md`](fold-stride-measurement-2026-08.md),
which reaches the same shape of conclusion about fold stride.

## The argument for changing it

Aion's answer is a distribution — nine quantile levels per lead time — but
selection is decided on WAPE, a point loss. A point loss cannot tell a model
whose uncertainty is well placed from one whose centre happens to land well,
so on the face of it the model chosen is optimised for the wrong output.
Pinball loss is the proper scoring rule for a quantile: under-prediction is
charged `level` per unit and over-prediction `1 - level`, so it is minimised
exactly at the true quantile.

## How it is scored

Reusing the fold forecasts already computed, so a distributional score costs
no extra model fits. Fold *i* is scored with quantiles built from the
residuals of folds **before** it and never its own — the same separation the
calibration fold enforces for the published interval, applied inside
selection. The first fold has nothing to calibrate from and is not scored.

## Measurement

50 real series (MTBench `finance_long`), horizon 12. Each arm's chosen model
is scored on the genuinely held-out `output_window`, which neither arm saw.

| | WAPE-selected | pinball-selected |
|---|---|---|
| mean held-out **pinball** | **0.871** | 0.891 |
| mean held-out WAPE | **0.0256** | 0.0262 |

Selection changed on 18 of 50 series. Of those, the pinball arm's choice
scored better on held-out pinball 8 times — a sign test at p = 0.81.

## Reading

Pinball selection is not better, and notably it is **not better on its own
metric**: the arm that optimised held-out pinball loss during selection went
on to score worse held-out pinball than the arm that optimised a point loss.
That is the result that settles it. The likely reason is that with a handful
of folds, the quantiles used for scoring are themselves calibrated on very
few residuals, so the distributional score is noisier than the point score
without being more informative — the same scarcity that forced
`MIN_RESIDUALS_PER_LEAD` and the isotonic fit.

The argument for pinball is sound in principle and may well become correct
with more folds per series. The measurement says it is not correct here, and
switching the default would have shifted every selection decision and every
golden on the strength of the principle alone.

## What would change the verdict

Series long enough for double-digit fold counts, where each fold's scoring
quantiles rest on a real residual sample. If the pinball arm wins held-out
pinball there, the default is worth revisiting — and at that point the
comparison should also be run on a corpus where interval quality is what the
caller is actually buying, rather than on daily finance, where the
distribution is close to a random walk's.
