# Does a finer fold stride pick better models?

**Verdict: the seam is worth having; changing the default is not supported by
the evidence.** `selection_stride` exists and is documented; it stays `None`
(one origin per horizon) until a larger corpus says otherwise.

## The question

`_origins` steps by `horizon`, so selection folds are non-overlapping and the
fold count collapses as the horizon grows — at `horizon=24` on 200
observations there are four origins, of which two are spoken for by
calibration and test. That scarcity is the reason per-lead residual mass is
thin, which is the reason `conformal_spreads` needs a pooled fallback and an
isotonic fit at all. If a denser stride bought better selection, some of that
machinery would be working around a self-inflicted problem.

## What is and is not safe to make denser

The two uses of a fold answer to different rules, and conflating them would
have introduced exactly the leak class fixed elsewhere in this branch:

- **Selection** compares candidates on identical windows. Overlapping folds
  are legitimate: the comparison is unchanged, there are simply more of it,
  which cuts variance.
- **Calibration** needs residuals that are exchangeable. Residuals from
  overlapping windows are dependent; treating *n* of them as *n* independent
  draws makes a conformal quantile look better determined than it is, which
  is precisely how an interval ends up anti-conservative.

So `selection_stride` widens the selection sample only. `residual_origins`
stays on the non-overlapping skeleton, and the last dense origin's window
still ends at the calibration origin, so no selection fold reads a point
belonging to the calibration or test partitions. Both are pinned by tests in
`tests/test_config_ensemble.py::TestSelectionStride`.

**It does not relax the four-fold cliff.** That threshold comes from needing a
calibration window and a test window after the selection region, which is
stride-independent. Denser selection buys precision, not answer rate — the
opposite of what a first reading suggests.

## Measurement

`stride = horizon // 4` against the default, comparing each arm's chosen model
by its score on the *held-out test fold* — the partition neither arm selected
on.

| corpus | n | selection changed | dense arm scored better | sign test |
|---|---|---|---|---|
| MTBench `finance_long` (real, daily) | 50 | 24 | 17 | p = 0.064 |
| synthetic seasonal | 30 | 0 | — | — |
| synthetic trend+seasonal | 30 | 0 | — | — |
| synthetic AR(1) | 30 | 7 | 2 | p = 0.45 |
| **combined** | **140** | **31** | **19** | **p = 0.281** |

Mean test-fold score on the real corpus: 0.0171 (disjoint) → 0.0155 (dense),
a 9.4% improvement. Mean measured coverage was unchanged (0.753 → 0.763),
which is the expected result given calibration never moved.

## Reading

The real-corpus result is suggestive but not significant, and it does not
survive contact with the synthetic corpora. Two things are worth separating:

1. **Denser selection is not baseline-biased.** On series with clear
   structure — seasonal and trend+seasonal — the selection did not change on
   a single one of 60 series. Both arms picked `seasonal_naive` and `ets`
   respectively. Where the signal is legible, the stride is irrelevant.
2. **The real-corpus gain is concentrated where the signal is not legible.**
   `finance_long` is close to a random walk; the dense arm moves selections
   toward `last_value`, which for that data is right. That is variance
   reduction doing its job, but it is one domain, and the AR corpus (also
   ambiguous) went the other way on a small sample.

Adopting `horizon // 4` by default would cost roughly 4x the selection compute
— prohibitive once TSFM adapters are in the candidate set — to buy an effect
that a 31-sample sign test cannot distinguish from noise.

## What would change the verdict

A corpus of several hundred real series across domains, with the same
test-fold comparison. If the dense arm wins at p < 0.01 there, the default is
worth revisiting for the built-in models with a per-candidate cost cap, since
the TSFM tier is what makes a 4x fold count expensive.
