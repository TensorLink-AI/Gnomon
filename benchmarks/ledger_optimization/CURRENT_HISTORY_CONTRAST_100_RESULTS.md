# Candidate 100: offline evidence contrast replay

The replay covered all 30 ledger sessions in the 097 pilot and independent
continuation audits 001–004. Source evidence was unchanged. It made zero Engy
calls and zero model fits, and did not read future task targets or final-set data.
This is a descriptive replay of observed development behavior, not a new agent
experiment or evidence of accuracy improvement.

| Check | Result |
|---|---:|
| Historical reviews encountered | 32 |
| Cold reviews without a catalog | 4 |
| Explicit backtests encountered | 57 |
| Returned CV fold references independently verified | 171 |
| Pair boundaries without a previously returned review | 10 |
| Contrast panels rendered | 23 |
| Panels with the pair on the supplied page | 17 |
| Panels with a configuration absent from the historical catalog | 6 |
| Distinct window summaries, excluding `same_as` aliases | 22 |
| Window summaries with opposing CV/history winners | 3 |
| Opposing summaries supported by only one historical origin | 3 |

All three opposing summaries occurred in the item 1047756/store 23 chain, at
rounds 1, 4 and 11. The first two historical origins were 14 days old; the third
was 154 days old. These are overlapping development records, not three
independent demonstrations of a failure or benefit.

At round 11, current three-fold mean RMSLE was 0.7895908201410293 for Ridge and
0.7426319833245875 for random forest. Random forest won all three current folds.
Their single matched historical origin favored Ridge (0.6448454479465352 versus
0.7464074238915297); the last-four-origins window had no matched evidence. The
panel preserves both facts and selects neither model. This does not prove that
the agent's Ridge choice was inferior on the subsequently observed outcome.

## Payload and validation

The full diagnostic panels total 44,130 compact UTF-8 bytes: mean 1,918.70,
maximum 2,360. The original response bytes counted once per emitted pair sum to
39,714, so this representation adds 111.12% relative to that per-pair denominator.
That is substantial overhead on a backtest response. It is not the increase in
whole-session tokens: multi-pair boundaries repeat the baseline response in this
denominator, other tool responses are excluded, and no tokenizer was measured.
The renderer is deliberately **not deployed** pending a compact presentation
and prospective assessment of its benefit and cost.

Eight new regression tests and ten existing historical-consistency tests pass.
They cover current-versus-historical cohort disagreement, aggregate ties,
no-match versus unavailable-page distinctions, immutable inputs, exact task and
provider identity, distinct complete folds with already ended targets, invalid
scores, and hash/summary disagreements. The replay additionally recomputes each
current CV RMSLE from its logged prediction/actual pairs and checks the original
returned execution references. The helper itself still requires a trusted caller
for current evidence authentication and visibility; tests do not establish that
arbitrary agent-supplied evidence is trustworthy.

## Reproduction and next gate

Evidence: `results/current-history-contrast-100-offline-001/` contains the replay
driver, every panel/current input, source hashes, report, and test stdout/stderr.
The committed receipt inventories these files. Replay reads preserved development
archives listed in the driver; it creates output files exclusively so original
results cannot be silently overwritten. Copy the driver to a new results folder
and run with `PYTHONPATH=.` from the repository root to reproduce.

Current deployment and paid-run rules remain frozen. Candidate 100 needs a
bounded compact view, trusted boundary integration, and actual Hermes tests
before any prospective comparison. No narrower correctness or payload test is
a substitute for the requested 20% RMSLE improvement with uncertainty excluding
zero on the untouched final evaluation.
