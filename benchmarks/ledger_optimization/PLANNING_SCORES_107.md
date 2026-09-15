# Independent score verification for candidate 107

`verify_planning_scores_107.py` recalculates every case RMSLE directly from saved
forecast points and the original, hash-authenticated development actuals. It
requires the complete planned stage denominator and retains failed workflows
and fallback forecasts. It checks task identity, origin, horizon, finite values,
source jobs and stage accounting without executing forecasts or calling Engy.

The arithmetic mean of per-case RMSLE remains the primary metric. Cold (0–3),
mature (10–25) and late (22–25) windows use the prospectively specified rounds.
Contrasts reuse the existing 2,000-draw series bootstrap with seed 142, retaining
undefined zero-control draws. That four-series development interval is
exploratory, not the untouched-final uncertainty requirement. The analysis never
opens a continuation or final gate and does not replace archive, temporal,
recipe, workflow or billing audits.

Example after a candidate-107 stage has finished:

```sh
python -m benchmarks.ledger_optimization.verify_planning_scores_107 \
  --root /path/to/completed-stage \
  --task-source /path/to/frozen-development-jobs.json \
  --output /path/to/new-independent-analysis
```

The source must match the stage's prospective task hash. Analysis must be outside
the original evidence and cannot overwrite an existing analysis. Every input
hash is retained and rechecked after reading; an incomplete stage is rejected
rather than silently reduced to a successful subset.

Six unit tests cover independently known scores, fallback retention, missing or
duplicated sessions, task/score corruption, nonfinite values, zero denominators,
and the difference between per-case and pooled error. On the full completed
synthetic host sequence, all 312 scores agree within 3.47e-18. All three arms
have equal synthetic scores, as expected for the scripted common choices.
This proves neither paid-agent efficacy nor the 20% target.

Raw evidence: `results/planning-scores-107-test-launch-001`,
`results/planning-scores-107-host-launch-001`, and
`results/planning-scores-107-host-001`. Compact receipt:
`evidence/planning-scores-107-host-001.json`. The paid pilot and its frozen
execution sources were not altered.
