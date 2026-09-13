# Earlier evidence prepared; strict attempt preserved as failed

All sixteen development series have usable evidence before the first of the
unchanged 416 scored tasks. **125 of 128 proposed earlier origins are usable.**
Fifteen series have eight complete warm-up origins; pedestrian sensor 1 has five.
No model forecasts or new accuracy result have been produced by preparation.

Strict attempt 041 required the complete preperiod for every series. It failed
at its metadata gate: sensor 1 has 120 absent hour labels beginning August 27,
2019. No counts were parsed in that attempt. Its original failure, source hashes
and coverage remain in [receipt 041](evidence/broad-warmup-041.json).

Separately frozen amendment 042 retains all eight attempted origins per series
but excludes an entire warm-up cohort when any historical or target observation
is missing. The missing rows invalidate sensor 1's earliest three origins. No
other series or scored task is excluded, and no value is imputed. This models
partial historical evidence; it is not a successful retry of the stricter design.

| Preparation result | Count |
| --- | ---: |
| Fixed development series | 16 |
| Attempted earlier origins | 128 |
| Usable earlier origins | 125 |
| Explicitly unavailable earlier origins | 3 |
| Unchanged scored cases | 416 |
| New forecast computations / API calls | 0 / 0 |

Source identities, units and original scored-history overlap are unchanged.
Every usable warm-up horizon closes before that series' first scored origin.
Five focused tests pass; the independent audit passed **1,033 checks with zero
failures** across identity, hashes, overlaps, every time boundary, missing counts
and admissibility. No reserved count field was accessed or numerically parsed;
the CSV reader still transits raw strings, as disclosed in earlier protocols.

The retrospective period-end availability/recording convention remains assumed,
not evidence of actual historical publication or provider execution times. A
future integration must label replay records accordingly. All controls must
receive equivalent warm-up data and have the same common computation charged.

With six recipes and three CV folds plus production, the 125 usable origins
would cost 3,000 additional computations, including 1,500 estimator fits. That
cost is additional to the original 9,984 computations and must not be hidden.
No fitting is authorized by this preparation's protocol. Next work must freeze
the selection experiment before running those models; the current algorithms
and oracle diagnostics have not established a 20% gain.

Artifacts: `results/broad-warmup-042-001/`. [Receipt 042](evidence/broad-warmup-042.json)
contains exact archive/file hashes and all audit results; both stage archives
and their contents were verified. Code frozen at `e72b169` before count parsing.
Replay preparation to a new output directory with the same pinned sources:

```sh
python3 -m benchmarks.ledger_optimization.broad_warmup_available results/broader-source-034-001/electricity_hourly_dataset.zip results/pedestrian-source-036-001/publisher-archive.zip results/broad-panel-037-001 results/pedestrian-source-036-001/timestamp-coverage.json results/broad-warmup-042-rerun
python3 -m benchmarks.ledger_optimization.broad_warmup_verify results/broad-panel-037-001 results/broad-warmup-042-rerun
```

Main/PyPI unchanged. No paid confirmation or reserved-outcome evaluation ran.
