# Ledger optimization development checkpoint

Branch: `dev/ledger-optimization`. **The 20% target has not been achieved.**
This work is development evidence, not a release or forecast-superiority claim.

The product change adds selectable MAE/RMSLE ledger summaries, calculated
ranks/ties, matched sample counts, pairwise differences and recent/lifetime
disagreement. MAE remains the default; forecast arithmetic is unchanged.
See [metric documentation](../../docs/ledger-comparison-metrics.md).

## Live matched experiments

| Run | Matched case/seed pairs | No-ledger RMSLE | Original ledger | RMSLE cards |
| --- | ---: | ---: | ---: | ---: |
| Legacy pilot 002 | 4 | 0.53450 | 0.53450 | 0.52976 |
| Legacy expanded 003 | 32 | 0.55377 | 0.54544 | 0.54493 |
| New development 004 | 64 | 0.53965 | 0.54176 | 0.53384 |

All completed decisions in these runs resolved explicit typed selections with
zero fallbacks. Experiment 003 also tested a blend suggestion; it performed
worse than control and was not promoted. Failed harness pilot 001 is retained
and excluded from accuracy claims, with incomplete API accounting disclosed.

On run 004, even the hindsight best of the eight candidate forecasts is only
11.71% better than the observed control. It is mathematically impossible to
reach 20% on that cohort by selection alone. This bound says nothing about
unseen series or expanded forecasting capabilities. The separately reserved
24-series confirmation set has not been scored.

## Reproduction and evidence

- [PLAN.md](PLAN.md): objective, fairness, original protocol.
- [PANEL_PROTOCOL.md](PANEL_PROTOCOL.md): disjoint new panel and confirmation rules.
- [JOURNAL.md](JOURNAL.md): experiments, failures, limitations and decision point.
- [evidence/](evidence/): pinned environments, input hashes, per-case scores and aggregates.
- Raw transcripts/candidate caches: local `results/ledger-optimization/`, with
  hashes and paths in reports. Large data and transcripts are intentionally
  excluded from Git under the registered retention rule.

For the completed new development run:

```sh
PYTHONPATH=/root/Gnomon/src .venv/bin/python -m benchmarks.ledger_optimization.summarize \
  results/ledger-optimization/new-agent-004 --output /tmp/new-agent-summary.json
.venv/bin/python -m benchmarks.ledger_optimization.headroom \
  /tmp/new-agent-summary.json \
  results/ledger-optimization/new-development-input-v2/cases.json \
  --output /tmp/new-agent-headroom.json
```

`analysis.py` supplies the preregistered paired bootstrap for a **complete,
consecutive-origin** run. It refuses the sparse-origin development run above.
Its statistical criteria are necessary but do not replace the final provenance
audit. Tests use synthetic balanced panels; no confirmation outcomes were read
to develop the analyzer.

Live runs require Engy credentials; never put credentials in tracked artifacts.
Retained runs contain API token usage, but the provider returned no dollar cost.
The calibration feasibility script changes predictions and is explicitly outside
the current fixed-candidate objective. It has not been promoted to production.

Validation: full product suite passed 1,166 tests with 29 skips after the core
metric implementation; subsequent focused checks and benchmark checks are
recorded in the journal. Main, release tags and PyPI are unchanged.
