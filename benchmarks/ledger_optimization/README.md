# Ledger optimization checkpoint

Branch: `dev/ledger-optimization`. **The 20% target has not been achieved.**
Development and a complete frozen confirmation are retained here. No release or
forecast-superiority claim is established.

The reserved confirmation completed **3,744 decisions** with identical raw
historical information and tools across all arms. Historical support improved
mean RMSLE only **0.56%** (95% paired interval **−0.36% to +1.82%**). This does
not establish a reliable mean or broad robustness advantage. Even perfect
hindsight selection among the fixed forecasts improves at most **12.49%**, so
20% is unattainable on this frozen benchmark without changing its constraints.
See [the full confirmation report](CONFIRMATION_010.md) and its linked audits.
No main merge or PyPI release was made.

Continuation screen 011 tested past outcomes conditioned on the current CV
leader, without changing forecasts. Its training-selected rule was 0.28% worse
than current CV and 2.23% worse than existing support on the later development
slice. It was not promoted to a live run. See [the retained result](CV_CONTEXT_011_RESULT.md).

The user's subsequent evaluation-development task is tracked separately in
[the accumulated-experience workflow benchmark](../experience_workflow/README.md).
It starts from arrived raw events and measures correct retrieval, revision review,
execution-bound decisions and agent work against persistent SQLite. Its new
efficiency objective does not replace or reverse this closed forecast-error result.

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
24-series confirmation set stayed closed until implementation freeze 1709b4c;
its subsequent complete result is reported above.

## Ledger-only continuation with equal historical information

The user clarified that the intended advantage is accumulating and using
experience, with unchanged forecasting tools. Context trial 005 gives **all**
arms identical matured historical score rows and current conditions; ledger
arms organize that evidence. This is a stronger control-information contract
than the earlier runs above, so their effects must not be pooled.

| Arm | Mean per-case RMSLE |
| --- | ---: |
| Raw-history control | 0.541443 |
| Existing MAE summaries | 0.543930 |
| Context retrieval | 0.540718 |

All 192 decisions passed the equal-information/typed-execution/arithmetic audit.
The **0.13%** mean improvement does not establish the 20% target. Tail losses
improved descriptively, while seed-level provider choice became less consistent.
See the robustness report before interpreting this as a reliability improvement.

The new public `retrieve_context` operation reads explicit progressively broader
historical cohorts in one database snapshot. Its sample-count threshold does
not imply confidence or select a forecasting provider. Complete results remained
identical across 208 regression queries after parsing reuse was optimized.

The subsequent registered development trial tested a concise historical-support
packet; see [CONTEXT_PROTOCOL.md](CONTEXT_PROTOCOL.md). Forecasts remained
unchanged, and the confirmation partition stayed closed during development.

The subsequent concise-support trial 008 completed 192 audited decisions:
control RMSLE **0.546149**, original ledger **0.537920**, support packet
**0.527158**. This is **3.48%** below its matched control, with 11 paired wins,
one loss and 52 ties. It is the implementation selected for confirmation under
the recorded development-choice rule. **The 20% target remains unestablished.**

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

Validation: the latest full product suite passed **1,203 tests with 29 skips**
after metric/context retrieval and parsing reuse changes. Subsequent focused
benchmark checks and complete confirmation audits are recorded in the journal.
This work remains on the development branch; no release was published.
