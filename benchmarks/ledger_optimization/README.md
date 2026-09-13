# Ledger optimization checkpoint

Branch: `dev/ledger-optimization`. **The 20% target has not been achieved.**
Development and a complete frozen confirmation are retained here. No release or
forecast-superiority claim is established.

## Latest completed comparison: Gnomon 1.2.0 and Hermes

DeepSeek **deepseek-v4.1-flash** through Engy; 104 tasks per arm, four retail
series and 26 forecast origins. All forecasts were valid; full workflow counts
additionally require backtest comparison and explicit selection.

| Arm | Mean case RMSLE | Completed workflows | Reported tokens |
| --- | ---: | ---: | ---: |
| Hermes | 0.480148 | 99/104 | 25,723,263 |
| Hermes + Gnomon 1.2.0 | 0.481454 | 102/104 | 24,581,379 |
| Hermes + Gnomon 1.2.0 + ledger | 0.475970 | 102/104 | 29,344,909 |
| Hermes + explicitly prompted native memory | 0.475949 | 104/104 | 26,831,582 |

Ledger improved error by **1.14%** versus no-ledger Gnomon; exploratory 95%
interval **−2.13% to +4.20%**. Native memory was a separate, later follow-up,
not a simultaneous randomized fourth arm. Its accuracy effectively tied ledger.
Main plain/ledger token accounting is incomplete; these are not billed costs.
The main and follow-up audits passed 59,003 and 19,312 checks respectively,
retaining one worker termination after a valid checkpoint. No agent evaluation
remains running from these runs.

**[Full latest results, costs, audit qualifications and evidence](ML_COMPLETED_030.md).**

The subsequent [fixed-cohort numerical diagnostic](ML_COHORT_032.md) also failed
the promotion gate: past selection improved error by 0.92%, while hindsight
offered only 5.90% on its three unchanged default recipes. It used 1,248 fits
and zero API calls, passed 11,235 independent checks, and is not a new agent
comparison. [Recorded negative result](evidence/ml-cohort-032.json).

The broader [hourly development screen 038](BROAD_SCREEN_038_RESULT.md) completed
416 electricity/pedestrian cases. Its primary historical selector was **4.98%
worse** than current CV; the secondary blend improved only **0.46%** overall.
The six-recipe hindsight diagnostic offered 21.69%, which the tested rule did
not capture. All 176,846 saved-evidence checks passed; 9,984 computations cost
838 seconds and zero API calls. No final outcomes were opened and no paid
confirmation was launched. This is a numerical mechanism screen, not a new
agent comparison.

[Calibration follow-up 039](BROAD_CALIBRATION_039_RESULT.md) also failed: using
past CV-to-outcome errors to correct current estimates was 0.26% worse overall.
Its 9,248 verification checks passed, with zero new forecasts/API calls. The
original historical overrides helped 129 cases and hurt 157; larger downside
outweighed gains even in electricity, where helpful overrides were more common.
No final outcomes were opened.

## Earlier frozen confirmation and development history

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

Continuation screen 012 tested whether past matured ledger overrides could
authorize future overrides. Its training-selected gate was 0.56% better than
current CV but 1.37% worse than existing support on the later development slice.
It was also rejected without an API run or new confirmation access. See
[the retained result](OVERRIDE_TRUST_012_RESULT.md).

Audit 013 found that correct historical provider linkage beats all 256
deliberately mislinked diagnostic trials. However, the development panel's
future-aware improvement ceiling is only 12.65% versus current CV, or 9.14% on
its later slice. This diagnoses useful but limited historical signal, not a
live-agent superiority result. See [the audit](OPPORTUNITY_013_RESULT.md) and
[the gate for another forecast evaluation](FORECAST_EVALUATION_GATE.md).

Source amendment 014 prepares a prospective M5 panel: eight development series
and 24 reserved series, disjoint by store and item. Selection uses only the
initial history, and two clean preparations produce identical outputs. Preparation
computed no forecasts; reserved outcomes remain unscored.
See [preparation and limitations](M5_PREPARATION_014_RESULT.md).

The subsequent complete M5 development screen 015 is negative: support is 1.42%
worse than CV and the hindsight ceiling is 11.66% against CV. All 4,992 production
and CV forecasts passed independent checks, with no recipe fallback. Reserved
stores remain closed and no paid comparison was launched. See
[the result and pending scope decision](M5_CANDIDATES_015_RESULT.md).

The user's subsequent evaluation-development task is tracked separately in
[the accumulated-experience workflow benchmark](../experience_workflow/README.md).
It starts from arrived raw events and measures correct retrieval, revision review,
execution-bound decisions and agent work against persistent SQLite. Its new
efficiency objective does not replace or reverse this closed forecast-error result.

The product change adds selectable MAE/RMSLE ledger summaries, calculated
ranks/ties, matched sample counts, pairwise differences and recent/lifetime
disagreement. MAE remains the default; forecast arithmetic is unchanged.
See [metric documentation](../../docs/ledger-comparison-metrics.md).

## Earlier matched experiments

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
