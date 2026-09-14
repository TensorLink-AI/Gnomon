# Ledger optimization checkpoint

Branch: `dev/ledger-optimization`. **The 20% target has not been achieved.**
Development and a complete frozen confirmation are retained here. No release or
forecast-superiority claim is established.

Latest boundary check: [recording-visible review088](REVIEW_VISIBILITY_088_RESULT.md)
confirmed and fixed a development-adapter catalogue issue: future-recorded
executions could reorder an earlier query's first page despite correctly excluded
scores. The separate adapter validates public execution records before discovery.
All38integration checks and17unit tests passed. A failed constructor-byte audit
is preserved; the corrected verifier opens a copy. No accuracy gain is claimed.

Latest ledger-interface work: [concise agent review087](AGENT_REVIEW_087_RESULT.md)
preserves the100saved review packets' metrics and evidence references while
reducing mean per-review compact UTF-8 bytes by56.2%. Pinned1.2.0public-runtime
checks passed, including temporal revision visibility and pagination. This is
presentation fidelity/size evidence, not measured token, completion or accuracy
gain. Original agent results and final-evaluation gates are unchanged.

Latest completed numerical comparison: [case-norm risk086](NORM_RISK_086_RESULT.md)
scored0.249618mean-case RMSLE versus0.257140for its matched control, a2.93%gain.
It failed the20%target and electricity guards. Its0.061% lower mean than
[conditional risk081](LOCAL_RISK_081_RESULT.md) is tiny and not established as a
reliable improvement. [Prior-error context084](LAGGED_RISK_084_RESULT.md) also
failed. These are repeated-development numerical proxies, not new agent/API
evaluations, held-out evidence or shipped ledger features. The latest actual-agent
results remain [experiment030](ML_COMPLETED_030.md). No experiment is approved
for final confirmation; the final goal remains unmet.

The [risk-transfer diagnostic082](RISK_TRANSFER_082_RESULT.md) explains a current
limitation: risk estimates systematically prefer their own fitted actions, and
estimated benefit magnitudes do not reliably predict realised improvements.
All later source results, negative variants, costs and audit receipts are retained
in [JOURNAL.md](JOURNAL.md). Historical checkpoints follow.

Earlier offline mechanism test: [context-matched ensemble 047](BROAD_CONTEXT_ENSEMBLE_047_RESULT.md)
improved mean RMSLE **1.90%** against the equally capable CV ensemble on 416
development tasks (electricity 0.90%, pedestrians 2.17%). All 97,777 audit checks
passed; zero new provider/API calls. This is a small prototype gain, not a new
agent comparison or held-out proof. The >=20% gate failed; final data remain
untouched.

The [predictive-evidence audit 048](BROAD_EVIDENCE_SKILL_048_RESULT.md) finds that
history slightly improves model ordering while worsening overall calibration
of pairwise error differences. The 047 changes helped 278 tasks and hurt 137;
even a perfect hindsight rejection filter would offer only 4.22% on those
proposals. All 120,453 audit checks passed with zero new calls/fits. This directs
further development toward stronger proposals, not treating fitted objective
improvement as evidence of a 20% benefit.

[Shared residual correction 049](BROAD_RESIDUAL_MEMORY_049_RESULT.md) also failed
the gate: 0.85% better than CV-only correction, but just 0.21% better than the
stronger uncorrected ensemble. Both methods had the same correction tool; the
additional control exposed degradation caused by CV-only correction. All 416
tasks and 832 vector fits were audited (38,434 checks), with zero new API/provider
calls. No final data were opened and no rule was promoted.

[Shared intraday mixtures 050](BROAD_INTRADAY_050_RESULT.md) improved **2.04%**
against the equally capable intraday CV control (2.77% versus the older global
mixture). The ledger gain was 0.13% for electricity and 2.54% for pedestrians.
All 416 tasks and 832 fits completed; 60,485 independent checks passed. No new
provider/API calls or final-data access. This remains a failed 20% promotion
gate and a development prototype, not a new Hermes comparison.

[Learned historical risk 051](BROAD_CONDITIONAL_RISK_051_RESULT.md) did not improve
the best rule: 1.72% better than its matched quadratic control, but only0.95%
better than the stronger intraday control and2.32% worse on electricity against
that control. All416 tasks,52 evidence models and832 weight fits passed the
independent saved-model audit. No new forecasting-provider/API calls or final
access. The20% promotion gate failed; no paid agent confirmation was launched.

[Disjoint-series validation 052](BROAD_VALIDATION_052_MANIFEST.md) now locks the
050 rule and sixteen additional series selected by the original hash order.
Identity preparation passed 41 independent checks; no new-series scored outcomes
were read. Original final-reserved series remain untouched. Forecast execution
and validation scoring have not yet occurred; this is preparation, not a result.

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

[Opportunity audit 040](BROAD_OPPORTUNITY_040_RESULT.md) rules out another filter
as a standalone solution here: perfect hindsight among the existing proposals
improves at most 10.09%. Even perfect six-recipe selection after a three-origin
CV cold start offers only 19.71%. These are fixed-development diagnostic bounds,
not deployable performance. All 6,346 independent checks passed; no new forecasts
or final data access. Further work needs stronger proposals or earlier usable
evidence, not reinterpretation of these negative results.

[Earlier-evidence preparation 041/042](BROAD_WARMUP_042_RESULT.md) now supplies
125 usable warm-up origins for the same sixteen development series, retaining
all 416 scored tasks. Three earlier origins are explicitly unavailable because
of missing source rows. The strict attempt remains failed; the separate partial
evidence preparation passed 1,033 checks. No models were fitted during that
preparation; it was not an accuracy result.

The subsequent [warm-start screen 043](BROAD_WARM_SCREEN_043_RESULT.md) is complete:
the primary contextual rule was **1.84% worse** than current CV. Its best
prespecified diagnostic, a warm CV/history blend, improved only 0.60% overall
and harmed electricity. All 73,138 audit checks passed. Added cost: 3,000 forecast
computations and 416 contextual fits; zero API calls or final-outcome access.
The gate failed despite supplying prior evidence; no paid confirmation launched.

[Shared-ensemble screen 044/045](BROAD_ENSEMBLE_045_RESULT.md) tests forecast
combination equally for both methods. Ledger weights improved RMSLE **0.96%**
against CV-fitted weights, but harmed electricity and missed the gate. The
control ensemble's own 7.44% gain over hard selection is not ledger value.
All 43,441 checks passed after an explicitly preserved solver failure and
uniform tighter-tolerance rerun. No provider refits, API calls or final access.

[Ensemble opportunity audit 046](BROAD_ENSEMBLE_BOUND_046_RESULT.md) bounds
hindsight improvement at roughly 24.114% against that stronger control. These
optimization bounds are not confidence intervals or deployable performance.
They do not rule out 20%, but achieving it would require capturing about 83%
of the oracle gain. All 15,838 checks passed; no original model fits or final
outcomes were accessed. The achieved ledger gain remains 0.96% on development.

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
