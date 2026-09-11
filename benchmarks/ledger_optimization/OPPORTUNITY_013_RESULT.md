# Opportunity audit 013: linked history helps, target exceeds this panel's headroom

The frozen support rule uses meaningful historical provider information on the
existing development panel, but the fixed forecasts cannot improve 20% over
the deterministic CV reference even with future knowledge. This does not change
the failed live confirmation or establish the requested objective.

Protocol, implementation and tests were committed as `45dc81b` before computing
this diagnostic. It uses the previously inspected eight development series and
208 cases only. No policy tuning, API calls, provider calls or new holdout access.
No forecast values were changed.

| Reference / diagnostic | All 208 cases RMSLE | Later 64 cases RMSLE |
| --- | ---: | ---: |
| Current CV leader | 0.5658325868 | 0.5456698185 |
| Frozen past-only support | 0.5540610449 | 0.5352551237 |
| Best constant provider per series, using future outcomes | 0.5472321275 | 0.5292641209 |
| Best provider per case, using future outcomes | 0.4942338380 | 0.4957951754 |
| Median of 256 deliberately mislinked historical-provider trials | 0.5974127747 | 0.5817120982 |

Support improves 2.08% over CV on all cases and 1.91% on the later slice. It
captures 16.44% and 20.88% of the respective hindsight selection opportunity.
Even the per-case hindsight ceiling is only 12.65% overall and 9.14% later.
Reaching 20% would require 158% and 219% of that available opportunity. These
bounds apply to this CV reference and fixed panel, not to an arbitrary live agent
or an unseen dataset. Both hindsight strategies are non-deployable diagnostics.

Correct-history support has lower aggregate error than all 256 provider-label
corruption trials, both overall and on the later slice. The best corrupted
trial scores 0.5637927679 overall and 0.5365152873 later; both exceed correct
support's error. All trials and mappings were retained. No favorable corruption
seed was selected. The same mapping is used consistently within each series.
Current CV, actuals and candidate forecasts remain unmodified; only the copied
historical score-to-provider association is corrupted.

This establishes a narrow diagnostic fact: correct historical linkage matters
to this policy on these cases. It is not a fair live no-ledger comparison, a
confidence interval, a p-value, or proof of general memory value. Incorrectly
attributed measurements are deliberately harmful inputs. The objective requires
an equally informed capable control, not this corrupted-history diagnostic.

Cold origins 0–3 produce identical CV, true-support and corrupted-history choices
because fewer than four eligible past records exist. They remain in the overall
denominator. Mature origins show a 2.47% support improvement over CV and a 12.60%
hindsight ceiling. Removing cold starts would still not make 20% attainable here.

Ten new tests passed. Independent post-run checks reproduced 53,248 mapped
historical selections, 1,024 trial/slice aggregates, all 208 screen-007 incumbent
choices, source/input hashes and cached forecast arithmetic. Each selection
uses only earlier origins with fully visible outcome horizons. The source and
recording availability assumptions remain inherited synthetic replay assumptions.

Retained evidence:

- [All mappings, choices, scores and hashes](evidence/opportunity-013.json).
- [Independent audit counts and report hash](evidence/opportunity-013-audit.json).
- [Reproducible audit](audits/opportunity_013.py) and its
  [separate saved verification](evidence/opportunity-013-reproducible-audit.json).
- [Frozen diagnostic protocol](OPPORTUNITY_013.md).
- [Prospective forecast-evaluation gate](FORECAST_EVALUATION_GATE.md).

Decision: do not fund another confirmation on this panel to pursue 20%. Do not
weaken the control or choose final cases for favorable oracle gaps. Continue
evaluation design through development-only gates. Preserve the original failed
confirmation, all experiment costs and the unchanged target. Main/PyPI unchanged.
