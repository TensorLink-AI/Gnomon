# Memory-v2 and FASE integration: regression results

Keep current routing defaults. Neither the new profiles nor the combined FASE memory
mode provides a general accuracy improvement across these tasks. Bounded distance
alone is promising for some crypto indicators, but does not improve alpha or Favorita
in this comparison. All new options remain explicit, opt-in experiments.

## What was checked and integrated

- `agent/episodic-memory-v2`: commits `63f6e5af` and `ea5118a1`, based on current main.
  Feature profiles/context, deduplication, optional statistical selection controls,
  switch penalty, diagnostics, replay acceleration and the ablation helper.
- `/tmp/gnomon-fase`: an uncommitted extension of those commits. Bounded group distance,
  18 univariate features and recent/long-term retention, with delayed distinctiveness
  receipts. These implement the earlier three-part memory plan as an adaptation.
- The release candidate retains the agent discovery/summaries from PR #128. Its existing
  defaults and episode feature identities are preserved. Live routing still admits
  eight providers; offline policy validation admits 32.
- Separate ranker, forecast-blending and LLM pilot experiments were inspected as
  existing work but are not added to the runtime here. This is not full FASE.

During integration, the ablation helper was corrected: after warm-up exclusion, its
headline scores, fixed-provider comparisons and switching counts now use the same
matched cohort as the paired loss difference. Accelerated replay preserves the new
neighbour evidence fields. Approximate standard errors are documented as descriptive,
not calibrated guarantees. Live FASE retention can read history beyond the usual
lookback; its latency can grow with ledger age.

## Matched results

Change in error versus **current episodic memory**, with unchanged saved model
forecasts. Negative is better. **Lower/higher** means the unadjusted paired 95%
time-block interval excludes zero; an unmarked result is uncertain. These are
exploratory comparisons across multiple tasks, not independent confirmations.

| Task | Memory-v2 profile only | Bounded distance only | FASE features + distance + retention | No-memory pooled selector |
| --- | ---: | ---: | ---: | ---: |
| Crypto 1-day RV | Same as current | -1.20% **lower** | -1.00% | -5.20% **lower** |
| Crypto 3-day RV | Same as current | -0.27% | +0.36% | -2.54% |
| Crypto 7-day RV | Same as current | -1.62% **lower** | -3.75% **lower** | +0.79% |
| Crypto 14-day RV | Same as current | -0.81% | +6.76% | +0.80% |
| Crypto 7-day downside | Same as current | -1.19% | -3.49% | -1.27% |
| Crypto daily range | Same as current | -1.11% **lower** | -0.64% | -3.89% **lower** |
| Alpha (all evaluated dates) | +0.25% | +0.27% | +0.74% | +3.60% **higher** |
| Favorita | +0.46% **higher** | +0.32% | +2.00% **higher** | +2.16% **higher** |

The full FASE mode improves 7-day crypto RV on this regression, but worsens Favorita
by 2.00% (relative MAE 0.794452 → 0.810358). Favorita's difference remains above zero
under both time-block and series resampling. The intermittent profile also worsens
Favorita by 0.46%. Most other full-FASE differences have intervals spanning zero,
including the larger 14-day crypto deterioration. This does not establish that the
paper's full controller fails: it uses different models and an LLM/learned policy.

Bounded distance has lower error in all six crypto point estimates, with intervals
below zero for 1-day RV, 7-day RV and daily range. The stronger comparator matters:
no-memory pooling is still better than current memory for 1-day RV and daily range.
Conversely, current memory beats no-memory pooling on alpha and Favorita. There is
no evidence here for universally replacing one selector with another.

Alpha mixes its prior regression period with a short 35-origin extension. Separated
results are in `results.json`: FASE and the returns profile improve in that short
extension, but not over the full evaluated history. The short extension and prior
inspection make this insufficient to select new production settings.

## Evaluation contract and coverage

- Crypto: the same 12 assets, 20 forecast configurations, six future indicator targets,
  and July 2024–March 2026 scoring windows. Targets are 1/3/7/14-day realised volatility,
  7-day downside and daily range. Complete targets only; 7,197–7,353 scored predictions
  per task. Related configurations and an ensemble are not 20 independent model families.
- Alpha: 24 original subnet series, four original forecast models, 7,872 scored
  predictions after the original 120-origin warm-up. 7,032 are in the prior regression
  period and 840 in the extension. Both are previously inspected data.
- Favorita: the original 96 series, six StatsForecast candidates, 56-step forecasts,
  150-origin warm-up and last 300 scored origins: 28,800 scored forecasts. Score is
  mean per-series MAE divided by that series' seasonal-naive MAE.
- All arms retain original target maturity, explicit windows, k, own-series weighting,
  thresholds, baseline eligibility and loss definitions. Crypto/alpha use mean
  per-origin RMSLE. No models were refit, no paid calls were made and no new outcomes
  were acquired. The defaults use the original features; profiles change only those
  features, bounded changes only distance, and the FASE arm changes three mechanisms.
- No shrinkage, confidence penalty, novelty abstention or switch penalty was enabled
  in these comparisons. Those options pass functional tests; this report makes no
  accuracy claim about enabling them. Crypto's `levels` profile equals current and
  its duplicate replay was omitted.
- Paired circular blocks keep all series at each origin together: 28 origins for
  crypto/alpha and 112 for Favorita, 2,000 draws, seed 0. Favorita also has paired
  series intervals. No multiplicity correction; no-memory was added as a descriptive
  control after execution began. No candidate settings were tuned after scoring.
- These are **regression results on previously inspected data**, not a fresh holdout.
  Full FASE changes features and retention together, so their individual contributions
  cannot be isolated by this combined arm.

## Reproduction and validation

Runtime commit: `c7b4c6e` (same Python source fingerprint for all 34 runs).
Pre-integration reference: `b741aa68bd86d534753ba3b8e5788d720c8c3543`.

- All 24 crypto current/bounded/FASE/no-memory arms reproduce archived model choices
  and loss scores; alpha's prior-period and Favorita's current-memory scores reproduce
  the original regression to 1e-12. See [archive parity](archive_parity.json).
- Current arms check 32 sampled decisions per task against the pre-integration release
  implementation: 256 checks, preserving provider, reason and evidence level.
- Integrated local suite: **1,441 passed, 30 optional-environment skips**. All eight
  GitHub checks passed on the runtime commit, including supported Python versions,
  package/MCP, Hermes compatibility, service isolation and container build. Pure-Python
  and NumPy parity, delayed outcomes, missingness and retention are covered.
- Clean wheel/sdist builds, Twine and installed-wheel/provider-plugin/MCP smoke pass.
  NumPy is an optional `replay` extra; core installation remains dependency-free.

[Protocol](protocol.json), [runner and input requirements](README.md),
[paired results and intervals](results.json), [policies and artifact hashes](replay_manifest.json).
Raw per-origin outputs remain in `/tmp/gnomon-140-memory-eval-v2`; input archives remain
unchanged under `/tmp/gnomon-fase/experiments`. Hashes bind these local artifacts to
this report. Fetch the reference commit if using a shallow checkout before replaying.
The source inputs are required to reproduce; they are not distributed with the package.
