# Broader task evaluation: source assessment, not a run authorization

Status: **draft pending the user's scope choice**. No new source archive,
forecast, agent run or reserved target has been opened for this proposal.
The failed retail experiments remain failed and the 20% target is unchanged.
The current version requirement remains Gnomon 1.2.0, with DeepSeek
deepseek-v4.1-flash through Engy if another agent evaluation is authorized.

The completed tests answer a narrower question: can historical comparisons help
these recipes on repeated retail tasks? They do not test a broad mix of
time-series problems. Another population must be selected for a concrete use
case before calculating its errors. Looking for a dataset on which ledger wins
would not establish the requested advantage.

## Sources checked through publisher descriptions

The [Monash archive catalog](https://forecastingdata.org/) lists related series
across several domains and provides research datasets in TSF format. Source
selection below uses the catalog's domain, length and series metadata, not its
baseline performance tables.

| Candidate | Published description | Proposed use case | Unresolved suitability work |
| --- | --- | --- | --- |
| [Electricity hourly, record 3898439](https://zenodo.org/records/3898439) | 321 hourly client series, 2012–2014; aggregated from higher-frequency electricity measurements | Recurring load forecasts, with backtesting and retuning | The record advertises a newer version. Pin the intended immutable record and verify measurement/aggregation units and timestamp interpretation before downloading data. |
| [Pedestrian counts, record 4656626](https://zenodo.org/records/4656626) | Catalog lists 66 hourly series with unequal lengths | Recurring footfall forecasts for staffing or capacity planning | Verify the record directly, sensor identity, coverage, timezone/DST representation, missingness and redistribution terms. The attempted record fetch timed out. |
| [Traffic hourly, record 4656132](https://zenodo.org/records/4656132) | 862 hourly freeway occupancy-rate series, 2015–2016 | Congestion/capacity forecasting | Rates have different units and magnitudes from retail counts. Do not rescale them after seeing RMSLE results to improve the pooled score. Network dependence also needs an explicit uncertainty treatment. |

The candidate source list is not a selected or pinned dataset manifest. In
particular, a DOI link alone is not proof that later targets are uninspected,
that source recording times are genuine, or that all underlying data rights
match the archive's wrapper license. Electricity's latest-record lookup was
rate-limited; no alternative version was silently substituted.

## Prospective rules needed before preparation

1. Specify operational target, native units, forecast frequency and horizon for
   each domain. Prefer a genuine day-ahead task for hourly load/footfall over
   carrying a 14-day retail horizon forward without justification. Record any
   recipe parameter changes symmetrically in every arm; do not introduce a
   stronger model only for ledger.
2. Freeze source record IDs, file names, publisher checksums and downloaded
   content hashes. Inspect schema and initial-history metadata before declaring
   eligibility. Choose development and reserved IDs by a fixed hash seed and
   identity/group rules, never by future errors or realized oracle headroom.
3. Define initial-history sufficiency and missing-value handling before targets
   are exported. Use the same calendar/availability conventions for all arms.
   If real publication/recording times are absent, label a synthetic visibility
   policy explicitly; do not infer ex-ante availability from a historical file.
4. Preserve native units. RMSLE is not invariant to multiplying a series by a
   constant because of its log(1+x) transform. Freeze domain weights before
   analysis, report every domain separately, and keep the agreed arithmetic
   mean per-case RMSLE as the primary metric. Do not replace that target with
   token savings, an alternative metric or a favorable subgroup.
5. Make the forecasting task identical across controls. Both controls and
   ledger receive the same history, known future covariates, raw matured
   outcomes, provider implementations, persistence opportunities, fit/API
   budgets and checkpoint completion rules. Native memory must be explicitly
   available under the same instructions if it is part of the control design.
6. Charge the cost of maintaining comparison cohorts. Forecasting unselected
   configurations creates future evidence but still consumes the numerical
   budget. Do not create that evidence for the ledger arm for free or silently
   change already submitted forecasts after outcomes mature.
7. Freeze development diagnostics and promotion rules before fitting. Retain
   all cold starts, unsuccessful calls, missing comparisons and exclusions.
   A restricted hindsight bound can reject a fixed-portfolio experiment, but
   cannot establish an achievable historical policy or select the final panel.
8. Keep final asset IDs and later targets sealed until a genuinely passing
   development candidate, frozen protocol and fairness audit exist. Account
   for common calendar shocks and correlated assets when specifying temporal
   blocks and resampling groups. A collection of sensors on one network is
   not automatically a collection of independent trials.

## Current decision boundary

Source/schema assessment and a draft protocol can be prepared without changing
the completed experiments. Starting a broader experiment is a separate scope
choice currently pending with the user. This draft does not reopen the reserved
M5 stores, relabel previously inspected Favorita series as held out, or claim
that changing domains will make a 20% improvement possible. No new paid run
should start merely because the current result is disappointing.
