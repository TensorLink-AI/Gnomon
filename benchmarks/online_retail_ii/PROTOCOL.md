# Online Retail II: prospective sales-forecasting benchmark v1

Frozen before inspecting product-level evaluation sales or running models.
Source: user-supplied `online+retail+ii.zip`, SHA-256
`572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`.
The archive contains `online_retail_II.xlsx`. Only its sheet names, headers and
first-row date/type metadata were inspected before this protocol.

## Question and comparison

Does Gnomon's execution/evidence interface help an agent choose forecasts more
accurately and reliably than an equally equipped agent, and does accumulated
ledger evidence add value beyond current rolling cross-validation?

Compare three matched agent arms: Hermes; Hermes + published Gnomon 1.2.0;
Hermes + Gnomon 1.2.0 + ledger. Use DeepSeek v4.1 Flash through Engy, requested
seeds 7 and 19, identical prompts except tool/evidence interface, numerical
libraries, current observations, CV tables, native-memory access and budgets.
All arms can read the same matured raw outcomes. The ledger treatment is their
validated retrieval and presentation, not privileged future data. Preserve
typed executed predictions; do not replace a successful unambiguous execution
because the agent's final prose is malformed. Multiple executions require an
explicit selection. Report strict final conformance separately.

The initial deliverable is an executable dataset/baseline/evidence harness and
a development smoke run. A deterministic historical selector is a diagnostic
policy, **not a substitute for the Hermes experiment**. No paid agent result,
general superiority claim or final-set promotion follows from that smoke run.

## Data contract

- Predict **recorded gross sales units**, not latent demand, profit or stockouts.
  Retain positive quantities and positive prices on non-cancellation invoices
  for United Kingdom customers and stock codes matching five digits optionally
  followed by letters. Exclude fees/non-product codes, cancellations, returns,
  nonpositive prices/quantities and malformed records with separate counters.
- Use Year 2009-2010 before 2010-12-01 and Year 2010-2011 from that date onward.
  This explicit ownership boundary prevents overlapping sheets double-counting
  December 2010. Keep repeated rows within the owning sheet: no line identifier
  establishes that repeated line items are accidental duplicates.
- Aggregate by StockCode and local calendar day, starting 2009-12-07. Do not
  carry customer identifiers, invoice identifiers or free-text descriptions
  into agent inputs. Fill no-transaction product-days with zero recorded sales.
  This is a sales convention, not proof of inventory availability.
- Dataset recording/revision timestamps are absent. Assume sales become known
  at local end of day (Europe/London); clearly label this simulated availability.
  Never claim a genuinely revision-aware source or infer promotion/stockout labels.
- Select the cohort using **only 2009-12-07 through 2010-11-30**. Eligibility:
  at least 24 sold units, eight distinct active weeks, 180 days between first
  and last sale, and a sale in the last 56 training days. Stratify on fraction
  of training days with sales: <=0.10, (0.10,0.30], >0.30. Within each stratum,
  select the first 16 stock codes ordered by SHA-256 of `online-retail-ii-v1:SKU`.
  If a stratum has fewer than 16, retain all and disclose the shortage; do not
  substitute after seeing test performance. Keep discontinued products.

## Rolling splits and isolation

Forecast the next **14 daily values**, every 14 days. Origins are Sundays from
2010-12-05 through 2011-11-20 (26 origins). Expand available history at each
origin; never access subsequent sales for fitting, feature construction or
selection. All arms finish an origin before its outcomes mature.

| Phase | Origin indices | Origins | Target endpoint |
|---|---|---|---|
| Development | 0–12 | 2010-12-05 … 2011-05-22 | 2011-06-05 |
| Validation | 13–18 | 2011-06-05 … 2011-08-14 | 2011-08-28 |
| Final | 19–25 | 2011-08-28 … 2011-11-20 | 2011-12-04 |

Final product sales are not materialized, profiled, scored or used for cohort
selection during development. Reading the XLSX container to filter dates is
not evidence access by an agent; reject future rows before processing quantities.
The partial last week of December 2011 is excluded. Freeze selected code,
runtime versions, cohort, agent settings and metric before one final evaluation.
Validation/final execution requires a separately frozen release manifest.
Chronological splits contain the same products; report this explicitly. This
tests within-retailer future performance, not transfer to unseen retailers.

## Strong numerical controls

Common registered candidates: last value; weekly seasonal naive; 28-day mean;
last-four-same-weekday mean; StatsForecast AutoETS(7), AutoARIMA(7), AutoTheta(7),
CrostonSBA; lagged Ridge on log1p sales; lagged histogram gradient boosting on
log1p sales. Forecasts are clipped at zero identically and clipping is counted.
Model configuration and package versions are saved. Model failures consume an
attempt, retain their error category and use the same seasonal-naive fallback.

Report every fixed candidate, **rolling-CV selection**, and an ensemble of the
three CV-best candidates (average in log1p space). Current CV uses three disjoint
14-day folds ending at the origin; refit preprocessing inside each fold. Select
by arithmetic mean fold RMSLE, breaking exact ties by published candidate order.
Historical selection uses only completed earlier forecasts, never CV residuals
presented as ex-ante outcomes. Its recent window is four matured origins; before
four origins are available, use current CV. Retain cold starts in the primary
denominator. A hindsight best candidate is optional diagnostic headroom only.

Gnomon and baseline paths must use the same numerical functions. Verify matching
request/provider/series/unit/timestamps and forecast parity through the published
Gnomon API before an agent run. The baseline table alone is not a Gnomon result.

## Metrics, completion and claims

Primary: equal-weight arithmetic mean **per-product/origin RMSLE**, matching the
existing 20% ledger hypothesis. Secondary: MAE, training-scaled seasonal MASE
(undefined zero scale reported), aggregate WAPE, bias, clipping, per-stratum and
cold/mature errors. Do not use MAE evidence to optimize an RMSLE primary score.
Compare against both matched no-ledger Gnomon and the strongest automatic control
selected on validation. Do not select the headline baseline using final scores.

Every planned case stays in the denominator; keep failures and fallbacks.
Record numerical attempts, fits, API requests, tokens, unknown usage, timeouts,
elapsed time and workflow completion. No fabricated billing dollar estimates.

Final uncertainty: 5,000 paired bootstrap draws, seed 20260916, resampling SKU
clusters and common contiguous two-origin blocks; preserve each draw across arms
and aggregate seeds within the same task. Report sensitivity to calendar-block
length, one-retailer scope and common demand shocks. The target is >=20% point
improvement with a 95% interval excluding zero, **not a promised outcome**.
Development/validation results cannot satisfy the untouched-final objective.

Sources: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii)
(Chen; DOI 10.24432/C5CG6D; CC BY 4.0),
[StatsForecast model reference](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html).
