# Source amendment 014: prospective M5 retail development panel

Registered before downloading or calculating scores on this source. The user
requested an evaluation that tests accumulating experience. This amendment
changes the retail source, not the 20% RMSLE target, forecasting capabilities,
equally informed control, or original failed confirmation result.

M5 is a longer daily store-item retail dataset accessible through Nixtla's
[documented source](https://nixtlaverse.nixtla.io/datasetsforecast/m5.html).
Pin the mirror to commit `72b8e7fd3b565b3c538adcb1d1a05117d8562d7e`, file
`datasets/m5.zip`, Git blob `925e92ee03cdb83c54a1dbbc12260a3b1939753c`, size
50,219,189 bytes. Verify the Git content identifier and record SHA-256 before
processing. Retain source attribution and use raw data only in ignored local
experiment storage. The mirror/code license is not asserted to license all
underlying competition data. This is a local numerical experiment, not dataset
redistribution or a claim of compliance with a customer's data requirements.

FreshRetailNet was considered because another local thread has an ROI plan for
it. Its [published 90-day scope](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
cannot supply this annual-history/26-origin schedule. That separate ROI work
and its data remain unchanged. No M5 source reference was found in the searched
local experiment manifests/receipts; this does not prove absence from all prior
work or the agent model's training data. No candidate scores were consulted to
choose M5. Public-benchmark memorization remains an evaluation limitation.

## Selection before forecast fitting

Use `sales_train_evaluation.csv` and `calendar.csv` only. Never use winner
forecasts, official test-evaluation targets, prices, or future weather. Fixed
window: days d_1212 through d_1941 inclusive, 730 dates. First forecast history
ends d_1577. Twenty-six consecutive 14-day horizons end at d_1941. Use the same
eight StatsForecast 2.0.3 recipes as the Favorita experiment and two rolling CV
folds. Freeze a separate preparation/candidate adapter before any fitting.

Selection uses only identity fields and d_1212..d_1577. Require a complete,
finite, nonnegative initial history and at least 28 nonzero initial observations.
Do not reject based on later sales, future model performance or oracle headroom.
Select stores by SHA-256 order of `20260912:m5:store:<store_id>`: first two stores
are development and the remaining eight reserved. Require exactly ten stores;
otherwise stop, do not alter the rule automatically.

Within each store, sort eligible items by SHA-256 of
`20260912:m5:series:<store_id>:<item_id>`. Traverse development stores first,
then reserved stores, in store-hash order. Choose four per development store and
three per reserved store, with no item repeated anywhere. Thus eight development
series and 24 reserved series, disjoint by both store and item. Insufficient
eligible items stop preparation without relaxing rules or rerolling the seed.
Retain all eligibility counts and rejected-prefix reasons.

Preparation writes full numeric target rows only for the eight development
series. Reserved output is identity/initial-history metadata and hashes only;
its later targets remain unread as numbers inside the pinned source archive.
CSV parsing may transit raw strings while scanning; this is not outcome analysis.
No reserved candidate fit, score, aggregate or plot before a passing development
candidate and a separate final implementation freeze. Do not call this a final
evaluation merely because its IDs have been reserved.

## Temporal and covariate meaning

Model daily sales, not uncensored latent demand. A published date denotes its
daily reporting period. Represent its period-end as next-day 00:00 UTC; this is
a uniform replay convention, not measured store timezone or publication time.
Assume sales become source-available and recorded at that period-end. Forecast
after receiving that day's sales, with future period-ends strictly later.
Only completely matured earlier forecast horizons may enter historical scores.
All arms receive the same arrived raw records and current evidence.

M5 has no matching observed `onpromotion` flag. Preserve the fixed provider code
and identifiers, but supply a constant-zero promotion feature explicitly labeled
`unavailable_assumed_zero`, plus calendar day-of-week features. Zero is a model
input convention, not evidence that there were no promotions. Do not turn future
prices into known promotion schedules. Report all numerical recipe fallbacks and
duplicate predictions; do not remove them to enlarge selection opportunity.
This covariate limitation may make the source unhelpful; retain that result.

## Progress and final evidence gates

First implement/test preparation, including future-target perturbation checks,
deterministic identity selection, store/item disjointness and output refusal on
existing paths. Freeze and prepare IDs before computing development predictions.
Then apply FORECAST_EVALUATION_GATE.md. An unpromising development result stops
paid testing; it must not cause selection of a different favorable final panel.

Final inference must cluster by reserved store (eight groups), keeping its three
items together, and use shared four-origin circular time blocks across stores,
retaining both agent seeds. There are not 24 independent stores. Use 5,000
bootstrap replicates with seed 20260912, and disclose limited geographic/store
diversity. Exact prompts, code, model configuration and final sample budget remain
unfrozen until development passes. No final API run is authorized by this file.

Keep the main metric, 20% point target, positive 95% improvement interval,
1.1.9 comparison, equal tools/information/budgets and common execution-bound
completion handling. No stronger treatment forecasting model, weaker control,
case deletion or inventory-profit substitution. Main and PyPI stay unchanged.

## Metadata adapter correction, before first successful preparation

The first attempt stopped at the calendar header: the pinned mirror omits `d`.
Its [published loader](https://raw.githubusercontent.com/Nixtla/datasetsforecast/main/datasetsforecast/m5.py)
reconstructs d_1, d_2, ... from one-based calendar row order. Apply that mapping
with an explicit consecutive-date check. Preserve the failed attempt log. No
development target export, forecast calculation or reserved-outcome analysis
occurred before this correction. Store/item sampling, prefix cutoff, dates and
source bytes are unchanged; no fallback seed or replacement cohort is introduced.
