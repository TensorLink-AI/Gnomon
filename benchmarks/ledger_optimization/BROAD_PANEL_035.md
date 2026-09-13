# Prospective source split 035: recurring day-ahead forecasting

Freeze this protocol and preparation code before inspecting observation values.
This extends development tasks beyond retail; it does not reopen the reserved
M5 panel or supersede the negative retail results. The primary goal remains
20% lower mean per-case RMSLE with a positive paired 95% interval on untouched
final tasks. Future Gnomon arms use 1.2.0, not the superseded 1.1.9 baseline.

## Operational task and immutable sources

Forecast the next 24 hourly observations, once per week for 26 weeks. The
use cases are recurring electricity-load and pedestrian-volume forecasting
with backtesting, model retuning, and accumulated outcomes. This is not a claim
about staffing savings, power-market trading or intervention decisions.

Use only the two verified archives in receipt 034:

- Electricity record 4656140, SHA-256
  `eff447075dde68dca0105ab7e2851c5637967ae3bb21556fd8b931f196d5968c`.
- Pedestrian record 4656626, SHA-256
  `6e81cb8cad43650e7e0f754a1103fc75c960e03387bae9b91146aa3241c9aa50`.

Keep reported numerical values unchanged. Label electricity units
`published_hourly_electricity_value` because the header does not fully establish
the aggregation's physical-unit transformation; pedestrian units are
`pedestrians_per_source_hour`. Do not rescale either series to improve RMSLE.

## Explicit time convention

TSF supplies a naive start label and regular hourly positions, not real source
publication times or a DST audit trail. For this replay only, map position i to
start-label + i hours as the start of a nominal source hour. Its valid/available
period-end is one hour later. Use UTC as a surrogate coordinate for those
nominal positions, **not as an assertion about the source's actual timezone**.
Hourly clock/calendar features are source-label features, not measured UTC
behavior. Do not claim that physical elapsed time through DST is reconstructed.

Assume source availability and recording at each nominal period-end. The same
assumption and arrived observations apply to every arm. Final target windows
close at 2015-01-01 00:00 for electricity and 2020-05-01 00:00 for pedestrians,
based on published end coverage. Last origin is 24 hours before that endpoint;
the preceding 25 origins are spaced 168 hours apart. This intentionally includes
the final published period, without screening for a favorable historical regime.

Each task has the most recent 730 nominal hourly observations. Three rolling
backtests use ends 658, 682 and 706, each followed by 24 targets. These indices
are a symmetric horizon adaptation of the previous three-fold lab. No fitting
is authorized by this preparation protocol; pin a separate numerical adapter
and common 60-fit budget before the development screen.

## Sampling and sealed reservation

Inspect source identity/start labels, row lengths and only the 730-hour history
ending at the first origin for eligibility. Require that the row covers every
scheduled history/target position by length, all initial-history values are
finite and nonnegative, and at least 28 initial observations are nonzero. Do
not inspect future values, volatility, model errors or potential gains to decide
eligibility. Duplicate source identities reject preparation.

For each source independently, sort eligible IDs by SHA-256 of
`20260914:panel035:<source>:<series_name>`. First eight are development; next
16 are reserved. Fewer than 24 eligible IDs stop preparation without replacement
dates, thresholds, seed changes or outcome-based substitution. There are 16
development and 32 reserved series, disjoint by source/identity. Asset identities
alone do not establish independence or eliminate common weather/calendar shocks.

Only after IDs are fixed may development values in the required 4,954-hour span
be parsed/exported. Nonfinite, missing or negative development values stop the
export and preserve the failure; do not silently choose a replacement series.
Reserved output contains IDs, time bounds, row lengths and initial-history hashes
only. It must not contain or numerically inspect reserved later values. The raw
archive necessarily holds their bytes; parsing/transiting strings is disclosed
and is not claimed to be physical separation from the source archive.

Verify selection invariance when later values are perturbed, deterministic
ordering, identity disjointness, period-end slicing, and refusal to overwrite
an existing preparation. Save exact source/protocol/code hashes, eligibility
counts/reasons, selected IDs and explicit temporal assumptions.

## Subsequent gates

Keep equal dataset weights via equal series/origin/seed counts and report each
domain separately. The unchanged primary score is arithmetic mean case RMSLE;
unit and source differences limit its interpretation. Before any final run,
freeze resampling that retains common calendar blocks and correlated asset
groups; do not count all sensors as independent evidence by default.

No final target export, candidate fit, aggregate or plot until a passing
development candidate and separate final freeze. The existing spending gate
still applies: no paid agent confirmation on a negative offline screen. All
controls receive the same recipes, raw arrived data, memory opportunities,
cohort-maintenance costs and checkpoint rules. No final superiority claim is
possible merely from preparing this panel.

## Grid-phase correction before successful preparation

Attempt 001 rejected a nonwhole-hour label before exporting data. Metadata-only
inspection shows the first three series in each source start at 00:00:01. Keep
the source's minute/second/microsecond phase at every nominal end boundary,
including the declared last-day endpoint. This changes only the coordinate
phase; do not round timestamps or move observations between rows. The source
dates, horizon, history, identity seed and eligibility thresholds are unchanged.
Preserve attempt 001 and test the one-second phase before another preparation.
