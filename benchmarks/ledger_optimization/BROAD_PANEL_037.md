# Source replacement 037: explicit pedestrian hour labels

Panel 035 failed its 24-series pedestrian coverage gate. Preserve that result.
Timestamp-only audit 036 of the original publisher archive found 43 sensors
with exactly one row at every one of the same 4,954 nominal hour labels.
No counts or model errors were used in this source decision.

Keep protocol 035's dates, 730-hour initial history, 26 weekly origins,
24-hour horizon, initial-history eligibility, eight development and sixteen
reserved series per source, identity hash seed, scoring objective and temporal
limitations. Replace only the pedestrian source with the SHA-pinned publisher
CSV from 036. Electricity remains the SHA-pinned TSF source from 034.

Use `sensor_<Sensor_ID>` as pedestrian source identity. Require complete unique
nominal-hour coverage and one sensor name throughout the window; reject name
changes rather than assuming location identity. This checks timestamp/identity
metadata, not future count values. Do not fill missing hours, merge duplicates,
rescale counts, substitute sensors, shift dates or lower quotas. The source's
naive wall-clock labels remain nominal hour bins mapped to surrogate UTC;
this is not a reconstruction of physical elapsed hours through DST. Even a
complete label grid does not prove all measurements were physically available.
Availability and recording are assumed at period end equally for all arms.

First validate immutable source hashes and audit 036 metadata. Parse only each
eligible sensor's first 730 count values. Require finite nonnegative values and
at least 28 positive observations. Apply the same checks to electricity initial
history. Freeze both sources' development/reserved IDs before parsing any later
development counts. Preserve initial-history hashes and rejection reasons.

Only development series' later values may then be parsed. Export all 4,954
values and source start labels for each selected development series. Stop on
any invalid development observation without replacements. Reserved later
values remain unparsed, including when the raw CSV strings pass the parser.
Do not claim the original publisher's counts equal TSF's processed values.

No model fitting, API calls or final evaluation are part of this preparation.
Keep the original negative retail results and the separate untouched M5 set.
A later numerical screen must freeze recipes and budgets before execution.
