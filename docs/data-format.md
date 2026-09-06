# Data and time semantics

Inputs support CSV/TSV, JSON arrays, JSONL and gzipped text. Parquet and Excel
readers are optional extras. Values must be finite; timestamps must be parseable
and have consistent timezone awareness. Column names default to timestamp/value;
select other names explicitly.

One panel column can identify independent series. Select a specific returned
series ID for forecasting or description; Gnomon does not guess a business target.

Regular whole-second subdaily steps and daily, weekly and month-start calendars
are supported. A horizon counts grid steps, not calendar days in general.
Named zones and daylight-saving behavior matter for calendar operations.

Repairs are explicit: `off` rejects malformed input, `safe` applies disclosed
normalization, and `aggressive` permits bounded disclosed imputation.
Calendar regrids `business_daily` and `month_start` must be declared and are
checked for implausible filling/collisions. Repairs never establish model accuracy.

An inspected reference freezes the snapshot; changing the source does not change
that reference. Plain files assume availability at each row's timestamp.
Historical revisions require `TemporalStore`: valid time, source availability
and local recording time are distinct. Source and recording cutoffs are applied
before model input preparation.

See [data references](production/INFERENCE.md#frozen-data-references-and-exact-summaries)
and [revision-aware operations](production/OPERATIONS.md).
