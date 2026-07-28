# Preparing data

## CSV schema

Headwater requires one timestamp column, one numeric target column, and
optionally one series identifier column. Column names are not fixed; provide
them through `--time`, `--target`, and optionally `--series`.

```csv
timestamp,requests,service_id
2026-01-01T00:00:00+10:00,120,api
2026-01-02T00:00:00+10:00,128,api
2026-01-03T00:00:00+10:00,135,api
```

CSV files are read as UTF-8, including UTF-8 files with a byte-order mark.

## Timestamps

Timestamps use ISO 8601 forms accepted by Python's `datetime.fromisoformat`:

```text
2026-01-01
2026-01-01T14:30:00
2026-01-01T14:30:00+10:00
2026-01-01T04:30:00Z
```

Do not mix timezone-aware and timezone-naive timestamps. Headwater preserves the
provided offset but does not currently accept a separate named-timezone option.

## Supported frequencies

| Code | Meaning | Default seasonal period |
| --- | --- | ---: |
| `h` | Hourly | 24 |
| `D` | Daily | 7 |
| `W` | Weekly | 52 |
| `MS` | Month start | 12 |

Aliases such as `hourly`, `daily`, `weekly`, and `monthly` are accepted. When
`--frequency` is omitted, Headwater infers a supported frequency from timestamp
differences. Supplying it explicitly is preferable in automation.

Month-start data must use the first day of each month. Weekly data is any exact
seven-day sequence; v0.1 does not impose a particular weekday.

## Regularity and missing periods

The v0.1 missing-data and duplicate policies are both `reject`. Within each
series:

- timestamps must be unique;
- each timestamp must be exactly one configured period after the previous one;
- the target must be parseable as a number; and
- rows may arrive unsorted because Headwater sorts them before validation.

Headwater does not silently aggregate duplicate timestamps or impute missing
periods. Resolve those choices upstream so the transformation is deliberate.

## Panel data

A panel file contains multiple independent time series:

```csv
timestamp,requests,service_id
2026-01-01,120,api
2026-01-02,128,api
2026-01-01,42,worker
2026-01-02,45,worker
```

Pass `--series service_id`. Duplicate checking is performed within each series,
so the same timestamp can legitimately appear once for every service. Every
series must match the resolved frequency.

## History requirements

Headwater needs four disjoint evaluation origins in addition to initial model
history. For seasonal period `S` and forecast horizon `H`, the current minimum is:

```text
max(2 × S, 2 × H, 8) + 4 × H observations per series
```

For daily data with a seven-day season and a seven-day horizon, that is 42
observations. Shorter valid series produce an `unsupported` result rather than
an input error.

## Parquet

Install the optional dependency before reading `.parquet` or `.pq` files:

```bash
pip install 'headwater-forecast[parquet]'
```

The same logical column and temporal rules apply to CSV and Parquet inputs.

