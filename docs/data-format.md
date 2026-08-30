# Preparing data

## CSV schema

Gnomon requires one timestamp column, one numeric target column, and
optionally one series identifier column. Column names are not fixed; provide
them through `--time`, `--target`, and optionally `--series`.

```csv
timestamp,requests,service_id
2026-01-01T00:00:00+10:00,120,api
2026-01-02T00:00:00+10:00,128,api
2026-01-03T00:00:00+10:00,135,api
```

CSV files are read as UTF-8, including UTF-8 files with a byte-order mark;
under the default repair level, non-UTF-8 files fall back to Windows-1252
with disclosure, and semicolon/tab/pipe delimiters are detected when the
header names your mapped columns. See "Other formats" below.

## Timestamps

Timestamps use ISO 8601 forms accepted by Python's `datetime.fromisoformat`:

```text
2026-01-01
2026-01-01T14:30:00
2026-01-01T14:30:00+10:00
2026-01-01T04:30:00Z
```

Under the default repair level, common non-ISO forms are also read and
disclosed as `timestamp_format_normalised`: `2026/05/18`, `05 Mar 2026`,
epoch seconds/milliseconds, and slash dates whose day/month order is
provable from the column (an unprovable order is a typed
`AMBIGUOUS_DATE_ORDER` error rather than a guess).

Do not mix timezone-aware and timezone-naive timestamps (`--repair
aggressive` assumes naive rows are UTC, with disclosure). Gnomon preserves the
provided offset but does not currently accept a separate named-timezone option.

## Supported frequencies

| Code | Meaning | Default seasonal period |
| --- | --- | ---: |
| `s` | 1 second | 60 (minute cycle) |
| `min` | 1 minute | 60 (hourly cycle) |
| `5min` | 5 minutes | 288 (daily cycle) |
| `10min` | 10 minutes | 144 (daily cycle) |
| `15min` | 15 minutes | 96 (daily cycle) |
| `30min` | 30 minutes | 48 (daily cycle) |
| `h` | Hourly | 24 |
| `D` | Daily | 7 |
| `W` | Weekly | 52 |
| `MS` | Month start | 12 |

The named codes are defaults, not the boundary: **any whole-second step
shorter than one day is supported**, written `<N>s`, `<N>min`, or `<N>h`
(for example `10s`, `7min`, `90s`, `2h`). Every duration has one canonical
spelling — named codes win, then the largest unit that divides the step
evenly, so `60s` is `min`, `600s` is `10min`, and `120min` is `2h`. The
default seasonal period for a general step follows the same rule the table
above encodes: the daily cycle when it fits in 288 lags, else hourly, else
a minute cycle; a step fitting no natural cycle assumes no seasonality and
relies on measured autocorrelation alone.

Aliases such as `hourly`, `daily`, `weekly`, `monthly`, `1s`, and market-style
candle codes (`1m`, `5m`, `10m`, `15m`, `30m`, `1h`, `T`, `5T`, …) are
accepted. When `--frequency` is omitted, Gnomon infers the frequency from
timestamp differences: a named step is inferred from the most common
spacing (gaps are then named by the grid validator, or filled by
`--repair aggressive`), while a general step is inferred only when the
series has exactly one unique spacing — an unusual step with gaps is
indistinguishable from a heavier grid with jitter, so it fails loudly
instead of guessing. Supplying `--frequency` explicitly is preferable in
automation.

Steps of a day or more stay named-only (`D`, `W`, `MS`): a fixed 48-hour
duration and "every second calendar day" diverge at the first DST
transition, and Gnomon refuses to guess which one is meant. Sub-second
steps are not supported.

Month-start data must use the first day of each month. Weekly data is any exact
seven-day sequence; Gnomon does not impose a particular weekday.

## Regularity, duplicates, and missing periods

Within each series the validated grid requires that:

- timestamps are unique;
- each timestamp is exactly one configured period after the previous one;
- the target is parseable as a number; and
- rows may arrive unsorted because Gnomon sorts them before validation.

What happens when a file falls short depends on `--repair`:

- `off`: strict rejection with a typed error, exactly as in v0.2.
- `safe` (default): cell text is normalised (formats, currency, sentinel
  missing values, byte-identical duplicate rows), and bounded timestamp
  jitter is aligned to a phase learned from the series. The bound is 1% of
  cadence, capped at 60 seconds. Alignment never merges observations or
  changes values; its cadence, phase, count, tolerance, and maximum
  displacement are disclosed. A genuine gap or conflicting duplicate is
  still an error.
- `aggressive`: the same bounded alignment applies, while interior gaps may
  additionally be linearly interpolated and conflicting duplicates resolved
  (last row in file order wins). Each fix is recorded in the artifact's
  `data_repair` evidence and surfaces as a `repaired_data:` warning that
  downgrades support. Value invention/selection is capped at roughly 30% of
  a series; timestamp alignment is counted separately and does not consume
  that ceiling.

Gnomon never aggregates or imputes silently: either you resolved the mess
upstream, or the artifact says exactly what was repaired.

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

Gnomon evaluates on disjoint rolling folds after an initial training window.
For seasonal period `S` and forecast horizon `H`, the training window is
`max(2 × S, 2 × H, 8)` observations, and evaluation adapts to how many folds
fit after it:

| Observations beyond the training window | Behaviour |
| --- | --- |
| ≥ 4 × H (four or more folds) | Full mode: separated selection, calibration, and final test windows. |
| 2 × H to < 4 × H (two or three folds) | Degraded mode: a forecast is produced as `weakly_supported`, with a "Limited evaluation" warning naming what was skipped. |
| < 2 × H | `unsupported`, with the exact required and available counts in the warning. |

For daily data with a seven-day season and a seven-day horizon, full mode
needs 42 observations and degraded mode starts at 28. Shorter valid series
produce an `unsupported` result rather than an input error.

## Other formats

Beyond comma-separated CSV, Gnomon reads:

- **`.tsv`** — tab-separated, always.
- **Semicolon, tab, or pipe-delimited `.csv`** — detected under the default
  repair level when the header provably names your mapped columns under
  that delimiter (common with European Excel exports); the detection is
  disclosed as a `delimiter_detected` repair action.
- **`.json`** — a top-level array of flat objects; **`.jsonl`/`.ndjson`** —
  one object per line.
- **`.gz`** — any of the text formats above, gzip-compressed (`.csv.gz`,
  `.jsonl.gz`, …).
- **`.parquet`/`.pq`** — with the `parquet` extra.
- **`.xlsx`** — first worksheet, first row as headers, with the `excel`
  extra.

```bash
pip install 'gnomon-forecast[parquet]'   # .parquet / .pq
pip install 'gnomon-forecast[excel]'     # .xlsx
```

Files that are not UTF-8 are read as Windows-1252 under repair, disclosed
as an `encoding_assumed` assumption (strict mode raises `INVALID_ENCODING`).
The same logical column and temporal rules apply to every format, and
`gnomon capabilities` reports which optional formats are installed.
