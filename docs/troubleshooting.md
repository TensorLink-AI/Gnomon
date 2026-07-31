# Troubleshooting and error reference

## Start with inspection

Run the same mapping and frequency options through `inspect` first:

```bash
aion inspect data.csv --time timestamp --target value --frequency D
```

Successful responses go to standard output. Structured errors go to standard
error and return exit code `2`.

## Common structured errors

| Code | Cause | Resolution |
| --- | --- | --- |
| `INPUT_NOT_FOUND` | The path is absent or is not a file. | Check the working directory and path. |
| `UNSUPPORTED_INPUT` | Extension is not a supported format. | Use `.csv`, `.tsv`, `.json`, `.jsonl` (optionally `.gz`), `.parquet`/`.pq`, or `.xlsx`. |
| `MISSING_OPTIONAL_DEPENDENCY` | Parquet or Excel input without its extra. | Install the extra named in `details.install`. |
| `MISSING_COLUMNS` | A mapped column is not present. | Inspect the reported available columns and correct the option. |
| `EMPTY_DATASET` | The file has headers but no observations. | Supply at least one data row. |
| `INVALID_ENCODING` | The file is not valid UTF-8 (strict mode only). | Re-export as UTF-8, or use the default repair level (Windows-1252 assumed, disclosed). |
| `INVALID_TIMESTAMP` | A timestamp has no accepted reading. | Normalize the indicated row, or `--repair aggressive` to drop unparseable rows (capped). |
| `INVALID_TARGET` | A target value has no numeric reading. | Fix the indicated row, or `--repair aggressive` to drop such rows (capped). |
| `AMBIGUOUS_DATE_ORDER` | Slash dates could be day-first or month-first and no row proves the order. | Use ISO dates, or `--repair aggressive` to assume month-first (disclosed). |
| `MIXED_TIMEZONES` | Aware and naive timestamps are mixed. | Normalize the column, or `--repair aggressive` to assume naive rows are UTC (disclosed). |
| `AMBIGUOUS_FREQUENCY` | Too few timestamps or no supported interval dominates. | Supply more regular data or an explicit supported frequency. |
| `UNSUPPORTED_FREQUENCY` | The requested code is unsupported. | Use a code from `aion capabilities` (e.g. `min`, `h`, `D`, `W`, `MS`). |
| `DUPLICATE_TIMESTAMPS` | Conflicting values share a timestamp (identical rows collapse under the default repair). | Resolve upstream, ingest as revisions, or `--repair aggressive` (last row wins, disclosed). |
| `IRREGULAR_TIME_GRID` | A period is missing or spacing is irregular. | Fill/reindex upstream, or `--repair aggressive` to interpolate interior gaps and snap jitter (capped). |
| `FREQUENCY_MISMATCH` | Requested and inferred frequencies disagree. | Correct the frequency or input timestamps. |
| `INVALID_HORIZON` | Horizon is less than one. | Use a positive integer. |
| `EXCESSIVE_REPAIR` | Repair would touch too much of a series (~30%, or >5% dropped rows). | Fix the export at the source; forecasting a mostly invented series is refused. |
| `INVALID_REPAIR_LEVEL` | Unknown `--repair` value. | Use `off`, `safe`, or `aggressive`. |

The JSON error's `details` field often includes the row, value, expected next
timestamp, available columns, or detected frequencies, and every error carries
machine-readable `repair_options` naming the next actions.

## The run is `unsupported`

Unsupported is not a command failure. Read `results[*].warnings`. The usual
cause is insufficient history for separated selection, calibration, and test
windows. The warning reports the minimum observation count for that series.

Possible remedies:

- provide more history;
- shorten the horizon;
- choose a frequency that truthfully matches the decision; or
- accept that a defensible evaluated forecast is not available.

Do not duplicate observations or invent finer-grained values merely to pass the
history requirement.

## Parquet still reports unavailable

`uv tool` installations are isolated. Installing `pyarrow` into an unrelated
environment will not add it to the Aion tool. Install with the extra in the
same environment, then check `aion capabilities` and confirm that
`inputs.parquet` is `true`.

## Artifact write failures

Check that the `--output` parent is writable and that a file does not occupy the
requested directory path. Incomplete work uses a hidden temporary directory and
is never exposed as a completed forecast directory.

