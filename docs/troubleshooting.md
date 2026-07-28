# Troubleshooting and error reference

## Start with inspection

Run the same mapping and frequency options through `inspect` first:

```bash
headwater inspect data.csv --time timestamp --target value --frequency D
```

Successful responses go to standard output. Structured errors go to standard
error and return exit code `2`.

## Common structured errors

| Code | Cause | Resolution |
| --- | --- | --- |
| `INPUT_NOT_FOUND` | The path is absent or is not a file. | Check the working directory and path. |
| `UNSUPPORTED_INPUT` | Extension is not CSV, `.parquet`, or `.pq`. | Convert the file or use a supported extension. |
| `MISSING_OPTIONAL_DEPENDENCY` | Parquet was requested without `pyarrow`. | Install the `parquet` extra. |
| `MISSING_COLUMNS` | A mapped column is not present. | Inspect the reported available columns and correct the option. |
| `EMPTY_DATASET` | The file has headers but no observations. | Supply at least one data row. |
| `INVALID_TIMESTAMP` | A timestamp is not accepted ISO 8601. | Normalize the indicated row's timestamp. |
| `INVALID_TARGET` | A target value cannot be converted to a number. | Repair or explicitly preprocess missing/non-numeric values. |
| `MIXED_TIMEZONES` | Aware and naive timestamps are mixed. | Normalize the entire timestamp column consistently. |
| `AMBIGUOUS_FREQUENCY` | Too few timestamps or no supported interval dominates. | Supply more regular data or an explicit supported frequency. |
| `UNSUPPORTED_FREQUENCY` | The requested code is unsupported. | Use `h`, `D`, `W`, or `MS`. |
| `DUPLICATE_TIMESTAMPS` | A series contains the same timestamp twice. | Deliberately aggregate or remove duplicates upstream. |
| `IRREGULAR_TIME_GRID` | A period is missing or spacing is irregular. | Fill/reindex upstream according to your chosen policy. |
| `FREQUENCY_MISMATCH` | Requested and inferred frequencies disagree. | Correct the frequency or input timestamps. |
| `INVALID_HORIZON` | Horizon is less than one. | Use a positive integer. |

The JSON error's `details` field often includes the row, value, expected next
timestamp, available columns, or detected frequencies.

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
environment will not add it to the Headwater tool. Install with the extra in the
same environment, then check `headwater capabilities` and confirm that
`inputs.parquet` is `true`.

## Artifact write failures

Check that the `--output` parent is writable and that a file does not occupy the
requested directory path. Incomplete work uses a hidden temporary directory and
is never exposed as a completed forecast directory.

