# Span-rejection census (measurement precondition)

Measures the precondition in `HYPOTHESIS.md`, "Measurement precondition".
Proposal and admission behaviour only: no score from this run is quoted
anywhere, here or elsewhere. Raw summary: `census.json`.

## Why a rerun was needed

The 2026-08-01..04 CiK dumps were classified first. Across six output
directories (1408 run dumps, 307 carrying a future-context gate), all
**522 of 522** span-parse rejections fell in `span_unrecorded`: those
dumps predate the span recording added in `e442593`, so none of them can
be bucketed. That is the condition the pre-registration names for making
one flag-on proposing rerun, so the rerun below was made.

## Run config

| field | value |
| --- | --- |
| date | 2026-08-04 |
| code revision | `10b2cf7` (claude/gnomon-harness-issues-hgerrj) |
| method | `gnomon-agent`, `--future-context` |
| model | `deepseek/deepseek-v4-flash-0731` (same as the original measurements) |
| seeds requested | 5 |
| max parallel | 4 |
| cache | `--no-cache` (the official HDF5 cache fails under current pandas) |
| grid | 71 CiK task classes x 5 seeds = 355 runs planned |

Command:

```
TMPDIR=$PWD/census-work .venv-cik/bin/python -m benchmarks.cik.run_cik \
  --method gnomon-agent --model deepseek/deepseek-v4-flash-0731 \
  --future-context --no-cache --seeds 5 --max-parallel 4 \
  --output-dir results-census
```

Classified with:

```
.venv-cik/bin/python -m benchmarks.cik.classify_rejections \
  "$PWD/census-work" --examples 10 --json census.json
```

## What actually ran

The run was **stopped after 200 gate-bearing runs of the 355 planned**,
two hours in (10:44–12:44 +1000). Two `BrokenPipeError`s in the
`multiprocessing` worker pool cost it workers about halfway through and
throughput fell from ~2.5 to ~0.6–0.9 runs/min; it was capped rather than
run to completion. The census is therefore over a truncated grid.

| quantity | count |
| --- | --- |
| runs reaching the forecast (gate-bearing dumps classified) | 200 |
| of those, scored runs | 162 |
| distinct CiK task classes among scored runs | 37 of 71 |
| task-seeds ending in short-history abstention before the forecast | 40 |
| runs carrying a future-context gate record | 99 |
| proposals across those gates | 189 |

The 40 pre-forecast abstentions are `GNOMON_ABSTAINED` short-history
refusals (for example, "Need at least 95 observations (have 77)"),
concentrated in the solar-forecast task families. No run errored for any
other reason.

Truncation caveat: the grid was walked in task order, so the 37 covered
task classes are a prefix of the 71, not a random sample of them.

## Admissions and rejections

| gate outcome | count |
| --- | --- |
| admitted, `constraint` | 16 |
| admitted, `override` | 10 |
| rejected, `span_states_the_value` | 135 |
| rejected, `span_states_the_bound` | 15 |
| rejected, `window_is_future_only` | 13 |

`window_is_future_only` is not a span-parse failure and is excluded from
the buckets below, per the classifier: it is a proposer error by
construction.

## Span-rejection buckets

Over the **150** span-parse rejections
(`span_states_the_value` + `span_states_the_bound`):

| bucket | count | share of span-parse rejections |
| --- | --- | --- |
| `numeric_no_parse` | 139 | 92.7% |
| `non_numeric_claim` | 11 | 7.3% |
| `parses_now` | 0 | 0.0% |
| `span_unrecorded` | 0 | 0.0% |

Every rejection in this run carried a recorded span, so nothing is
unclassifiable.

## Precondition branch

**Deferred:** `non_numeric_claim` is 11 of 150 span-parse rejections
(7.3%, against the ≥ 25% threshold) and 11 instances (against the ≥ 40
instances threshold), below both.

## `numeric_no_parse` examples (verbatim, 10 of 139)

```
4 times the number of usual withdrawals during that period
But in this case, residents sought to conserve energy and used lesser air conditioning, resulting in excessive usage of only 3 times the usual electricity.
The maximal fan speed is 3000 rpm
the maximal pressure is 37.5 Pa
At 05:27:09, it rapidly and smoothly changes to 1593.0
At 05:19:57, it rapidly and smoothly changes to 1594.4.
At 05:20:17, it rapidly and smoothly changes to 285.7.
The maximal fan speed is 3000 rpm
the maximal pressure is 37.5 Pa.
(2022-03-23 00:00:00, 0)
```

## `non_numeric_claim` examples (verbatim, 10 of 11)

Recorded for review beside the bucket that decides the precondition.

```
Assume that the ATM will not be in maintenance in the future.
Assume that the ATM will not be in maintenance in the future.
Assume that the sensor will not have this glitch in the future.
Assume that the sensor will not have this glitch in the future.
Assume that the ATM will not be in maintenance in the future.
Assume that the ATM will not be in maintenance in the future.
Assume that the ATM will not be in maintenance in the future.
Assume that the sensor will not be in maintenance in the future.
Assume that the ATM will not be in maintenance in the future.
Assume that the ATM will not be in maintenance in the future.
```
