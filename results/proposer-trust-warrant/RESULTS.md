# Result: precondition not met — the experiment is not run

Census run 2026-08-04 against `HYPOTHESIS.md` (registered at `e442593`).
Outcome recorded per the registration's own rule: *"Precondition census
< 25% and < 40 instances → the experiment is **not run**; recording that
outcome in RESULTS.md closes this registration honestly."*

## The census

`benchmarks/cik/classify_rejections.py` over a flag-on CiK proposing run,
`deepseek/deepseek-v4-flash-0731`. Partial grid: **200 of 355 runs — the
first 37 of 71 tasks in order**, so task families later in the ordering
are unmeasured.

Of 150 classified span-parse rejections:

| bucket | count | share |
| --- | --- | --- |
| `numeric_no_parse` | 139 | 92.7% |
| `non_numeric_claim` | **11** | **7.3%** |

Both pre-registered thresholds are missed by a wide margin: 7.3% < 25%,
and 11 < 40.

## What the numeric_no_parse examples are

The dominant failure is not directional claims needing trust — it is
spans that **do state a number, in a shape the parser cannot read**:

- "4 times the number of usual withdrawals" — a multiple of a baseline
- "3 times the usual electricity" — same
- "The maximal fan speed is 3000 rpm" — an attributive bound
- "the maximal pressure is 37.5 Pa" — same

Nobody needs to be trusted for these; the number is in the text. The
roadmap is therefore a **grammar fix** (multiples-of-baseline and
attributive bounds), not a new warrant.

## Status

**Deferred, not falsified.** The hypothesis about directional claims was
never tested — its own precondition ruled the experiment out. The bet
that motivated it ("proposals fail because claims are unverifiable")
was wrong on the measured grid: proposals fail because the parser is
narrow.

Reopening condition, per the partial-grid caveat: finishing the task
grid (tasks 38–71) could move the 7.3%. If the completed census crosses
either threshold (≥ 25% of span rejections or ≥ 40 instances), this
registration reopens exactly as written — no redesign, no threshold
adjustment.
