# A4 (analog pooling): stopped by its own kill criterion

**Not built.** The integration plan gave A4 an explicit kill criterion:

> Kill criterion: if Phase 1 instrumentation shows event supply below ~20% of
> tasks, pooling has nothing to pool and the spike stops.

Measured supply: **0%**.

## What was measured

Every benchmark run on disk, counting rows that supplied at least one context
event:

| run | rows | rows supplying ≥1 event |
|---|---|---|
| CiK `gnomon-pure` | 71 | 0 |
| MTBench `gnomon` (agent) | 50 | 0 |
| MTBench `gnomon-tools` | 50 | 0 |
| TemporalBench `gnomon-agent` | 50 | 0 |
| TimeSage `ts-gnomon` | 127 | 0 |

The `gnomon-tools` row is the one that settles it. That condition was built
specifically so a model could propose context events: `context_events` is an
explicit parameter of the `gnomon_forecast` tool, the schema is in the tool
description, and GLM-5.2 made 6.9 tool calls per sample. It proposed zero
admissible events across 50 samples.

## Why this kills A4 rather than deferring it

Analog pooling estimates an event's effect by borrowing from *other* series
that saw the same event type — a schema v4 `event_effects` table keyed on
(event_type, series fingerprint), with a leakage rule enforcing that an
analog is usable at a fold cutoff only if its own outcome was known by then.
All of that machinery is downstream of events arriving. At 0% supply the
table would be empty, the leakage rule would never fire, and the feature
would be untestable except against fixtures written to exercise it.

The measurement also points at where the real problem is. The bottleneck is
not that Gnomon pools event effects badly; it is that **nothing supplies
events**. Two plausible reasons, neither addressed by pooling:

1. The benchmark corpora do not carry dated, sourced events. CiK's context is
   prose; MTBench's is news text. Extracting a `ContextEvent` with a
   verifiable `known_at` from prose is the unsolved step.
2. Even given the tool, the model did not produce claims that clear
   `backtest_admissible` — which requires a source of a verifiable type, not
   an assertion.

Work on either of those changes the 0%. Work on pooling does not.

## What would revive it

A corpus where event supply exceeds ~20% of tasks — most likely an internal
one with a planning file or calendar as the source, since that is what
`VERIFIABLE_SOURCE_TYPES` admits. At that point re-run the count above; if it
clears the bar, the design in the integration plan stands as written, and its
leakage rule must land as a test rather than a docstring.

Related: [`shrinkage-admission-measurement-2026-08.md`](shrinkage-admission-measurement-2026-08.md)
records a separate limitation on the same path — the context candidate's
drift base means admission is rare even when events *are* supplied.
