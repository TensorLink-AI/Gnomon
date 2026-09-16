# Queued full-span replay

The [frozen protocol](PROTOCOL.md) uses the actual two-year archive, an initial
two-week warm-up, and 51 complete two-week forecast periods. It is a fresh
experiment: the existing 48-product full-year cohort and later agent memories
must not be reused at its earlier forecast origins.

The queue waits for the current 3,744-session run and its supervisor to finish
successfully. It then verifies the source, prepares the new warm-up-selected
48-product panel, runs numerical controls, runs six real-Hermes sessions with
synthetic API replies, and starts the 14,688-session paid comparison only if
that preflight passes. The paid comparison retains the same 36-session
accuracy-blind interface gate and all failures in its denominator.

Nothing in the active `/root/online-retail-agent-001/code` bundle is modified.
The sequel is isolated at `/root/online-retail-two-week-001`. Its `queue-001`
directory contains the plan, current stage, per-stage commands/logs/exits and
terminal receipt. `paid-001/progress.json` becomes available only after the
predecessor, preparation, baselines and preflight have completed. A waiting
queue is not a started paid comparison. Failed stages block subsequent stages
and are not automatically retried.

The queue entry point is `python -m benchmarks.online_retail_ii.full_span.queue`.
It requires an isolated code bundle with `inventory.json`, the source ZIP,
predecessor root and pinned runtime paths. Run it under the same numerical
Python used for the current baseline; credentials remain host-only and are not
read until the paid stage. Do not point it at a mutable checkout.

The archive contains Dec 2009–Dec 2011, not four years. Roughly 100 distinct
fortnightly periods require a longer real source; this runner never repeats
rows or concatenates products to create fictitious time coverage. The last
partial source week is excluded.

The earlier 18-week queue was cancelled while still waiting. This replacement
starts with 14 observations and empty memory. The first origin has no completed
backtest; models become available at their declared minimum history. Unavailable
fixed-model controls use labelled fallbacks; unavailable agent calls reject
before numerical execution. Ledger rankings never credit those fallbacks to
an unavailable model. 28 checks passed, including a 12-case synthetic cold-start run and direct/Gnomon
forecast parity. First-14-day cohort screening found 554 sparse, 915 intermittent
and 1,250 frequent eligible products; 16 per stratum are selected by the frozen hash.
The actual full-span Hermes preflight runs after the predecessor.

This consumes the source dates previously reserved for validation/final.
The full-span result is labelled exploratory, including previously observed
development periods; it is not an untouched-final proof of the 20% target.
