# Queued full-span replay

The [frozen protocol](PROTOCOL.md) uses the actual two-year archive, an initial
18-week warm-up, and 43 complete two-week forecast periods. It is a fresh
experiment: the existing 48-product full-year cohort and later agent memories
must not be reused at its earlier forecast origins.

The queue waits for the current 3,744-session run and its supervisor to finish
successfully. It then verifies the source, prepares the new warm-up-selected
48-product panel, runs numerical controls, runs six real-Hermes sessions with
synthetic API replies, and starts the 12,384-session paid comparison only if
that preflight passes. The paid comparison retains the same 36-session
accuracy-blind interface gate and all failures in its denominator.

Nothing in the active `/root/online-retail-agent-001/code` bundle is modified.
The sequel is isolated at `/root/online-retail-full-span-001`. Its `queue-001`
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

Warm-up-only preview found 112 sparse, 887 intermittent and 665 frequent
eligible products; the frozen hash selection retains 16 from each. The preview
used no post-warm-up quantities. A synthetic 12-case numerical smoke completed
all 13 model/control methods without fallbacks. Unit tests cover chronology,
future-independent cohort selection and the terminal-success queue gate.
The actual full-span Hermes preflight deliberately runs after the predecessor.

This consumes the source dates previously reserved for validation/final.
The full-span result is labelled exploratory, including previously observed
development periods; it is not an untouched-final proof of the 20% target.
