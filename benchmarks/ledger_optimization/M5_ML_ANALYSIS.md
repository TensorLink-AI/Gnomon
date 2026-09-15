# Prospective M5 ML numerical analysis

Implemented with synthetic data only, while guarded093 development is running.
This records the exact calculation before any reserved targets or outcomes are
opened. It neither grants final access nor selects a new cohort. Final prompts,
agent seeds, source/runtime hashes, dispatch budget and fairness audits still
require the separate final freeze after the development gate passes.

`m5_ml_analysis.analyze(panel, seeds, rows)` has no file/network access. The
caller supplies the original frozen 24-series identity metadata and the two
prospectively selected agent seeds. The function requires the entire balanced
grid: eight stores, three globally distinct items per store, 26 origins, two
seeds and three arms, totaling 3,744 decisions. It rejects missing, duplicate,
unexpected or malformed decisions. The primary analysis includes cold starts,
invalid executions and baseline-only workflows, scored under the already frozen
common fallback rule. No success-only filtering or series replacement occurs.

The primary contrast is development ledger versus Gnomon 1.2.0 without ledger;
plain Hermes is secondary. The mean is the arithmetic mean of per-case RMSLE,
not pooled root-mean-square log error. Upstream independent auditing must verify
every score against its actuals, typed forecast or recorded fallback and clipping
policy. This numerical module does not establish provenance by trusting a score
field. It hashes the exact analysis inputs and reports completion counts,
per-store and per-origin means so that they remain independently checkable.

## Frozen resampling calculation

Use 5,000 paired bootstrap replicates, Python `random.Random(20260912)`.
Sort store identifiers lexicographically. For each replicate, first draw eight
store indices uniformly with replacement. Then draw seven independent start
indices uniformly from 0 through 25. Expand each to a four-origin consecutive
circular block modulo 26; concatenate and retain the first 26 origin indices.
Use this one origin sample for every sampled store and every arm. Keep all three
items and both seeds within each sampled store-origin. Never treat the items or
seed repetitions as independent stores. Averaging the six item/seed scores in
each store-origin before resampling is algebraically equivalent on this complete
balanced grid.

Compute `1 - mean(ledger)/mean(control)` within every replicate. Report the 2.5th
and 97.5th percentiles with linear interpolation at `(replicates - 1) * p`.
If a control mean is zero, its relative improvement is undefined. If any bootstrap
draw has a zero control mean, report that count and leave that contrast's interval
undefined; do not silently drop those draws or claim a significant improvement.

The primary numerical criteria require the ledger mean to be at most 0.8 times
the Gnomon control mean, with a strictly positive control mean and a defined
interval whose lower bound is strictly positive. The direct mean comparison
avoids rejecting exactly 20% due to subtraction rounding. The 20% target applies
to the point estimate, not to the interval's lower bound.

Even if those numerical criteria pass, `target_established` remains false here.
Goal completion additionally requires untouched final data, the complete frozen
run and independent fairness, visibility, cost and provenance audits. Eight
reserved stores provide limited diversity. Development monitoring retains its
existing explicitly exploratory calculation; no historical result is rewritten.

## Verification before real data

Synthetic tests cover a known exact-20% result, an independently calculated
heterogeneous bootstrap, row/panel/seed ordering invariance, shared circular
blocks and repeated store draws, complete-grid rejection, invalid scores and
flags, retained failed-workflow errors, and zero-control handling. They run no
providers or agent requests and access no real M5 archive or reserved targets.

## Full-objective reference comparison

This original three-arm helper alone does not establish improvement over the
prior ledger. `M5_ML_REFERENCE_ANALYSIS.md` defines the separate prospective
four-arm numerical contract that retains plain Hermes and adds a frozen ledger
reference. Existing three-arm outputs remain unchanged. The final reference
implementation and complete fairness/runtime freeze are still required before
any reserved data access.
