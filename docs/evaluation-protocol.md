# Gnomon matched evaluation protocol

Status: preregistered for estimator and publication-policy changes after
2026-08-23. Changes to this document apply only to runs whose implementation
commits postdate the change and must be called out in the release manifest.

## Primary question

Does the governed Gnomon publication policy improve forecast error relative
to the registered robust baseline without hiding failures through abstention?

## Frozen comparisons

- The matched TemporalBench T2/T4 corpus contains all 80 selected rows and all
  480 forecast channels. Conditions use identical rows and official futures.
- The robust control is the last-value/registered baseline produced at the
  same cutoff. An LLM control is reported separately and is never called a
  model-admission baseline.
- Forecast arrays are scored with TemporalBench's unmodified official metrics.
  Per-channel MASE uses the same history, seasonality rule, and horizon in
  every arm.
- Boundary, leakage, and reasoning-choice results are separate product
  properties. They are not combined into a forecast-accuracy headline.

The four forecast arms are frozen before estimator work:

1. direct matched LLM (reported as a product comparator, never an admission
   baseline);
2. robust last-value/registered baseline;
3. enhanced classical plus within-file pooling;
4. the same enhanced engine with Toto2-4m eligible under its normal admission
   policy.

Primary comparisons are arm 3 versus arm 2 and arm 4 versus arm 3. Their
two-sided tests each use alpha 0.025 (Bonferroni familywise alpha 0.05).
Arm 1 comparisons are secondary.

## Power and strata

`python -m benchmarks.power_analysis` is run before paid evaluation. The
default design targets 80% power for a 55% non-tied paired win probability,
prices in a 10% tie rate, and uses alpha 0.025. A smaller run may be labelled
exploratory but cannot graduate an estimator. The calculation and parameters
ship with the release manifest.

The preregistered strata are history length (fold-starved / degraded /
fully-separated), channel family (vitals / operational / other), and
publication tier. Strata are descriptive unless separately powered before
the run; no post-hoc slice becomes a graduation claim.

## Outcomes and denominators

Report, before any pooled headline:

1. complete rows and forecast channels expected, produced, and paired;
2. publication, baseline-fallback, and abstention counts;
3. paired sMAPE and per-channel MASE wins/losses/ties, by tier and channel;
4. coverage with its eligible-case denominator;
5. calls, redundant calls, tokens, timeouts, and provider failures;
6. error with abstentions priced as baseline error and as a separate yield
   curve. An accuracy gain obtained only by suppressing hard cases does not
   graduate.

Every paired record is retained with task id, channel, stratum, arm values,
baseline value, support/admission state, and scoring eligibility. Aggregates
without their raw paired records are not publishable evidence.

Classify every eligible channel outcome as exactly one of:

- **uplift:** safely published and lower error than the robust baseline;
- **safety preservation:** exact robust fallback or an abstention priced at
  robust-baseline error;
- **regression:** higher error, an unsupported publication, or a missing
  result not covered by the preregistered abstention policy.

Safety preservation remains a valuable safety row and never counts as uplift.

The primary paired test is two-sided Wilcoxon signed-rank over row-level sMAPE
when its assumptions are satisfied; otherwise use a paired randomization test.
Per-channel direction uses an exact two-sided sign test excluding ties. Report
effect sizes and confidence intervals beside p-values. The nominal alpha is
0.05; exploratory slices are labelled and are not graduation gates.

## Graduation

A publication-policy change graduates only when:

- no named channel has a material paired MASE regression against its robust
  baseline, or that channel deterministically publishes the baseline;
- pooled error is non-inferior to the previous Gnomon policy;
- publication yield does not materially fall;
- long-history candidate choices are unchanged where the new gate is
  inapplicable, and numeric paths match within the documented tolerance;
- LeakTrap remains 0 leaks and boundary mutation/fuzz tests pass.

No dataset name, channel name, task ID, ground-truth future, MCQ option, or
benchmark label may enter runtime selection or admission logic.

## Reproducibility

Every published run records the evaluated commit, dirty-tree state, task-set
fingerprint, model/provider identity, configuration, seeds, completeness, and
curated-file hashes. A result produced by older code may be retained as
history but cannot be described as evidence for newer code.
