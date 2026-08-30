# v0.7 loop I001: production-selector baseline

Status: **frozen synthetic gate passed; no selector change; S1 pre-empts Q1.**

The exact-head run at `b49b0a3` exercised the production selector on all 200
preregistered cases. The final horizon was unavailable to selection and TSFM
discovery was disabled. Product publication and engine support were counted
separately.

## Result

- Product completion: 200/200. The engine supported 187 cases; the remaining
  13 zero-scale intermittent cases became explicit `best_effort` last-value
  publications, matching the production boundary rather than being counted as
  missing forecasts.
- Departures from last-value: 77/200; 68 wins and 9 losses, or 88.31%
  empirical admission precision.
- Median bounded relative MAE gain among departures: 65.72%; deterministic
  paired 90% bootstrap interval 43.11% to 74.03%.
- Every frozen gate passed: completion, no silent fallback, prefix-only
  selection, departure supply, precision, positive paired effect, uncertainty,
  family median non-inferiority, and provenance.
- The median effect over all cases was exactly zero because safe last-value
  preservation remained the majority outcome. The result supports selective
  departures, not an always-complex forecast policy.

## Response inspection

The raw retained records were inspected rather than relying on the aggregate:

- a supported `window_average` level-series win improved bounded relative MAE
  by 28.47%; another improved it by 71.95% but carried a correctly downgraded
  interval-coverage warning;
- two `window_average` level-series losses regressed by 13.77% and 4.53%,
  showing why aggregate precision is not universal safety;
- representative intermittent engine abstentions published unchanged
  last-value paths as `best_effort`, with the unscoreable-fold warning retained;
- support-tier mapping was corrected before this exact-head run so warned,
  non-degraded paths are `weakly_supported`, matching production.

## Decision

The current selector demonstrates useful, selective non-flat paths on the
synthetic mechanism screen, so the earlier TemporalBench saturation does not
justify loosening admission. Nor does synthetic evidence justify changing the
default: Q1 still requires a fresh naturalistic confirmation, and the
intermittent family contained one harmful departure despite a non-regressing
family median.

No production forecasting logic changed. The benchmark boundary gained a
faithful production-selector lane and now separates engine support from
product answer yield and mirrors production support tiers.

Per the preserved external-evaluation intake, S1 decision-input integrity and
then S2 timestamp jitter now pre-empt Q1 production work. Raw runs v1–v4 remain
preserved locally; v4 is the citable exact-head source.

