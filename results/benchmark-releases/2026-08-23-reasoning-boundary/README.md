# Reasoning-boundary validation — 2026-08-23

This release validates Gnomon's reasoning-boundary implementation at product
commit `0ac55e4` with the same DeepSeek V4 Flash 0731 model through Engy for
both matched TemporalBench arms. The shard-recovery/reporting fixes are in
follow-up harness commits `7b3720a` and `2c03581`; they do not alter task
answers or product execution.

## Boundary contract

BoundaryBench generated 100 cases for each of seeds 20260823, 94721, and
27183. All 300 cases passed canonical immutability, fact-source traceability,
argument completeness, actionable rejection, and redundant-call attribution.
BoundaryBench is a safety-contract benchmark, not evidence of reasoning or
forecast accuracy.

The complete local suite passed: 1,755 tests, with 3 intentional skips.

## Matched TemporalBench

Both arms completed the same 80 T2/T4 tasks and 240 choice fields. There were
no terminal errors, abstentions, or voided rows. Seven Engy 502 failures in one
Evidence shard were recovered by the bounded infrastructure retry policy and
are retained in the summary.

| Metric | DeepSeek control | Evidence | Paired result |
| --- | ---: | ---: | --- |
| Choice accuracy | 76/240 (31.7%) | 80/240 (33.3%) | 65 fixed, 61 broken; McNemar p=0.789 |
| T2 choice accuracy | 40/120 (33.3%) | 40/120 (33.3%) | p=1.0 |
| T4 choice accuracy | 36/120 (30.0%) | 40/120 (33.3%) | p=0.699 |
| Mean row sMAPE | 11.931 | 10.756 | 45 wins, 35 losses; sign p=0.314 |
| Completion | 80/80 | 80/80 | equal |

The accuracy differences are not statistically demonstrated. The boundary
did demonstrate its intended behavioral guarantees: Evidence used one observed
and one surface-required product call per task (median and p95 both 1), made
zero redundant product calls, preserved 240/240 canonical answers, returned
240/240 requested typed engine answers, and preserved the primary forecast in
80/80 receipts.

Per-channel MASE is an important negative result. Across all 480 matched
channels, Evidence won 160, lost 225, and tied 95 (sign p=0.00108); median MASE
was 1.159 versus 0.800 for the control. Losses concentrate in respiratory
rate, SpO2, and systolic blood pressure; heart rate and diastolic blood pressure
are not significantly different. This roadmap improved the reasoning boundary,
not the underlying channel forecasters, and must not be presented as a general
forecast-accuracy win.

The merged transport-usage totals conservatively include one discarded
overlap execution per arm from parallel shard recovery. Task-derived calls,
tokens, accuracy, preservation, forecast metrics, and coverage are computed
from the canonical 80 unique records and are unaffected.

## Files

- `temporalbench-control/` and `temporalbench-evidence/`: final summaries and
  manifests.
- `choice-comparison.json`: exact field-level matched choice comparison.
- `matched-report.json`: exact task-level sMAPE comparison.
- `channel-comparison.json`: matched per-channel MASE and coverage.
- `boundarybench/`: the three independent seeded safety-contract summaries.
