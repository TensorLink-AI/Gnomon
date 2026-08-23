# Product-hardening validation — 2026-08-23

This release evaluates product commit `2d14967` under the protocol frozen in
`docs/evaluation-protocol.md`. Both TemporalBench arms used DeepSeek V4 Flash
0731 through Engy at temperature zero on the same 80 T2/T4 tasks. All 160 task
executions completed; 11 exhausted-request 502 failures in Evidence were
recovered by bounded row-level retry and remain disclosed in its summary.

## Result

| Metric | Direct DeepSeek | Gnomon Evidence | Paired result |
| --- | ---: | ---: | --- |
| Choice accuracy | 74/240 (30.8%) | 80/240 (33.3%) | 66 fixed, 60 broken; McNemar p=0.656 |
| Mean row sMAPE | 11.341 | 10.623 | 43 wins, 36 losses, 1 tie; sign p=0.500 |
| Per-channel MASE | median 0.797 | median 1.089 | 183 wins, 190 losses, 107 ties; sign p=0.756 |
| Completion | 80/80 | 80/80 | equal |

None of the differences against the direct LLM establishes superiority.
Gnomon's short-history contract did meet its release gate: every one of the
480 published channels matched the robust last-value baseline exactly. The
previous release's statistically significant regression against last-value
is gone without channel names or benchmark labels entering runtime code.

Evidence used one product call per task (median and p95), made zero redundant
product calls, returned all 240 typed temporal answers, and preserved all 80
immutable primary forecast receipts. Its 480 channel labels were 362
`degraded` and 118 disclosed `best_effort`; these are not claims of supported
model superiority.

Balanced choice accuracy was 31.9% for Evidence and 24.1% for control, but the
class behavior remains uneven: Evidence improved level-direction and aggregate
seasonality balance while leaning heavily on `Uncertain` for volatility and
seasonality. This is diagnostic product evidence, not an accuracy claim.

## Boundary and operational validation

BoundaryBench passed all seven falsifiable gates on 100 cases at each of three
independent seeds. The complete test suite passed 1,787 tests. The packaged
Compose demo successfully queried Prometheus, wrote an immutable offline
report, persisted and delivered one idempotent webhook event, and emitted a
rule accepted by Prometheus 3.5 `promtool`.

The files in this directory are aggregate, digest-pinned release summaries;
raw task records remain untracked benchmark artifacts.
