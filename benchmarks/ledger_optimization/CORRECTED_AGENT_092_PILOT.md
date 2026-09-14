# 092 completed pilot: reliable completion, worse ledger accuracy

All 36 pilot workflows completed; ledger accuracy was worse than both controls.
The 20% improvement objective remains unmet. This is four reused development
series, three origins per series and one requested seed, not final confirmation.

| Arm | Mean per-case RMSLE | Full workflows | Agent tokens | Agent requests | Numerical fits |
|---|---:|---:|---:|---:|---:|
| Hermes | 0.465567876 | 12/12 | 2,703,314 | 159 | 255 |
| Hermes + Gnomon | 0.456422367 | 12/12 | 2,726,055 | 151 | 231 |
| Hermes + Gnomon + ledger | 0.545557462 | 12/12 | 2,560,441 | 157 | 288 |

Ledger was **19.53% worse** than Gnomon without ledger and **17.18% worse**
than Hermes alone. Against Gnomon, ledger won four cases, lost seven and tied
one; its mean was worse on every series. The exploratory four-series bootstrap
interval for relative improvement was [-29.20%, -2.60%]. This small, reused
development sample does not support a general population or causal claim.
All origins are in the cold phase; mature and late-phase effects remain untested
in this pilot. No fallback, HTTP failure or corrective continuation occurred.

The ledger arm used 6.08% fewer agent tokens than Gnomon without ledger, but more
numerical fits and aggregate session time. This does not establish an overall
efficiency improvement. Total agent usage was 467 requests and 7,989,810 tokens
(7,870,063 prompt; 119,747 completion). Readiness probes added 36 requests and
504 tokens: **503 requests and 7,990,314 tokens including readiness**. Billed
dollar costs were unavailable, not zero. Earlier cumulative pilot audits overlap
this total and must not be added to it.

## Verification and interpretation

Independent saved-evidence recomputation passed 3,014 checks, with no integrity
failures, and reproduced the remote report and gate exactly. A separate wire
audit passed 1,473 checks; all 467 agent responses identified
`deepseek-v4.1-flash`. All 4,205 sealed original pilot files matched their remote
SHA-256 hashes. The archive's 4,215 files were then checked after compression.
These audits made zero provider or agent calls.

Published Gnomon **1.2.0** executed and stored forecasts. The ledger treatment
bundles the separately frozen 091 historical-comparison correction, 088/090
recording-visible catalogue and 087 concise evidence cards. It does not isolate
their individual effects. The query correction restored history visibility, but
the pilot gives no evidence that this improved forecast selection. The largest
loss diagnostic remains in `evidence/corrected-agent-092-live-audit-003.json`;
all arms selected the lowest cross-validation error within their own tested
portfolios. Its realized loss was not a fallback or scoring error.

A separate all-pilot coverage diagnostic found eight history-bearing reviews,
one for each session after the opening origin. Every returned pair comparison
contained only **one matched past origin**; no review was truncated by pagination.
Only three of those eight final configurations had any matched prior comparison
in the review (all three at origin two). The other five selected configurations
had no matched prior comparison. This describes evidence available in the review,
not proof of whether or how the agent used it. It does not excuse the worse
forecast scores or establish that more history will help. It shows why cold
completion cannot by itself establish the value of accumulated evidence.
The diagnostic reads all 12 ledger sessions, including four with no prior history;
its inputs match the sealed pilot, and it makes zero model or ledger calls.
See `evidence/corrected-agent-092-pilot-coverage.json` for hashes and the retained
correction to the diagnostic's initial expectation about opening review logs.

## Continuing experiment and preserved evidence

Pilot completion at 08:25:35 UTC on 14 September 2026 passed the predeclared
11/12-per-arm completion and integrity gate. The pipeline began the separate
312-session development evaluation at 08:25:40 UTC, with fresh homes and ledgers.
Accuracy was explicitly excluded from the promotion gate. The negative pilot
did not trigger a setting change, selected rerun or cancellation. The longer run
tests accumulation over 26 origins; its result is not yet available. Protected
validation055 and final M5 reserve data remain unopened. Main/PyPI are unchanged.

* Receipt and full arm summaries: `evidence/corrected-agent-092-pilot.json`.
* Archive: `results/corrected-history-092-pilot-bundle.tar.gz` (40,184,620 bytes).
* Archive SHA-256: `965bee06d7a3a4a367a0339df3ecb7281ba4d468aae6d2943d50a8240dc2e231`.
* Independent audit: `results/corrected-history-092-pilot-audit/`.
* Protocol and frozen source: `benchmarks/hermes_ml_checkpoint_v5/` and archive
  `pilot/frozen-source/`; source freeze `a7b50d1`, task-path correction `5c05bd0`.

The archive excludes the live development evaluation. To reproduce numerical
verification without overwriting archived reports, extract to a new directory
and call the public analyzer function from this development checkout:

```python
from benchmarks.hermes_ml_checkpoint_v5.analyze import analyze
report = analyze('/path/to/extracted/pilot', '/tmp/092-pilot-recomputed')
assert report['complete'] and not report['audit_failures']
```

This clarifies the archive README's abbreviated reproduction instruction: the
CLI takes only a root and writes there; the Python function accepts a separate
output directory. Do not run mutation drivers against original evidence.
