# Usefulness-roadmap validation — 2026-08-23

This release evaluates product commit `10b830a` under the protocol frozen in
`docs/evaluation-protocol.md`. It separates three outcomes throughout:
**uplift** beats the robust baseline, **safety preservation** matches it, and
**regression** loses to it. Matching last-value is never counted as uplift.

## External matched result

Both TemporalBench arms used DeepSeek V4 Flash 0731 through Engy at
temperature zero on the same 80 T2/T4 tasks (240 choices and 480 forecast
channels). All tasks completed. Provider HTTP 502s were recovered by bounded
row-level retry and remain disclosed in the Evidence summary.

| Metric | Direct DeepSeek | Gnomon Evidence | Paired result |
| --- | ---: | ---: | --- |
| Choice accuracy | 70/240 (29.2%) | 68/240 (28.3%) | 53 fixed, 55 broken; McNemar p=0.923 |
| Mean row sMAPE | 11.395 | 10.623 | 47 wins, 32 losses, 1 tie; sign p=0.115 |
| Median per-channel MASE | 0.793 | 1.089 | 191 wins, 183 losses, 106 ties; sign p=0.717 |
| Completion | 80/80 | 80/80 | equal |

Neither accuracy difference establishes superiority. Gnomon's 480 channels
all exactly matched last-value: zero uplift, 480 safety-preserving outcomes,
and zero regressions. That is honest protection against a weak selected model,
not evidence that Gnomon produced a better forecast. The run had one product
call per task (median and p95), zero redundant product calls, complete typed
answer coverage, and unchanged primary forecast receipts.

This 80-task run is diagnostic rather than a powered superiority test. The
preregistered power calculation estimates roughly 1,054 total pairs are needed
for 80% power for the stated modest effect (55% wins, 10% ties, Bonferroni
alpha 0.025).

## Independent product validation

- LeakTrap proved cutoff-bounded access on 40/40 tasks, with zero future
  transcription and zero tasks flagged as leaking.
- BoundaryBench graduated all falsifiable response-boundary gates on 100
  adversarial cases.
- AdmissionBench passed every gate across 1,000 generated admission cases.
- AdapterBench graduated the statistical adapter contract; no external TSFM
  was installed in the evaluation environment.
- ShortHistoryBench passed its preregistered gates at three seeds. Strong,
  comparable panels admitted pooling on 97.5–100% of cases with 89.7–92.5%
  admission precision; null and mixed-direction panels admitted none. The
  classical tier's median gain was positive at all three seeds. Raw case
  outcomes include and retain regressions rather than averaging them away.

These generated tests establish mechanism behavior, not customer uplift.
The short-history pooling mechanism did not activate on the external
TemporalBench panels, so the external result remains last-value safety
preservation. PyPI publication and real-user outcome evidence are explicitly
outside this release: the former needs release credentials after merge/tag;
the latter can only be earned from deployed, resolved receipts.

The JSON files here are aggregate, digest-pinned summaries. Raw paired records
remain in the untracked run artifacts named by each release file's provenance.
