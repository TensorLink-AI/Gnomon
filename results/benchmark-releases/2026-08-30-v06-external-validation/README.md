# v0.6 exact-head external validation

This release evaluates product and harness commit `466d577` without changing
production behavior or tuning to benchmark labels. Runs used DeepSeek V4 Flash
0731 through Engy, bounded retries and timeouts, sequential execution, small
resumable shards, and the existing frozen task identities.

## TemporalBench

Both arms completed the same 80 T2/T4 tasks, 240 choice fields, and 480
forecast channels. Gnomon Evidence preserved 240/240 canonical typed answers
and 80/80 immutable primary forecast receipts, used one product call per task,
made zero redundant product calls, and sourced all 480 submitted channels from
Gnomon. Sixteen row-level Engy 502 failures were recovered; two initially
exhausted the retry ceiling and then completed through resumable replay.

Choice accuracy was 68/240 (28.3%) for Evidence versus 73/240 (30.4%) for the
direct control. Evidence fixed 52 fields and broke 57 (exact McNemar
`p=0.702`). Mean official row sMAPE was 10.641 versus 11.834; Evidence won 47
rows, lost 32, and tied one (exact sign `p=0.115`). Across 478 mutually
scoreable channels, Evidence won 195, lost 179, and tied 104 on MASE
(`p=0.438`). Heart-rate MASE improved significantly in isolation, but the
overall result does not establish general forecast or reasoning superiority.

## Context is Key

The fresh no-cache sensor slice covered four official tasks at one seed. Both
arms completed 4/4 with no abstentions or errors. Gnomon's capped/imputed mean
RCRPS was 0.10126 versus 0.13380 for the direct control, while its median was
worse (0.06375 versus 0.03143). Gnomon won one task and lost three; the one
large trend-accumulation win drove the lower mean (exact sign `p=0.625`). Every
Gnomon case used one product call, retained the immutable primary, and refused
to authorize automation. This is a useful heterogeneous diagnostic, not a CiK
superiority claim.

## Related ContextBench evidence

ContextBench was not rerun as part of this exact-head external comparison. The
retained v0.6 evidence remains complementary: the 80-case naturalistic
compiler evaluation improved mean sMAPE from 4.037 to 1.761 with 67 wins and
13 losses, while the later four-case agent-boundary probe completed 4/4 and
preserved all three applicable `no_distinct_numeric_path` relationships. The
full 112-case engine stress run retained zero leakage, perfect empirical
admission precision, and complete typed scenario contracts, but only about 46%
admission recall. These runs support contextual safety and governed execution;
they do not erase the mixed external TemporalBench and CiK findings above.

Raw responses, traces, receipts, checkpoints, and per-task artifacts remain
under `results/v06-release-validation-466d577/` and are intentionally excluded
from the aggregate git evidence.
