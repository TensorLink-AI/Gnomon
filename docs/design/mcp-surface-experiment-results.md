# MCP surface experiment: Phase 3 decision

Date: 2026-08-12
Code revision measured: `0c97e29` (contains Phase 3 head `72571ed`)
Model: `deepseek-v4-flash-0731`
Endpoint: `https://api.engy.ai/v1`

## Decision

Keep `full` as the default MCP profile. All four candidates were run against
the same first T1, T2, T3, and T4 TemporalBench rows with the same model,
endpoint, temperature, prompts, and harness caps. None passed the
pre-committed default gate.

The result rejects a default-profile change; it does not say that smaller
surfaces are useless. `core` reduced mean conversation tokens by 19.4% without
losing a row in this run. `evidence` was cheapest, but still missed both
economics targets and answered only three of four rows. `mega` had the smallest
schema, yet used slightly more conversation tokens than `core` and routed five
forecast channels through model-authored values rather than Gnomon artifacts.
Tool count and schema bytes were not reliable proxies for completed-task cost.

## Matched result

| Metric | `full` | `core` | `describe` | `evidence` | `mega` | Gate |
|---|---:|---:|---:|---:|---:|---:|
| Matched rows | 4 | 4 | 4 | 4 | 4 | same rows |
| Schema bytes, maximum | 42,102 | 20,351 | 22,833 | 12,343 | 7,936 | <= 12,000 target |
| Mean tokens / attempted row | 120,095 | 96,824 | 117,607 | 71,123 | 97,499 | <= 50,000 |
| Median calls | 4 | 4 | 4 | 4 | 4 | <= 2 |
| p95 calls | 4 | 4 | 4 | 4 | 4 | <= 4 |
| Answer yield | 4/4 | 4/4 | 4/4 | 3/4 | 4/4 | >= 80% |
| Forecast rows completed | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | no regression |
| Gnomon-routed forecast channels | 12/12 | 12/12 | 6/12 | 6/12 | 7/12 | disclose |
| Leaktrap | 0/40 | invariant | invariant | invariant | invariant | 0/40 |

The schema maximum includes the tier-specific `submit_answer` schema supplied
by the benchmark harness as well as the Gnomon tools. The `evidence` Gnomon
surface itself remains CI-pinned below 12 KiB; its maximum combined benchmark
schema was 12,343 bytes.

Choice accuracy was identical for `full`, `core`, `describe`, and `evidence`
on the scored rows: T1 0/4 fields, T2 1/3 questions, T3 0/6 questions, and T4
1/3 questions. `mega` scored 1/4 on T1 and matched the other three tiers.
Four rows are far too few for an accuracy claim, and no candidate reaches the
economics gate regardless.

The scored forecast means are also not an apples-to-apples model-quality
ranking when a profile leaves Gnomon's path. `full` and `core` used Gnomon
artifacts for all 12 forecast channels and both recorded OW_sMAPE 9.2409.
`describe` used six Gnomon and six informed-direct channels (mean 8.6945),
while `mega` used seven Gnomon and five informed-direct channels (mean 8.4872).
Those lower errors are model-authored exits, not evidence that either Gnomon
surface selected a better executable. `evidence` completed one forecast row,
through Gnomon, at 9.2409.

## Provider recovery and provenance

The initial `full` sweep completed T1 and T2, then Engy returned an upstream
504 on T3 and a transient model-not-found 404 on T4. The documented one-row
recovery path reran only offsets 2 and 3 with the same code, model, endpoint,
temperature, and caps. Both retries completed. The aggregate `full` figures
above are the sum of those four valid row records; provider failures are not
priced as product abstentions. Every other arm completed in one four-row run.

Raw run directories were written outside the repository under
`/tmp/phase3-current-{profile}-4`; the repaired control shards are
`/tmp/phase3-current-full-row2` and `...-row3`. They contain the manifests,
summaries, per-row records, and details used for this table.

## What this closes—and what it does not

This completes the pre-committed surface decision: retain `full`; keep
`describe`, `evidence`, and `mega` experimental; do not treat consolidation as
a product conclusion. A future candidate must pass the same gates on a larger
leakage-controlled corpus before changing the default.

It does not complete the broader product validation program. Four rows cannot
establish accuracy significance, and this adapter does not yet score
quote-versus-paraphrase, caveat survival, or repaired-call completion as
first-class metrics. Those remain follow-up agent-evaluation instruments, not
missing evidence that could reverse this decision: every candidate already
fails at least one binding economics, yield, or trusted-route gate.

## What Phase 3 changed

- executable `full`, `core`, `describe`, `mega`, and `evidence` profiles;
- profile, schema bytes, cumulative/mean tokens, answer yield, and call
  median/p95 in summaries and manifests;
- `--offset` one-row shards for recoverable matched runs;
- a four-tool browsing cap, ten-round cap, final submit-only round, and
  recovery of a complete verified artifact after formatting failure;
- CI tests for profile membership and schema budgets.
