# v0.6 loop iteration I001: three-layer scorecard

Status: **P1 promoted; product smoke decision `revise`.**

This iteration added a fail-closed product scorecard that keeps three claims
separate:

1. whether the published forecast or decision is better;
2. whether the reasoning and agent boundary preserve the evidence contract;
3. whether the execution topology completes reliably within its registered
   budget.

The scorecard itself passed its completion gate. It requires complete case
accounting, visible metric denominators, internally consistent gates, exact
provenance, digest-verified retained evidence, and root-contained evidence
paths. A smoke, dirty-tree, incomplete, failing-layer, or failing-invariant
scorecard cannot promote a change.

## Smoke result

- Output: **fail** for promotion. The 10-case short-history classical shard had
  seven uplifts, two exact safety preservations, and one intermittent-series
  regression. Its median relative gain was positive, but the regression-free
  gate correctly remained false.
- Reasoning: **pass** on 20 BoundaryBench cases. All contract and mutation
  gates passed, including canonical immutability.
- Topology: **pass for smoke completion** on eight engine-only ContextBench
  cases. All eight completed and leakage remained zero. The shard is not
  decision-ready for an accuracy claim.

The combined scorecard therefore records `revise`, not `promote`. The output
failure is carried into P2, the short-history accuracy iteration.

## Defect found by the loop

The first curated build exposed that the aggregate-release scrubber removed
`rows` and `raw_results` but not the synonymous per-case key `raw_records`.
The scrubber and its tests now reject that key, and the release was rebuilt
without per-case data.

## Reproduce

```bash
uv run pytest -q \
  benchmarks/tests/test_scorecard.py \
  benchmarks/tests/test_release.py \
  benchmarks/tests/test_boundarybench.py \
  benchmarks/tests/test_short_history_modelbench.py \
  benchmarks/tests/test_contextbench.py

uv run python -m benchmarks.release validate \
  results/benchmark-releases/2026-08-30-v06-loop-i001

uv run python -m benchmarks.scorecard \
  results/benchmark-releases/2026-08-30-v06-loop-i001/scorecard.json
```

Focused result: **62 passed**.

The machine-readable handoff to P2 is in `checkpoint.json`. Raw smoke runs
remain ignored local artifacts; the committed JSON files are curated
aggregates with source hashes.
