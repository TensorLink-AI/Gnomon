# RecoveryBench

RecoveryBench is the frozen P9 agent-boundary check. It runs six serial public
tool cases: three forecast recoveries that Gnomon can determine itself and
three failures that require an external choice. Candidate mode executes only
the exact tool patch published by Gnomon; it never parses repair prose.

```bash
GNOMON_MCP_PROFILE=full uv run python -m benchmarks.recoverybench.run \
  --mode baseline --output results/v06-p9-recovery-baseline

GNOMON_MCP_PROFILE=full uv run python -m benchmarks.recoverybench.run \
  --mode candidate --baseline results/v06-p9-recovery-baseline \
  --output results/v06-p9-recovery-candidate
```

The runner is jobs=1, uses one bounded retry and a 30-second per-case timeout,
and writes each completed case atomically so an interrupted run can resume.
