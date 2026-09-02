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

## Agent efficiency arm

The GFR one-retry case measures the agent turn saved by the executable plan.
Its control asks the model to derive a patch from the ordinary recovery
response; its treatment applies the recommended patch directly. Both arms
then give the same model the recovered answer to relay, and both must preserve
the `best_effort` tier and withheld automation. Completed provider samples and
their cumulative usage are checkpointed independently for each arm.

```bash
PYTHONPATH=src:. python -m benchmarks.recoverybench.run_agent \
  --model deepseek-v4-flash-0731 \
  --base-url https://api.engy.ai/v1 --api-key-env ENGY_API_KEY \
  --output-dir results/recoverybench/agent-efficiency
```
