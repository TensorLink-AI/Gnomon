# ReliabilityBench

ReliabilityBench is the frozen P10 local fault/load probe. It runs six cases
serially; two cases create exactly two synchronized same-ID artifact writers to
exercise the production publication race. There are no API or hosted-model
calls.

```bash
uv run python -m benchmarks.reliabilitybench.run --mode baseline \
  --output results/v06-p10-reliability-baseline

uv run python -m benchmarks.reliabilitybench.run --mode candidate \
  --baseline results/v06-p10-reliability-baseline \
  --output results/v06-p10-reliability-candidate
```

Each case has a 30-second alarm, one bounded retry, and an atomic resumable case
file. Use a new output directory for an independent replication.
