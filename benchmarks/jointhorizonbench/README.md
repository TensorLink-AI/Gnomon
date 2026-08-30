# JointHorizonBench

This serial, resumable engine benchmark evaluates P6's frozen 60-window
BreachBench corpus without calling an LLM. It preserves every pre-existing
forecast/risk/decision field as an immutable hash, scores threshold policy
cost, and—when a candidate exposes `cumulative_horizon`—scores realised-total
coverage and width against summed marginal bands.

Baseline:

```bash
uv run python -m benchmarks.jointhorizonbench.run \
  --output results/v06-p6-joint-horizon-baseline \
  --retries 1 --timeout 60
```

Matched treatment:

```bash
uv run python -m benchmarks.jointhorizonbench.run \
  --output results/v06-p6-joint-horizon-treatment \
  --baseline results/v06-p6-joint-horizon-baseline \
  --retries 1 --timeout 60
```

Each case is written atomically before the next begins. Existing case files
are resume checkpoints. Use a new output directory when code or corpus
identity changes; `run_identity.json` records the commit, dirty state,
generator, corpus hash, seed, retry policy, and single-worker topology.

The benchmark does not treat a blocked-bootstrap total as joint dependence or
automation evidence. The realised future is runner-side truth and is never
included in Gnomon's runtime packet.
