# HierarchyBench

HierarchyBench screens the frozen P7 bottom-up point-total candidate before a
product surface is built. It creates exact three-leaf partitions of four
checked-in real series, forecasts the root independently, forecasts the leaves
as one wide panel, and compares the standalone root with the exact sum of leaf
q50 paths against a held-out realised future.

```bash
uv run python -m benchmarks.hierarchybench.run \
  --output results/v06-p7-hierarchy-baseline \
  --retries 1 --timeout 60
```

The run is local, serial, atomic per case, and resumable. Realised futures and
split-family labels remain scorer-side. This benchmark does not establish a
production hierarchy surface or a real-world superiority claim.
