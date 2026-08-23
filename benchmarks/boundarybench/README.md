# BoundaryBench

BoundaryBench is a deterministic property benchmark for the agent-facing
reasoning contract. It generates multiple public verb shapes and checks that
the boundary preserves canonical results, gives every quotable fact one valid
source pointer, supplies a complete compact argument, attributes host calls
made after sufficiency as redundant, and makes rejection repairable.

It does not measure forecasting or LLM reasoning accuracy. Those remain the
job of TemporalBench and the matched provider-controlled evaluations.

```bash
python -m benchmarks.boundarybench.run_boundarybench --cases 100
```
