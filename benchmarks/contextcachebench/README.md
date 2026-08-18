# ContextCacheBench

ContextCacheBench compares the normal context-event product path with replay
through a persistent `context_ref`. It requires identical published points,
receipt and numeric-assessment cache hits, and at least 80% less context
argument payload. Generated futures and event schedules never enter cache
identity or product execution.

```bash
python3 -m benchmarks.contextcachebench.run_contextcachebench \
  --output-dir results/contextcachebench
```
