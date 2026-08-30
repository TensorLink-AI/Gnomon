# SeasonalBench

SeasonalBench is the deterministic, paired seasonal-period and fold-starvation
screen frozen in `docs/v0.7-q1-seasonal-admission-protocol.md`. It gives the
production evaluator history prefixes only and uses each untouched six-step
future solely for scoring.

```bash
python -m benchmarks.seasonalbench.run \
  --output-dir results/v07-q1-seasonalbench
```

The runner appends one JSONL row after every case and resumes completed case
IDs. The aggregate summary keeps engine support, product completion, and
fallback disclosure separate.

