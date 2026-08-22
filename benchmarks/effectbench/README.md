# EffectBench

EffectBench evaluates the Context Effects Registry after outcomes accumulate:
hierarchical transfer to related series, knowledge-time filtering, held-out
event types, false influence, interval calibration, and robust decision regret.
Training effects are observed outcomes; test effects remain only in the sealed
oracle. A model cannot pass by learning benchmark case identifiers.

```bash
PYTHONPATH=src:. python -m benchmarks.effectbench.generate \
  --output-dir results/effectbench/corpus --seed 8127 --cases 80
PYTHONPATH=src:. python -m benchmarks.effectbench.run_effectbench \
  --corpus-dir results/effectbench/corpus \
  --output-dir results/effectbench/run
```

Run multiple seeds before a product decision. The hard gates require zero
false influence in the generated controls, abstention on held-out event types,
nominal 80% coverage that is not statistically rejected by the available
denominator, and no worse realized decision regret than always waiting. Pool
multiple seeds for a precise calibration decision; a single 20-case test split
is intentionally not treated as precise enough to reject 80% on point rate alone.
