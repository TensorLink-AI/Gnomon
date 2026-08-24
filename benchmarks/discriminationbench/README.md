# DiscriminationBench

Known-truth evaluation of `gnomon.discrimination` — the held-out
hypothesis-fit mechanism the evidence dossier reports. Every case is a
seeded synthetic series whose true interpretation is known by
construction (a trend, a persistent level shift, a volatility change, a
spike vs. a shift, or nothing), so the mechanism is falsifiable without
an LLM in the loop.

```
uv run python benchmarks/discriminationbench/run_discriminationbench.py
```

## Gates

| Gate | Meaning |
| --- | --- |
| `always_identifiable` | The generator only emits histories long enough for the three-window split; every case must be measured. |
| `accuracy_beats_chance` | `best` matches the truth well above the three-way chance rate. |
| `clear_separation_is_reliable` | Where the mechanism grades separation "clear", it must be right ≥ 90% of the time — the grade means what it says. |
| `truth_rarely_excluded` | The truth's hypothesis is listed with exactly zero weight in ≤ 5% of cases; being unsure is fine, ruling the truth out is not. |
| `no_confident_false_transitions` | On truly quiet series, a "clear" transition verdict is manufactured in ≤ 10% of cases. |

This measures the mechanism, not model uplift: whether an LLM consuming
the measured discrimination answers temporal questions better than one
consuming the descriptive packet is a separate, matched LLM experiment
under `docs/evaluation-protocol.md`.
