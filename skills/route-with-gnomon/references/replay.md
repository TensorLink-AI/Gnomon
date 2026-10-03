# Replay a router on your data

Build folds with a Gnomon evaluation study per series, then replay each policy. Replays
need many folds; the default evaluation budget allows 8, so raise it in the operator
TOML.

```python
import csv, math
from datetime import datetime, timedelta, timezone

# Two small daily series; replace with your own CSVs (timestamp,value).
start = datetime(2026, 1, 1, tzinfo=timezone.utc)
for name, phase in (("store-1", 0.0), ("store-2", 1.5)):
    with open(f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "value"])
        for i in range(200):
            w.writerow([(start + timedelta(days=i)).isoformat(), 50 + 10 * math.sin(i / 7 + phase) + (i % 5)])

open("gnomon.toml", "w").write("schema_version = 1\n[evaluation_limits]\nmax_folds = 500\nmax_calls = 2000\n")

from gnomon import GnomonSession
from gnomon.adaptive_router import replay_router, validate_policy

models = ["seasonal_naive", "last_value", "historical_mean"]
folds, histories = [], {}
with GnomonSession.from_config("gnomon.toml") as session:
    for name in ("store-1", "store-2"):
        ref = session.call("gnomon_inspect", {"input": f"{name}.csv", "unit": "units",
                                              "purpose": "evaluate"}, compact=False)["data_ref"]
        study = session.call("gnomon_evaluate", {"data_ref": ref, "baseline": models[0],
                                                 "candidates": models[1:], "horizon": 7, "folds": 60,
                                                 "stride": 1, "season": 7,
                                                 "budget": {"max_folds": 60, "max_calls": 180}}, compact=False)
        for fold in study["folds"]:
            if fold["status"] != "complete":
                continue
            actuals = sorted(fold["actuals"], key=lambda a: a["valid_time"])
            folds.append({"series_id": name, "origin": fold["origin"],
                          "target_time": actuals[-1]["valid_time"],  # matures when the last target is known
                          "actual": [a["value"] for a in actuals],
                          "points": {p: list(fold["runs"][p]["point"]) for p in models}})
        rows = list(csv.DictReader(open(f"{name}.csv")))
        histories[name] = [(r["timestamp"], float(r["value"])) for r in rows]

base = {"candidates": models[1:], "baseline": models[0], "min_origins": 5, "recent_origins": 20,
        "lookback_seconds": 90 * 86400, "pool": {"series": ["store-1", "store-2"], "own_weight": 2.0}}
memory = {"features": ["volatility_ratio", "trend", "level_shift", "cv"], "short_window": 7,
          "long_window": 56, "k": 32, "min_effective_n": 8, "own_weight": 2.0}
for label, policy in (("pooled", base), ("memory", {**base, "memory": memory})):
    result = replay_router(folds, validate_policy(policy), histories)
    print(label, round(result["router_score"], 3), result["fixed_provider_scores"], result["evidence_levels"])
```

Compare `router_score` with the best fixed provider, not only the baseline, and judge on
origins after a warm-up. `return_decisions=True` adds each origin's choice, evidence level
and label for held-out scoring; `covariates=` supplies the mask, future and `context` covariates for
memory features.

Then run the twin test: `memory_ablation(folds, validate_policy(memory_policy), histories)`
replays the same policy without memory and returns the paired difference with a 90%
bootstrap interval and a `verdict`. Use `accelerate=True` (needs numpy) for large replays.
