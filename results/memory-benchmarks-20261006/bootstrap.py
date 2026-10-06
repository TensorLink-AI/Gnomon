"""Recompute the documented six-benchmark improvements and exact bootstrap.

Uses only Python's standard library. Does not rerun forecasts or change results.
"""

import itertools
import json
import math
from pathlib import Path
from statistics import fmean


def percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def main():
    saved = json.loads(Path(__file__).with_name("summary.json").read_text())
    ratios = [r["memory_mase"] / r["no_memory_mase"] for r in saved["datasets"].values()]
    improvements = [100 * (1 - ratio) for ratio in ratios]
    logs = [math.log(ratio) for ratio in ratios]
    arithmetic, geometric = [], []
    for sample in itertools.product(range(len(ratios)), repeat=len(ratios)):
        arithmetic.append(fmean(improvements[i] for i in sample))
        geometric.append(100 * (1 - math.exp(fmean(logs[i] for i in sample))))
    result = {
        "bootstrap_resamples": len(arithmetic),
        "arithmetic_mean_improvement_percent": fmean(improvements),
        "arithmetic_mean_95ci_percent": [percentile(arithmetic, p) for p in (.025, .975)],
        "geometric_mean_improvement_percent": 100 * (1 - math.exp(fmean(logs))),
        "geometric_mean_95ci_percent": [percentile(geometric, p) for p in (.025, .975)],
    }
    for key, value in result.items():
        actual = value if isinstance(value, list) else [value]
        expected = saved[key] if isinstance(saved[key], list) else [saved[key]]
        assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
                   for a, b in zip(actual, expected)), key
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
