"""Small exact statistics, shared so that every adapter reports the same ones.

Nothing here is novel; the reason it is one module is that a rate quoted
without an interval, or a paired comparison quoted without a test, is the
form in which a benchmark result stops being defensible. Two adapters had
already grown their own copies of Wilson intervals and exact binomial tails,
and a third was computing a McNemar p-value by hand in prose. Prose does not
re-run.

Exact throughout: sample sizes here are in the tens, which is precisely the
range where the normal approximations are worst and where the exact
computation is free.
"""

from __future__ import annotations

import math
from typing import Any


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the textbook normal interval because it stays inside
    [0, 1] and remains sensible at the boundaries — and a leakage rate of
    0/40 is exactly the boundary case a benchmark most needs an honest
    interval for.
    """
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials
                           + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def two_sided_binomial(successes: int, trials: int) -> float:
    """Exact two-sided p under p = 0.5."""
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k)
               for k in range(0, min(successes, trials - successes) + 1))
    return min(1.0, 2 * tail / 2 ** trials)


def mcnemar_exact(baseline: dict[str, bool],
                  treatment: dict[str, bool]) -> dict[str, Any]:
    """Paired test on binary outcomes over the matched subset.

    Reports the discordant counts alongside the p-value, because a p-value
    over three discordant pairs and one over thirty read the same otherwise.
    """
    shared = sorted(set(baseline) & set(treatment))
    fixed = sum(1 for key in shared if not baseline[key] and treatment[key])
    broken = sum(1 for key in shared if baseline[key] and not treatment[key])
    discordant = fixed + broken
    return {"test": "mcnemar_exact", "n": len(shared), "treatment_fixed": fixed,
            "treatment_broke": broken, "discordant": discordant,
            "p_value": two_sided_binomial(min(fixed, broken), discordant)}
