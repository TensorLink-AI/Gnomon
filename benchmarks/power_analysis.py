"""Pre-run power calculation for paired win/loss evaluation.

Uses an exact binomial rejection region and exact power under a declared
alternative. Ties reduce the effective sample and must be priced before a
run rather than discarded after seeing results.
"""

from __future__ import annotations

import argparse
import json
import math


def _tail(n: int, start: int, probability: float) -> float:
    return sum(math.comb(n, k) * probability ** k
               * (1 - probability) ** (n - k)
               for k in range(start, n + 1))


def rejection_cutoff(n: int, alpha: float) -> int:
    """Smallest treatment-win count significant in a two-sided sign test."""
    wins = (n // 2) + 1
    tail = _tail(n, wins, .5)
    while wins <= n:
        if 2 * tail <= alpha:
            return wins
        tail -= math.comb(n, wins) * .5 ** n
        wins += 1
    return n + 1


def exact_power(n: int, win_probability: float, alpha: float) -> float:
    cutoff = rejection_cutoff(n, alpha)
    return _tail(n, cutoff, win_probability) if cutoff <= n else 0.0


def required_pairs(win_probability: float, power: float, alpha: float,
                   maximum: int = 10000) -> int:
    low, high = 2, 4
    while high <= maximum and exact_power(high, win_probability, alpha) < power:
        low, high = high, high * 2
    if high > maximum:
        high = maximum
        if exact_power(high, win_probability, alpha) < power:
            raise ValueError("required sample exceeds maximum")
    # Exact-test power has small saw-tooth changes as the rejection cutoff
    # moves. Binary search finds the neighbourhood, then a bounded scan finds
    # the first actual crossing rather than assuming strict monotonicity.
    while high - low > 32:
        middle = (low + high) // 2
        if exact_power(middle, win_probability, alpha) >= power:
            high = middle
        else:
            low = middle
    for n in range(max(2, low - 32), high + 1):
        if exact_power(n, win_probability, alpha) >= power:
            return n
    raise ValueError("power search failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-probability", type=float, default=.55)
    parser.add_argument("--power", type=float, default=.8)
    parser.add_argument("--alpha", type=float, default=.025)
    parser.add_argument("--expected-tie-rate", type=float, default=.1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    effective = required_pairs(args.win_probability, args.power, args.alpha)
    total = math.ceil(effective / (1 - args.expected_tie_rate))
    result = {
        "test": "two-sided exact paired sign test",
        "alternative_win_probability": args.win_probability,
        "power": args.power,
        "familywise_alpha_per_primary_comparison": args.alpha,
        "expected_tie_rate": args.expected_tie_rate,
        "required_non_tied_pairs": effective,
        "required_total_pairs": total,
        "power_at_80_pairs": exact_power(
            round(80 * (1 - args.expected_tie_rate)),
            args.win_probability, args.alpha),
        "power_at_480_pairs": exact_power(
            round(480 * (1 - args.expected_tie_rate)),
            args.win_probability, args.alpha),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
