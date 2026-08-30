from __future__ import annotations

import math

from benchmarks.hierarchybench.run import _shares, _split, generate_cases


def test_partitions_are_positive_and_exactly_additive() -> None:
    values = [100.0 + index for index in range(40)]
    for family in ("stable", "periodic"):
        leaves = _split(values, 37, family)
        assert all(part > 0 for leaf in leaves for part in leaf)
        assert all(math.fsum(leaf[index] for leaf in leaves) == value
                   for index, value in enumerate(values))


def test_periodic_shares_stay_positive_and_sum_to_one() -> None:
    for index in range(1000):
        shares = _shares(index, "periodic")
        assert all(value > 0 for value in shares)
        # Trigonometric rounding differs by one ulp across supported Python
        # builds. `_split` assigns the final value residual explicitly and its
        # separate test enforces exact hierarchy arithmetic.
        assert math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-15)


def test_frozen_cases_have_nonoverlapping_futures_and_exact_truth() -> None:
    cases, provenance = generate_cases()
    assert len(cases) == 32
    assert provenance["future_windows_non_overlapping"] is True
    for case in cases:
        assert all(math.isclose(
                       math.fsum(leaf[step] for leaf in case.leaf_future), value,
                       rel_tol=0.0, abs_tol=1e-9)
                   for step, value in enumerate(case.root_future))
