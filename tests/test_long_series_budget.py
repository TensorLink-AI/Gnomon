"""Long series stay inside interactive budgets, disclosed.

The 2026-08 MCP evaluation ran Gnomon against a 23,594-row daily series
(64 years of Treasury yields) and had to kill `investigate` and `monitor`
at its 300 s client timeout; `forecast` ran 235 s. Three costs compounded:
an O(n²) recompute-from-scratch changepoint scan, a per-fold O(n) rebuild
of the bitemporal view for single-vintage data, and full-history model
refits per fold. Each fix here is either behaviour-preserving (prefix
sums, prefix slices) or disclosed (the fit window). A host killing the
process produces no artifact at all — worse than any abstention — so
staying inside interactive budgets is part of the honesty contract, not
just a nicety.
"""

from __future__ import annotations

import random
import time
from datetime import date, timedelta
from pathlib import Path

import pytest


def _long_daily_csv(path: Path, rows: int) -> Path:
    generator = random.Random(42)
    start = date(1962, 1, 2)
    level = 4.0
    lines = ["timestamp,value"]
    for index in range(rows):
        level = max(0.5, min(16.0, level + generator.gauss(0, 0.02)))
        if index in (rows // 4, rows // 2):
            level += 3.0
        lines.append(f"{(start + timedelta(days=index)).isoformat()},{level:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_changepoint_scan_is_linear_per_segment() -> None:
    """20k observations must be interactively cheap. Before the prefix-sum
    rewrite this scan alone was ~5.6e8 inner operations (killed at 300 s
    in the field); afterwards a generous CI bound is still two orders of
    magnitude above the observed cost."""
    from gnomon.operators import changepoint_detection

    generator = random.Random(7)
    values = [10.0 + generator.gauss(0, 0.5) + (8.0 if index > 12_000 else 0.0)
              for index in range(20_000)]
    timestamps = list(range(len(values)))
    started = time.monotonic()
    result = changepoint_detection(timestamps, values)
    elapsed = time.monotonic() - started
    assert elapsed < 30.0, f"changepoint scan took {elapsed:.1f}s on 20k rows"
    assert any(11_500 < point["index"] < 12_500
               for point in result["changepoints"]), (
        "the injected level shift went undetected")


def test_prefix_sum_split_matches_the_naive_scan() -> None:
    """The rewrite is an optimisation, not a behaviour change: on series
    where cancellation could bite (large mean, small spread) the chosen
    split index must match the direct SSE computation."""
    from gnomon.operators import MIN_SEGMENT, _best_split, _sse

    generator = random.Random(3)
    for mean in (0.0, 1e6):
        values = [mean + generator.gauss(0, 1.0) + (5.0 if index >= 60 else 0.0)
                  for index in range(120)]
        naive_best, naive_gain = None, 0.0
        total = _sse(values)
        for split in range(MIN_SEGMENT, len(values) - MIN_SEGMENT + 1):
            gain = total - _sse(values[:split]) - _sse(values[split:])
            if gain > naive_gain:
                naive_best, naive_gain = split, gain
        fast = _best_split(values, 0)
        assert fast is not None and naive_best is not None
        assert fast[0] == naive_best
        assert fast[1] == pytest.approx(naive_gain, rel=1e-9)


def test_long_series_fit_window_is_disclosed(tmp_path: Path) -> None:
    """Past MAX_FIT_HISTORY the per-fold fits see a trailing window; the
    result must complete AND say so — the same disclosure contract as the
    anomaly grading window."""
    from gnomon.evaluation import MAX_FIT_HISTORY
    from gnomon.toolspec import runner_for

    source = _long_daily_csv(tmp_path / "long.csv", 3_000)
    assert 3_000 > MAX_FIT_HISTORY
    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 14,
        "output_dir": str(tmp_path / "out"),
    })
    assert payload["status"] == "complete"
    warnings = payload["results"][0]["warnings"]
    assert any("fit_window" in warning for warning in warnings), warnings


def test_short_series_never_sees_the_fit_window(tmp_path: Path) -> None:
    source = _long_daily_csv(tmp_path / "short.csv", 200)
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": str(source), "horizon": 7,
        "output_dir": str(tmp_path / "out"),
    })
    assert payload["status"] == "complete"
    warnings = payload["results"][0]["warnings"]
    assert not any("fit_window" in warning for warning in warnings), warnings


def test_single_vintage_fast_path_matches_the_rebuild(tmp_path: Path) -> None:
    """The prefix fast path is gated on exact equivalence; when it engages,
    the fold histories must be byte-identical to the honest per-cutoff
    rebuild it replaces."""
    from gnomon.pipeline import load_stage

    source = _long_daily_csv(tmp_path / "series.csv", 300)
    loaded = load_stage(str(source), time_column="timestamp",
                        target_column="value", series_column=None,
                        frequency=None)
    snapshot, variable = loaded.snapshot, loaded.variable
    for name, items in loaded.groups.items():
        values = [item.value for item in items]
        timestamps = [item.timestamp for item in items]
        for origin in (50, 150, 295):
            cutoff = timestamps[origin - 1]
            rebuilt = [
                item.value
                for item in snapshot.series(name, variable, cutoff=cutoff)
                if item.valid_time <= cutoff
            ]
            assert rebuilt == values[:origin]
