"""Seeded Monte Carlo harness: published intervals cover ~nominal.

Thirty synthetic daily series with known generating processes (trend +
weekly season + Gaussian noise, parameters drawn from a seeded RNG) are
forecast through the full pipeline; the published q10–q90 band is then
scored against actuals drawn from the same process. Empirical coverage
must land within the declared bounds around the nominal 80%.

Declared design:
- sample: 30 series × 7-step horizon = 210 scored points, correlated
  within series, so the effective sample is nearer the series count;
- bounds, asymmetric because conformal's guarantee is one-sided:
  UNDER-coverage below 0.68 fails hard — that is the class of bug
  where an interval is not what it says it is. The upper bound 0.97
  only guards against degenerating into trivially-wide bands that
  cover everything and inform nothing.
- measured at introduction (seed fixed): 0.933 — systematically
  conservative, the direction conformal promises, inflated by
  small per-fold calibration sets. Recorded as a width-efficiency
  finding for the unified plan's Phase 3 measurement (interval width
  is a first-class metric there); the harness pins today's behaviour
  so movement in either direction is a visible, reviewed change.
- determinism: everything derives from the fixed seed, so a failure is
  reproducible, never flaky.
"""

import csv
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from gnomon.ids import FixedClock
from gnomon.runtime import forecast

SEED = 20260812
N_SERIES = 30
HISTORY = 140
HORIZON = 7
NOMINAL = 0.80
LOWER_BOUND = 0.68   # below: calibration is broken
UPPER_BOUND = 0.97   # above: the band is trivially wide

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))


def _generate(rng: random.Random) -> tuple[list[float], list[float]]:
    """One synthetic series: (history, future_actuals) from one process."""
    level = rng.uniform(50.0, 150.0)
    trend = rng.uniform(-0.3, 0.3)
    amplitude = rng.uniform(0.0, 15.0)
    phase = rng.uniform(0.0, 2 * math.pi)
    sigma = rng.uniform(1.0, 6.0)
    total = HISTORY + HORIZON
    values = [
        level + trend * i + amplitude * math.sin(2 * math.pi * i / 7 + phase)
        + rng.gauss(0.0, sigma)
        for i in range(total)
    ]
    return values[:HISTORY], values[HISTORY:]


def _write_csv(path: Path, values: list[float]) -> None:
    start = date(2026, 1, 1)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "value"])
        for index, value in enumerate(values):
            writer.writerow([(start + timedelta(days=index)).isoformat(),
                             f"{value:.6f}"])


def test_interval_coverage_within_declared_tolerance(tmp_path):
    rng = random.Random(SEED)
    covered = 0
    scored = 0
    calibrated_runs = 0

    for series_index in range(N_SERIES):
        history, future = _generate(rng)
        csv_path = tmp_path / f"series_{series_index}.csv"
        _write_csv(csv_path, history)
        artifact, out_path = forecast(
            str(csv_path), time_column="timestamp", target_column="value",
            horizon=HORIZON, frequency="D",
            output=str(tmp_path / f"out_{series_index}"), clock=CLOCK,
        )
        result = artifact.results[0]
        # Only evaluated (calibrated) intervals make a coverage claim;
        # a best_effort fallback publishes no measured band to score.
        if not result.forecast or result.support in ("best_effort",):
            continue
        rows = list(csv.DictReader(
            open(Path(out_path) / "forecast.csv", encoding="utf-8")))
        if not rows or "q10" not in rows[0] or "q90" not in rows[0]:
            continue
        calibrated_runs += 1
        for step, row in enumerate(rows[:HORIZON]):
            lower, upper = float(row["q10"]), float(row["q90"])
            actual = future[step]
            scored += 1
            if lower <= actual <= upper:
                covered += 1

    # The harness must actually measure something: a quietly-empty run
    # would pass any band.
    assert calibrated_runs >= N_SERIES * 2 // 3, (
        f"only {calibrated_runs}/{N_SERIES} runs produced calibrated "
        "intervals; the harness is not measuring what it claims"
    )
    coverage = covered / scored
    assert coverage >= LOWER_BOUND, (
        f"empirical q10-q90 coverage {coverage:.3f} over {scored} points "
        f"({calibrated_runs} series) is under-covering a nominal "
        f"{NOMINAL:.0%} band: the interval is not what it says it is"
    )
    assert coverage <= UPPER_BOUND, (
        f"empirical q10-q90 coverage {coverage:.3f} over {scored} points "
        f"({calibrated_runs} series): the band has become trivially wide"
    )
