"""Property tests for the four integrity invariants (unified plan 1F).

Each is a property rather than an example: it quantifies over inputs
instead of pinning one output, so a regression cannot hide in the case
nobody thought to write down.

1. **Candidate identity sensitivity** — any option that changes a
   published number changes the run's identity. An answer that moves
   under a fixed id is an artifact that lies about what produced it.
2. **Quantile ordering** — published quantiles are monotone at every
   step. A crossed interval is not an interval.
3. **Point-in-time replay** — a run at ``as_of`` T is a function of the
   data visible at T. Appending later vintages cannot change it.
4. **Cross-surface equivalence** — Python, CLI, and MCP compile to the
   same numbers. A surface that disagrees is a surface that lies.
"""

import csv
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from gnomon.config import GnomonConfig
from gnomon.ids import FixedClock
from gnomon.runtime import forecast

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]
HORIZON = 7


def _values(periods: int = 160) -> list[float]:
    return [100 + i * 0.4 + NOISE[i % 10] * 6 for i in range(periods)]


def _csv(path: Path, values: list[float] | None = None) -> Path:
    start = date(2026, 1, 1)
    rows = [f"{(start + timedelta(days=i)).isoformat()},{v:.4f}"
            for i, v in enumerate(values if values is not None else _values())]
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n",
                    encoding="utf-8")
    return path


def _run(tmp_path, label, **kwargs):
    return forecast(
        str(_csv(tmp_path / "series.csv")), time_column="timestamp",
        target_column="value", horizon=HORIZON, frequency="D",
        output=str(tmp_path / label), clock=CLOCK, **kwargs,
    )


# -- 1. candidate identity sensitivity ------------------------------------

def _answer_changing_configs():
    """Options that provably move a published number."""
    coverage = GnomonConfig()
    coverage.evaluation.target_coverage = 0.5
    pooled = GnomonConfig()
    pooled.evaluation.pool_residuals = False
    ensemble = GnomonConfig()
    ensemble.ensemble.enabled = True
    ensemble.ensemble.strategy = "median"
    restricted = GnomonConfig()
    restricted.models.statistical_candidates = ["drift"]
    return {"target_coverage": coverage, "pool_residuals": pooled,
            "ensemble_median": ensemble, "restricted_pool": restricted}


@pytest.mark.parametrize("label", sorted(_answer_changing_configs()))
def test_moving_an_answer_moves_the_identity(tmp_path, label):
    config = _answer_changing_configs()[label]
    # The ensemble only changes numbers when it actually publishes, so
    # that arm forces it rather than hoping it wins selection.
    extra = ({"selection_strategy": "ensemble"}
             if label == "ensemble_median" else {})
    base, _ = _run(tmp_path, "base", **extra)
    variant, _ = _run(tmp_path, label, config=config, **extra)

    def rows(artifact):
        return [(row.get("point"), row.get("q10"), row.get("q90"))
                for row in artifact.results[0].forecast]

    if rows(base) == rows(variant):
        pytest.skip(f"{label} did not move a number on this series")
    assert base.forecast_id != variant.forecast_id, (
        f"{label} changed the published numbers under an unchanged "
        "forecast id: the artifact lies about what produced it"
    )


def test_identical_inputs_keep_one_identity(tmp_path):
    """The converse: identity must not churn on re-runs, or content
    addressing means nothing."""
    first, _ = _run(tmp_path, "first")
    second, _ = _run(tmp_path, "second")
    assert first.forecast_id == second.forecast_id


# -- 2. quantile ordering -------------------------------------------------

@pytest.mark.parametrize("coverage", [0.5, 0.8, 0.95])
def test_published_quantiles_never_cross(tmp_path, coverage):
    config = GnomonConfig()
    config.evaluation.target_coverage = coverage
    artifact, path = _run(tmp_path, f"q{coverage}", config=config)
    result = artifact.results[0]
    if not result.forecast:
        pytest.skip("this run abstained")

    for step, row in enumerate(result.forecast, 1):
        low, high = row.get("q10"), row.get("q90")
        if low is None or high is None:
            continue
        assert low <= high, f"step {step}: q10 {low} above q90 {high}"

    rows = list(csv.DictReader(
        open(Path(path) / "forecast.csv", encoding="utf-8")))
    for step, row in enumerate(rows, 1):
        if "q10" not in row or "q90" not in row:
            continue
        values = [float(row[key]) for key in ("q10", "q50", "q90")
                  if row.get(key) not in (None, "")]
        assert values == sorted(values), (
            f"step {step}: artifact quantiles are not monotone: {values}"
        )


# -- 3. point-in-time replay ----------------------------------------------

def test_replay_is_a_function_of_what_was_visible(tmp_path):
    """A run at as_of T must not move when later vintages are appended:
    that is the leakage boundary stated as a property."""
    # Naive, to match the naive timestamps in the CSV: comparing a
    # tz-aware as_of against a naive dataset is refused outright.
    cutoff = datetime(2026, 4, 1)
    # The series starts 2026-01-01, so index 90 IS the cutoff date: the
    # short file must carry every observation visible at as_of, and the
    # long file may differ only in observations dated after it. A short
    # file that stops earlier tests nothing — the two runs would see
    # different visible data and differ for that reason alone.
    visible = 91
    short = _values(visible)
    long = _values(160)          # identical prefix, plus post-cutoff rows

    early, _ = forecast(
        str(_csv(tmp_path / "short.csv", short)), time_column="timestamp",
        target_column="value", horizon=HORIZON, frequency="D",
        output=str(tmp_path / "early"), clock=CLOCK, as_of=cutoff,
    )
    later, _ = forecast(
        str(_csv(tmp_path / "long.csv", long)), time_column="timestamp",
        target_column="value", horizon=HORIZON, frequency="D",
        output=str(tmp_path / "later"), clock=CLOCK, as_of=cutoff,
    )

    def points(artifact):
        return [row.get("point") for row in artifact.results[0].forecast]

    assert points(early) == points(later), (
        "observations dated after as_of changed the forecast"
    )
    assert early.results[0].selected_model == later.results[0].selected_model


# -- 4. cross-surface equivalence -----------------------------------------

def test_python_cli_and_mcp_publish_the_same_numbers(tmp_path):
    source = _csv(tmp_path / "series.csv")

    artifact, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=HORIZON, frequency="D", output=str(tmp_path / "py"),
        clock=CLOCK,
    )
    python_points = [row.get("point") for row in artifact.results[0].forecast]

    repo = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    completed = subprocess.run(
        [sys.executable, "-m", "gnomon", "forecast", str(source),
         "--time", "timestamp", "--target", "value", "--horizon", str(HORIZON),
         "--frequency", "D", "--output", str(tmp_path / "cli")],
        capture_output=True, text=True, cwd=repo, env=env,
    )
    assert completed.returncode == 0, completed.stderr
    # --output names the parent; each run writes its own content-addressed
    # directory beneath it.
    cli_csvs = sorted((tmp_path / "cli").rglob("forecast.csv"))
    assert len(cli_csvs) == 1, f"expected one CLI artifact, got {cli_csvs}"
    cli_rows = list(csv.DictReader(open(cli_csvs[0], encoding="utf-8")))
    cli_points = [float(row["point"]) for row in cli_rows]

    from gnomon.toolspec import _run_forecast
    payload = _run_forecast({
        "input": str(source), "time_column": "timestamp",
        "target_column": "value", "horizon": HORIZON, "frequency": "D",
        "output_dir": str(tmp_path / "mcp"), "format": "full",
    })
    mcp_rows = list(csv.DictReader(
        open(Path(payload["artifact_path"]) / "forecast.csv",
             encoding="utf-8")))
    mcp_points = [float(row["point"]) for row in mcp_rows]

    assert python_points == pytest.approx(cli_points), (
        "the CLI published different numbers from the Python API"
    )
    assert python_points == pytest.approx(mcp_points), (
        "the MCP surface published different numbers from the Python API"
    )
    assert (artifact.results[0].selected_model
            == payload["results"][0]["selected_model"]), (
        "the surfaces disagree about which model was selected"
    )
    assert artifact.forecast_id == payload["forecast_id"], (
        "the same inputs produced different identities on two surfaces"
    )
