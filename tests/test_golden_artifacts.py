"""Golden-artifact tests: the forecast pipeline must produce byte-identical
artifact.json output for fixed inputs under a pinned clock.

These goldens are the safety net for every structural change to the
pipeline (stage extraction, store migration, planner execution): behaviour
changes show up as a diff here. Refresh deliberately with
``pytest --update-goldens`` and review the diff like code.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aion.ids import FixedClock
from aion.runtime import forecast

REPO = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))

CASES = {
    "daily_requests_h7": dict(
        input_path=str(REPO / "examples" / "daily_requests.csv"),
        time_column="timestamp", target_column="requests", horizon=7,
    ),
    "daily_requests_h7_threshold": dict(
        input_path=str(REPO / "examples" / "daily_requests.csv"),
        time_column="timestamp", target_column="requests", horizon=7,
        threshold=120.0,
    ),
    "two_series_h5": dict(
        input_path=str(REPO / "tests" / "data" / "golden_two_series.csv"),
        time_column="timestamp", target_column="value", series_column="series",
        horizon=5,
    ),
    "short_history_degraded_h3": dict(
        input_path=str(REPO / "tests" / "data" / "golden_short.csv"),
        time_column="timestamp", target_column="value", horizon=3,
    ),
    "short_history_strict_unsupported_h3": dict(
        input_path=str(REPO / "tests" / "data" / "golden_short.csv"),
        time_column="timestamp", target_column="value", horizon=3,
        strict_abstention=True,
    ),
}


def _normalise(text: str) -> str:
    """Strip the only machine-specific detail (the repo location) so the
    goldens are portable while everything else stays byte-exact."""
    return text.replace(str(REPO), "<REPO>")


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_artifact(name, tmp_path, request):
    parameters = dict(CASES[name])
    input_path = parameters.pop("input_path")
    artifact, artifact_dir = forecast(
        input_path, output=str(tmp_path), clock=CLOCK, **parameters,
    )
    produced = _normalise((artifact_dir / "artifact.json").read_text(encoding="utf-8"))
    golden_path = GOLDEN_DIR / f"{name}.json"
    if request.config.getoption("--update-goldens"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(produced, encoding="utf-8")
        return
    assert golden_path.is_file(), (
        f"Missing golden {golden_path}; run pytest --update-goldens to create it."
    )
    assert produced == golden_path.read_text(encoding="utf-8"), (
        f"artifact.json for {name} deviates from its golden; if the change is "
        "intentional, refresh with pytest --update-goldens and review the diff."
    )


def test_forecast_id_is_content_addressed(tmp_path):
    """The same task yields the same ID and the artifact write is idempotent."""
    parameters = dict(CASES["daily_requests_h7"])
    input_path = parameters.pop("input_path")
    first, first_dir = forecast(input_path, output=str(tmp_path), clock=CLOCK, **parameters)
    second, second_dir = forecast(input_path, output=str(tmp_path), clock=CLOCK, **parameters)
    assert first.forecast_id == second.forecast_id
    assert first_dir == second_dir
    data = json.loads((first_dir / "artifact.json").read_text(encoding="utf-8"))
    assert data["forecast_id"] == first.forecast_id


def test_different_task_different_id(tmp_path):
    parameters = dict(CASES["daily_requests_h7"])
    input_path = parameters.pop("input_path")
    base, _ = forecast(input_path, output=str(tmp_path), clock=CLOCK, **parameters)
    longer, _ = forecast(
        input_path, output=str(tmp_path), clock=CLOCK,
        **{**parameters, "horizon": 8},
    )
    assert base.forecast_id != longer.forecast_id
