"""Golden-artifact tests: the forecast pipeline must produce byte-identical
artifact.json output for fixed inputs under a pinned clock.

These goldens are the safety net for every structural change to the
pipeline (stage extraction, store migration, planner execution): behaviour
changes show up as a diff here. Refresh deliberately with
``pytest --update-goldens`` and review the diff like code.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gnomon.ids import FixedClock
from gnomon.runtime import forecast

# CPython 3.12 changed builtin sum() to Neumaier compensated summation
# (gh-100425), which shifts float results by an ulp relative to 3.11.
# Deterministic replay is guaranteed per interpreter: goldens are captured
# on 3.12+ and byte-checked there; on older interpreters the values are
# checked to 1e-9 relative tolerance instead.
BYTE_EXACT = sys.version_info >= (3, 12)

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
    golden = golden_path.read_text(encoding="utf-8")
    if BYTE_EXACT:
        assert produced == golden, (
            f"artifact.json for {name} deviates from its golden; if the change is "
            "intentional, refresh with pytest --update-goldens and review the diff."
        )
    else:
        _assert_semantically_equal(json.loads(golden), json.loads(produced), name)


def _assert_semantically_equal(golden, produced, context: str) -> None:
    """Structural equality with 1e-9 relative tolerance on floats — the
    cross-interpreter guarantee, where byte equality is not promised."""
    assert type(golden) is type(produced) or (
        isinstance(golden, (int, float)) and isinstance(produced, (int, float))
    ), context
    if isinstance(golden, dict):
        assert golden.keys() == produced.keys(), context
        for key in golden:
            _assert_semantically_equal(golden[key], produced[key], f"{context}.{key}")
    elif isinstance(golden, list):
        assert len(golden) == len(produced), context
        for index, (a, b) in enumerate(zip(golden, produced)):
            _assert_semantically_equal(a, b, f"{context}[{index}]")
    elif isinstance(golden, float) and not isinstance(golden, bool):
        assert produced == pytest.approx(golden, rel=1e-9, abs=1e-12), context
    else:
        assert golden == produced, context


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
