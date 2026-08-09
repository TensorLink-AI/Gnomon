"""Every caller-settable parameter is classified, and every epistemic one
is priced.

The boundary rule (docs/agent-dogfood-review-2026-08.md, follow-up): a
parameter either states what the caller *wants* (intent), describes the
caller's *data*, or changes what counts as *evidence* (epistemic). The
first two are free; an epistemic parameter moved off its default must
leave a trace in the artifact. These tests keep the classification
complete — a parameter that reaches a front door unclassified fails here,
which makes the classification the design review — and check the priced
behaviours the table promises.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gnomon.contracts import (
    DEFAULT_MINIMUM_BASELINE_IMPROVEMENT,
    EPISTEMIC_TRACES,
    PARAMETER_AUTHORITY,
)
from gnomon.ids import FixedClock

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _csv(tmp_path: Path, periods: int = 200) -> Path:
    from datetime import date, timedelta

    start = date(2026, 1, 1)
    rows = [
        f"{(start + timedelta(days=index)).isoformat()},"
        f"{100 + index * 0.4 + NOISE[index % 10] * 6:.4f}"
        for index in range(periods)
    ]
    path = tmp_path / "series.csv"
    path.write_text("timestamp,value\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _cli_parameters() -> set[str]:
    from gnomon.cli import build_parser

    names: set[str] = set()

    def walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    walk(sub)
            elif action.dest not in ("help",):
                names.add(action.dest)

    walk(build_parser())
    return names


def _mcp_parameters() -> set[str]:
    from gnomon.toolspec import TOOLS

    names: set[str] = set()
    for tool in TOOLS:
        names |= set((tool.get("inputSchema", {}).get("properties") or {}).keys())
    return names


def test_every_front_door_parameter_is_classified():
    unclassified = sorted(
        (_cli_parameters() | _mcp_parameters()) - set(PARAMETER_AUTHORITY)
    )
    assert not unclassified, (
        f"Parameters reached a front door without an authority "
        f"classification: {unclassified}. Add each to "
        f"contracts.PARAMETER_AUTHORITY (intent / data / epistemic); an "
        f"epistemic one also needs its trace in EPISTEMIC_TRACES."
    )


def test_classifications_are_well_formed():
    assert set(PARAMETER_AUTHORITY.values()) <= {"intent", "data", "epistemic"}


def test_every_epistemic_parameter_names_its_trace():
    epistemic = {
        name for name, kind in PARAMETER_AUTHORITY.items() if kind == "epistemic"
    }
    untraced = sorted(epistemic - set(EPISTEMIC_TRACES))
    assert not untraced, (
        f"Epistemic parameters without a priced trace: {untraced}. "
        f"Moving one of these off its default changes what the numbers "
        f"mean; say where the artifact discloses it."
    )
    assert all(EPISTEMIC_TRACES.values()), "a trace entry must not be empty"


# -- The priced behaviours the table promises ------------------------------

def test_lowered_baseline_margin_is_disclosed_and_capped(tmp_path):
    from gnomon.runtime import forecast

    artifact, _ = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
        minimum_baseline_improvement=0.0,
    )
    assessment = artifact.results[0].support_assessment
    codes = [reason["code"] for reason in assessment["reasons"]]
    assert "nonstandard_evaluation" in codes
    assert assessment["status"] != "supported"


def test_raised_baseline_margin_is_disclosed_but_not_capped(tmp_path):
    from gnomon.runtime import forecast

    artifact, _ = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
        minimum_baseline_improvement=0.5,
    )
    assessment = artifact.results[0].support_assessment
    codes = [reason["code"] for reason in assessment["reasons"]]
    assert "nonstandard_evaluation" in codes
    # Stricter than documented is worth knowing, not worth capping: the
    # verdict stays whatever the evidence earned.


def test_default_margin_leaves_no_deviation_reason(tmp_path):
    from gnomon.runtime import forecast

    artifact, _ = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
    )
    assessment = artifact.results[0].support_assessment
    codes = [reason["code"] for reason in assessment["reasons"]]
    assert "nonstandard_evaluation" not in codes
    assert "candidate_pool_restricted" not in codes


def test_candidate_restriction_is_disclosed_not_capped(tmp_path):
    from gnomon.runtime import forecast

    artifact, _ = forecast(
        str(_csv(tmp_path)), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "out"), clock=CLOCK,
        candidates=["drift", "theta"],
    )
    assessment = artifact.results[0].support_assessment
    restriction = next(
        reason for reason in assessment["reasons"]
        if reason["code"] == "candidate_pool_restricted"
    )
    assert "drift" in restriction["message"]
    assert "theta" in restriction["message"]


def test_restricted_and_open_pools_get_distinct_forecast_ids(tmp_path):
    from gnomon.runtime import forecast

    source = _csv(tmp_path)
    open_run, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "open"), clock=CLOCK,
    )
    restricted_run, _ = forecast(
        str(source), time_column="timestamp", target_column="value",
        horizon=7, output=str(tmp_path / "restricted"), clock=CLOCK,
        candidates=["drift"],
    )
    assert open_run.forecast_id != restricted_run.forecast_id, (
        "a restricted contest must not share a content id with an open one:"
        " first-write-wins would then serve whichever artifact landed first"
    )


def test_default_run_default_margin_matches_constant():
    import inspect

    from gnomon.runtime import forecast

    signature = inspect.signature(forecast)
    assert (signature.parameters["minimum_baseline_improvement"].default
            == DEFAULT_MINIMUM_BASELINE_IMPROVEMENT)
