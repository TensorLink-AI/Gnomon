"""Graduated support: always answer, honestly graded.

The default publication floor is best_effort — every request answers at
the highest tier the evidence achieves, tier-labelled — while nothing
about how any tier is earned changes. `minimum_support` restores the
refusal one parameter away, and a series with no usable history still
abstains.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

DAILY = str(Path(__file__).resolve().parent.parent / "examples" / "daily_requests.csv")


def _short_csv(tmp_path: Path, rows: int = 12) -> str:
    start = date(2026, 3, 1)
    lines = ["timestamp,requests"] + [
        f"{start + timedelta(days=day)},{100 + 3 * day + (day % 3)}"
        for day in range(rows)
    ]
    path = tmp_path / "short.csv"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_default_answers_on_fold_starved_data(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["forecast"], "default must answer, not abstain"
    assert result["support"] == "best_effort"
    assert any("NO RELIABLE FORECAST" in warning
               for warning in result["warnings"])
    # The abstention's reasons are preserved, typed, alongside the rows.
    codes = {reason["code"]
             for reason in result["support_assessment"]["reasons"]}
    assert "no_reliable_forecast" in codes


def test_minimum_support_supported_restores_the_refusal(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14,
        "minimum_support": "supported",
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["forecast"] == []
    assert result["support"] == "unsupported"
    assessment = result["support_assessment"]
    assert assessment["status"] == "inconclusive"
    codes = {action["code"] for action in assessment["recovery_actions"]}
    assert "provide_more_history" in codes


def test_floor_refuses_a_conditionally_supported_result(tmp_path) -> None:
    # daily_requests earns at most conditionally_supported at this horizon
    # (degraded evaluation); floor `supported` must refuse it with the
    # typed below-floor reason, publishing nothing.
    from gnomon.toolspec import runner_for

    payload = runner_for("gnomon_forecast")({
        "input": DAILY, "horizon": 7, "minimum_support": "supported",
        "output_dir": str(tmp_path / "out"),
    })
    result = payload["results"][0]
    assert result["forecast"] == []
    codes = {reason["code"]
             for reason in result["support_assessment"]["reasons"]}
    assert "below_minimum_support" in codes
    recovery = {action["code"] for action in
                result["support_assessment"]["recovery_actions"]}
    assert "lower_minimum_support" in recovery


def test_nothing_computable_still_abstains(tmp_path) -> None:
    # Two observations cannot even seed the naive fallback's dispersion in
    # a meaningful way — but the guard here is the truly-empty case: zero
    # usable rows after load fails loudly, and one observation publishes a
    # labelled flat line. The invariant under test: an unparseable series
    # never fabricates rows. A file whose target column never parses is a
    # typed error, not a best_effort forecast.
    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,requests\n2026-01-01,not_a_number\n")
    import pytest
    with pytest.raises(GnomonError):
        runner_for("gnomon_forecast")({
            "input": str(bad), "horizon": 3,
            "output_dir": str(tmp_path / "out"),
        })


def test_legacy_best_effort_parameter_still_works(tmp_path) -> None:
    from gnomon.toolspec import runner_for

    legacy = runner_for("gnomon_forecast")({
        "input": _short_csv(tmp_path), "horizon": 14, "best_effort": True,
        "output_dir": str(tmp_path / "legacy"),
    })
    result = legacy["results"][0]
    assert result["support"] == "best_effort"
    assert result["forecast"]


def test_invalid_minimum_support_is_typed(tmp_path) -> None:
    import pytest

    from gnomon.contracts import GnomonError
    from gnomon.toolspec import runner_for

    with pytest.raises(GnomonError) as excinfo:
        runner_for("gnomon_forecast")({
            "input": _short_csv(tmp_path), "horizon": 3,
            "minimum_support": "everything",
            "output_dir": str(tmp_path / "out"),
        })
    assert excinfo.value.code == "INVALID_ARGUMENTS"


def test_capabilities_report_the_default_floor() -> None:
    from gnomon.runtime import capabilities

    graduated = capabilities()["forecast_surface"]["graduated_support"]
    assert graduated["default_minimum_support"] == "best_effort"
    assert graduated["tiers"] == [
        "best_effort", "conditionally_supported", "supported",
    ]
