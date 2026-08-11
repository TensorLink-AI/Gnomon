"""The four canonical macros end-to-end: real signals, trap datasets,
verified lineage, structured abstention."""

from datetime import datetime, timezone
from pathlib import Path

import json
import pytest

from gnomon.cli import main
from gnomon.contracts import GnomonError
from gnomon.ids import FixedClock
from gnomon.macros import decide, investigate_change, monitor

CLOCK = FixedClock(datetime(2026, 7, 1, tzinfo=timezone.utc))
NOISE = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2, -0.1, 0.4, -0.5]


def _csv(path: Path, values: list[float], header: str = "timestamp,value") -> Path:
    from datetime import date, timedelta
    start = date(2026, 3, 1)
    rows = [f"{(start + timedelta(days=i)).isoformat()},{value}" for i, value in enumerate(values)]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def _shifted_series(pre: float = 100, post: float = 118, n_pre: int = 20, n_post: int = 15):
    return (
        [pre + NOISE[i % 10] for i in range(n_pre)]
        + [post + NOISE[i % 10] for i in range(n_post)]
    )


# -- A. investigate_change ------------------------------------------------

def test_investigate_finds_planted_shift(tmp_path):
    source = _csv(tmp_path / "shift.csv", _shifted_series())
    payload, directory = investigate_change(
        str(source), time_column="timestamp", target_column="value",
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    result = payload["results"][0]
    assert result["onset"] == "2026-03-21T00:00:00"
    assert result["classification"] == "regime_shift"
    assert result["support_assessment"]["status"] == "supported"
    lineage = json.loads((directory / "lineage.json").read_text())
    classes = {claim["claim_class"] for claim in lineage["claims"]}
    assert classes <= {"descriptive", "associational"}  # never causal
    assert any(claim["claim_id"].startswith("claim:change") for claim in lineage["claims"])


def test_investigate_no_change_is_a_conclusion(tmp_path):
    source = _csv(tmp_path / "flat.csv", [100 + NOISE[i % 10] for i in range(30)])
    payload, directory = investigate_change(
        str(source), time_column="timestamp", target_column="value",
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    result = payload["results"][0]
    assert result["onset"] is None
    assert result["support_assessment"]["status"] == "supported"
    lineage = json.loads((directory / "lineage.json").read_text())
    assert any(claim["claim_id"].startswith("claim:no_change") for claim in lineage["claims"])


def test_investigate_short_history_abstains(tmp_path):
    source = _csv(tmp_path / "short.csv", [100.0, 101.0, 99.0, 100.5, 100.0, 101.0])
    payload, _ = investigate_change(
        str(source), time_column="timestamp", target_column="value",
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    assert payload["results"][0]["support_assessment"]["status"] == "inconclusive"


def test_investigate_ranks_concurrent_event(tmp_path):
    from gnomon.context import ContextEvent
    source = _csv(tmp_path / "shift.csv", _shifted_series())
    event = ContextEvent(
        event_id="promo-1", event_type="promotion", entity_scope=("__default__",),
        effective_start="2026-03-20", effective_end="2026-03-27",
        known_at="2026-03-19",
    )
    payload, _ = investigate_change(
        str(source), time_column="timestamp", target_column="value",
        context_events=[event], output=str(tmp_path / "out"), clock=CLOCK,
    )
    explanations = payload["results"][0]["explanations"]
    assert explanations and explanations[0]["kind"] == "concurrent_event"
    assert explanations[0]["event_id"] == "promo-1"
    assert payload["results"][0]["residual_uncertainty"] is not None


def test_investigate_records_suspected_cause_without_influence(tmp_path):
    source = _csv(tmp_path / "shift.csv", _shifted_series())
    bare, _ = investigate_change(
        str(source), time_column="timestamp", target_column="value",
        output=str(tmp_path / "bare"), clock=CLOCK,
    )
    noted, directory = investigate_change(
        str(source), time_column="timestamp", target_column="value",
        suspected_cause="pricing change rolled out March 21",
        output=str(tmp_path / "noted"), clock=CLOCK,
    )
    assert "suspected_cause" not in bare
    record = noted["suspected_cause"]
    assert record["text"] == "pricing change rolled out March 21"
    assert record["influence"] == "none"
    # The hypothesis is on the record but never steers the analysis...
    assert noted["results"] == bare["results"]
    # ...and it is part of the record's identity.
    assert noted["investigation_id"] != bare["investigation_id"]
    artifact = json.loads((directory / "artifact.json").read_text())
    assert artifact["suspected_cause"]["text"] == record["text"]


def test_investigate_suspected_cause_is_in_the_tool_schema():
    from gnomon.registry import MACROS
    schema = MACROS["investigate_change"].input_schema
    assert "suspected_cause" in schema["properties"]
    assert "does not influence" in schema["properties"]["suspected_cause"]["description"]


# -- C. decide ------------------------------------------------------------

def _forecastable(tmp_path) -> Path:
    values = [100 + i * 1.5 + NOISE[i % 10] * 3 for i in range(48)]
    return _csv(tmp_path / "trend.csv", values)


def test_decide_without_utilities_degrades(tmp_path):
    payload, directory = decide(
        str(_forecastable(tmp_path)), time_column="timestamp", target_column="value",
        horizon=6, threshold=170.0,
        actions=[{"name": "scale_up"}, {"name": "wait"}],
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    assert payload["evaluation"]["selected"] is None
    assert payload["support_assessment"]["status"] == "conditionally_supported"
    assert payload["support_assessment"]["reasons"][0]["code"] == "missing_utility_inputs"
    assert payload["scenario_probabilities"] is not None
    lineage = json.loads((directory / "lineage.json").read_text())
    assert not any(claim["claim_class"] == "decision" for claim in lineage["claims"])


def test_decide_with_utilities_chooses(tmp_path):
    payload, directory = decide(
        str(_forecastable(tmp_path)), time_column="timestamp", target_column="value",
        horizon=6, threshold=170.0,
        actions=[{"name": "scale_up"}, {"name": "wait"}],
        utilities={
            "scale_up": {"exceed": 100, "no_exceed": -10},
            "wait": {"exceed": -500, "no_exceed": 5},
        },
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    assert payload["evaluation"]["selected"] in ("scale_up", "wait")
    assert payload["support_assessment"]["status"] == "supported"
    lineage = json.loads((directory / "lineage.json").read_text())
    decision_claims = [claim for claim in lineage["claims"] if claim["claim_class"] == "decision"]
    assert decision_claims and decision_claims[0]["constraints_evaluated"] is True


def test_decide_refuses_when_forecast_is_not_calibrated(tmp_path):
    # Under graduated support the embedded forecast publishes labelled
    # sub-supported rows instead of abstaining — but the decision still
    # refuses: uncalibrated rows carry no exceedance risk to ground it.
    source = _csv(tmp_path / "short.csv", [100.0, 102.0, 101.0, 103.0])
    payload, _ = decide(
        str(source), time_column="timestamp", target_column="value",
        horizon=3, threshold=110.0, actions=[{"name": "act"}],
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    assert payload["support_assessment"]["status"] == "inconclusive"
    assert payload["support_assessment"]["reasons"][0]["code"] in (
        "forecast_not_calibrated", "forecast_abstained")
    assert payload["scenario_probabilities"] is None
    assert payload["evaluation"]["selected"] is None


def test_decide_multiple_series_requires_selection(tmp_path):
    from datetime import date, timedelta
    start = date(2026, 3, 1)
    rows = ["timestamp,series,value"]
    for i in range(30):
        day = (start + timedelta(days=i)).isoformat()
        rows.append(f"{day},a,{100 + i}")
        rows.append(f"{day},b,{200 - i}")
    source = tmp_path / "two.csv"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(GnomonError) as caught:
        decide(
            str(source), time_column="timestamp", target_column="value",
            series_column="series", horizon=3, threshold=150.0,
            actions=[{"name": "act"}], output=str(tmp_path / "out"), clock=CLOCK,
        )
    assert caught.value.code == "MULTIPLE_SERIES_UNSUPPORTED"


# -- D. monitor -----------------------------------------------------------

def test_monitor_with_costs_gives_optimal_rule(tmp_path):
    payload, directory = monitor(
        str(_forecastable(tmp_path)), time_column="timestamp", target_column="value",
        horizon=6, threshold=165.0, alert_cost=1.0, miss_cost=9.0,
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    trigger = payload["triggers"][0]
    assert trigger["armed"] is True
    assert trigger["alert_probability_threshold"] == pytest.approx(0.1)
    assert trigger["support_assessment"]["status"] == "supported"
    lineage = json.loads((directory / "lineage.json").read_text())
    risk_claims = [claim for claim in lineage["claims"] if "sequential_risk" in claim["claim_id"]]
    assert risk_claims and risk_claims[0]["calibration_ref"] is not None


def test_monitor_without_costs_is_conditional(tmp_path):
    payload, _ = monitor(
        str(_forecastable(tmp_path)), time_column="timestamp", target_column="value",
        horizon=6, threshold=165.0,
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    trigger = payload["triggers"][0]
    assert trigger["alert_probability_threshold"] == 0.5
    assert trigger["support_assessment"]["status"] == "conditionally_supported"
    assert trigger["support_assessment"]["reasons"][0]["code"] == "missing_cost_inputs"


def test_monitor_abstains_on_short_history(tmp_path):
    source = _csv(tmp_path / "short.csv", [1.0, 2.0, 1.5])
    payload, _ = monitor(
        str(source), time_column="timestamp", target_column="value",
        horizon=2, threshold=5.0, output=str(tmp_path / "out"), clock=CLOCK,
    )
    trigger = payload["triggers"][0]
    assert trigger["armed"] is False
    assert trigger["support_assessment"]["status"] == "inconclusive"


# -- CLI smoke ------------------------------------------------------------

def test_cli_investigate_decide_monitor(tmp_path, capsys):
    source = _csv(tmp_path / "shift.csv", _shifted_series())
    assert main([
        "investigate", str(source), "--time", "timestamp", "--target", "value",
        "--output", str(tmp_path / "out"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["classification"] == "regime_shift"

    trend = _forecastable(tmp_path)
    assert main([
        "decide", str(trend), "--time", "timestamp", "--target", "value",
        "--horizon", "6", "--threshold", "170",
        "--actions", '[{"name": "scale_up"}, {"name": "wait"}]',
        "--output", str(tmp_path / "out"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["support_assessment"]["status"] == "conditionally_supported"

    assert main([
        "monitor", str(trend), "--time", "timestamp", "--target", "value",
        "--horizon", "6", "--threshold", "165",
        "--alert-cost", "1", "--miss-cost", "9",
        "--output", str(tmp_path / "out"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["triggers"][0]["armed"] is True
