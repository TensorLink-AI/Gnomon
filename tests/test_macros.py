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
    assert payload["support_assessment"]["status"] == "inconclusive"
    assert payload["support_assessment"]["reasons"][0]["code"] == \
        "horizon_event_probability_insufficient"
    assert payload["scenario_probabilities"] is None
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
    assert payload["evaluation"]["selected"] is None
    assert payload["support_assessment"]["status"] == "inconclusive"
    assert payload["exceedance"]["horizon_event"]["support"] == "insufficient"
    lineage = json.loads((directory / "lineage.json").read_text())
    decision_claims = [claim for claim in lineage["claims"] if claim["claim_class"] == "decision"]
    assert not decision_claims


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


def test_monitor_single_shot_policy_is_typed_and_may_withhold(tmp_path):
    payload, _ = monitor(
        str(_forecastable(tmp_path)), time_column="timestamp",
        target_column="value", horizon=6, threshold=165.0,
        action_cost=2.0, miss_cost=10.0,
        output=str(tmp_path / "out"), clock=CLOCK,
    )
    trigger = payload["triggers"][0]
    decision = trigger["governed_decision"]
    assert trigger["horizon_event"]["dependence_preserved"] is True
    assert decision["cost_model"] == "single_shot_mitigation_v1"
    assert decision["break_even_probability"] == 0.2
    assert decision["primary_risk_unchanged"] is True
    # This short fixture has too few independent origins: a probability is
    # useful evidence, but it must not become a governed action.
    assert decision["recommended_action"] is None
    assert decision["decision_support"] == "insufficient"


def test_monitor_refuses_ambiguous_alert_and_action_cost_models(tmp_path):
    with pytest.raises(GnomonError) as caught:
        monitor(
            str(_forecastable(tmp_path)), time_column="timestamp",
            target_column="value", horizon=6, threshold=165.0,
            alert_cost=1.0, action_cost=2.0, miss_cost=10.0,
            output=str(tmp_path / "out"), clock=CLOCK,
        )
    assert caught.value.code == "INVALID_COSTS"


@pytest.mark.parametrize("verb", ["decide", "monitor"])
def test_probability_macros_refuse_out_of_band_forecast_coverage(
    tmp_path, monkeypatch, verb,
):
    import gnomon.runtime as runtime_module

    original = runtime_module.forecast

    def miscalibrated(*args, **kwargs):
        artifact, path = original(*args, **kwargs)
        artifact.results[0].interval_coverage = 0.25
        return artifact, path

    monkeypatch.setattr(runtime_module, "forecast", miscalibrated)
    common = dict(
        input_path=str(_forecastable(tmp_path)), time_column="timestamp",
        target_column="value", horizon=6, threshold=165.0,
        output=str(tmp_path / verb), clock=CLOCK,
    )
    if verb == "decide":
        payload, _ = decide(**common, actions=[{"name": "wait"}])
        assert payload["evaluation"]["selected"] is None
        support = payload["support_assessment"]
        assert payload["scenario_probabilities"] is None
    else:
        payload, _ = monitor(**common, alert_cost=1.0, miss_cost=9.0)
        assert payload["triggers"][0]["armed"] is False
        support = payload["triggers"][0]["support_assessment"]
    assert support["reasons"][0]["code"] == "interval_coverage_out_of_band"


def test_decide_and_monitor_share_immutable_typed_answer_contract(tmp_path):
    source = str(_forecastable(tmp_path))
    question = [{"id": "trend", "verb": "predict", "target": "value",
                 "property": "trend", "horizon": 6}]
    decided, _ = decide(
        source, time_column="timestamp", target_column="value", horizon=6,
        threshold=170.0, actions=[{"name": "wait"}], questions=question,
        output=str(tmp_path / "decide"), clock=CLOCK)
    monitored, _ = monitor(
        source, time_column="timestamp", target_column="value", horizon=6,
        threshold=170.0, questions=question,
        output=str(tmp_path / "monitor"), clock=CLOCK)
    for payload in (decided, monitored):
        assert payload["answers"][0]["artifact_id"] == payload["forecast_id"]
        receipt = json.loads(Path(payload["answer_receipt"]).read_text())
        assert receipt["primary_forecast_unchanged"] is True
        assert receipt["answers"] == payload["answers"]


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
    assert payload["support_assessment"]["status"] == "inconclusive"

    assert main([
        "monitor", str(trend), "--time", "timestamp", "--target", "value",
        "--horizon", "6", "--threshold", "165",
        "--alert-cost", "1", "--miss-cost", "9",
        "--output", str(tmp_path / "out"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["triggers"][0]["armed"] is True


def test_cli_monitor_auto_scores_in_registry_not_temporal_store(
        tmp_path, capsys, monkeypatch):
    """The temporal-store override must never be reused as the registry DB."""
    source = _forecastable(tmp_path)
    temporal_store = tmp_path / "temporal-store"
    temporal_store.mkdir()
    registry = tmp_path / "tracking.sqlite"
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(registry))

    assert main([
        "monitor", "run", str(source), "--time", "timestamp",
        "--target", "value", "--horizon", "6", "--threshold", "165",
        "--alert-cost", "1", "--miss-cost", "9", "--project", "ops",
        "--store-path", str(temporal_store),
        "--output", str(tmp_path / "out"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["auto_scored_forecasts"] == []
    assert registry.is_file()
    assert temporal_store.is_dir()


def test_cli_monitor_run_scores_due_forecast_before_opening_next(
        tmp_path, capsys, monkeypatch):
    """The cron-friendly loop closes yesterday's receipt from today's data."""
    source = _forecastable(tmp_path)
    registry = tmp_path / "tracking.sqlite"
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(registry))
    command = [
        "monitor", "run", str(source), "--time", "timestamp",
        "--target", "value", "--horizon", "6", "--threshold", "165",
        "--alert-cost", "1", "--miss-cost", "9", "--project", "ops",
        "--output", str(tmp_path / "out"),
    ]

    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["auto_scored_forecasts"] == []

    # The next scheduled read contains the six observations that were future
    # at the first cutoff. Auto-scoring must resolve that existing receipt
    # before the command registers a new forecast at the later cutoff.
    values = [100 + i * 1.5 + NOISE[i % 10] * 3 for i in range(54)]
    _csv(source, values)
    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)

    scored = second["auto_scored_forecasts"]
    assert len(scored) == 1
    assert scored[0]["forecast_id"] == first["forecast_id"]
    assert scored[0]["mase"] is not None
    assert scored[0]["scored_at"]
