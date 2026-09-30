"""Exercise the trading showcase as shipped, including temporal ledger behavior."""

from datetime import datetime
import json
from math import expm1, log1p
import os
from pathlib import Path
import re
import ast
from datetime import timedelta, timezone
import runpy
import sqlite3
import subprocess
import sys

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.ids import FixedClock


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/trade-with-gnomon"
SCRIPT = SKILL / "scripts/trade_lifecycle.py"
DECISIONS = SKILL / "scripts/trade_decisions.py"


@pytest.fixture(scope="module")
def demo():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture(scope="module")
def decisions():
    return runpy.run_path(str(DECISIONS))


def test_cli_lifecycle_ranking_costs_and_recording_cutoffs(tmp_path):
    output = tmp_path / "showcase"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(output)], check=True, capture_output=True,
        text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=30,
    )
    printed = json.loads(completed.stdout)
    evidence = json.loads((output / "evidence.json").read_text())
    assert printed["orders_submitted"] == 0
    assert printed["action"] == "LONG"
    assert [r["provider"] for r in printed["ranking"]] == ["zero-return", "empirical"]
    assert [r["score"] for r in printed["ranking"]] == pytest.approx([.0004, .0012])
    assert printed["ranking"] == evidence["comparison"]["evidence_summary"]["lifetime"]["ranking"]
    assert printed["outcome"] == evidence["outcome"]
    assert printed["outcome"]["policy_net_simple_return"] == pytest.approx(expm1(.0004) - .001)
    assert printed["outcome"]["hypothetical"] is True
    assert "missed the cost hurdle" in evidence["lesson"]["lesson"]
    assert evidence["lesson"]["metrics"] == evidence["review"]["metrics"]

    ledger = TemporalLedger(output / "ledger.db", create=False)
    decision_id = evidence["intent"]["decision_id"]
    for review in (evidence["review"], evidence["revised_review"]):
        retrieved = ledger.review_decision(
            decision_id=decision_id, source_as_of=review["source_as_of"],
            recorded_as_of=review["recorded_as_of"],
        )
        assert retrieved == review
        assert review["scored_pairs"][0]["recorded_at"] == review["recorded_as_of"]
    assert evidence["review"]["actual_ids"] != evidence["revised_review"]["actual_ids"]
    # A later source cutoff still cannot expose a revision recorded after the local cutoff.
    mixed = ledger.review_decision(
        decision_id=decision_id, source_as_of=evidence["revised_review"]["source_as_of"],
        recorded_as_of=evidence["review"]["recorded_as_of"],
    )
    assert mixed["actual_ids"] == evidence["review"]["actual_ids"]
    saved = ledger.export_lesson(lesson_id=evidence["lesson"]["lesson_id"],
                                 recorded_as_of=evidence["revised_review"]["recorded_as_of"])
    assert saved == evidence["lesson"]
    context = evidence["review"]["summary"]["context"][0]
    assert context["value"] == "positive-trailing-24h"
    assert context["recorded_at"] < evidence["review"]["scored_pairs"][0]["timestamp"]


def test_changed_outcome_changes_lesson_and_ranking(demo, tmp_path, capsys):
    output = tmp_path / "profitable"
    demo["main"](output, observed_log_return=.002)
    printed = json.loads(capsys.readouterr().out)
    evidence = json.loads((output / "evidence.json").read_text())
    assert "exceeded the cost hurdle" in evidence["lesson"]["lesson"]
    assert "missed the cost hurdle" not in evidence["lesson"]["lesson"]
    assert printed["ranking"][0]["provider"] == "empirical"
    assert printed["outcome"]["policy_net_simple_return"] == pytest.approx(expm1(.002) - .001)
    journal = (output / "intent.json").read_bytes()
    with pytest.raises(FileExistsError):
        demo["main"](output)
    assert (output / "intent.json").read_bytes() == journal


def test_policy_boundaries_and_no_trade_costs(demo):
    hurdle = log1p(.001)
    assert demo["choose_intent"](hurdle, .001) == "NO_NEW_POSITION"
    assert demo["choose_intent"](.002, .001) == "LONG"
    for value, relation in ((-.001, "missed"), (hurdle, "matched"), (.002, "exceeded")):
        result = demo["assess_outcome"]("LONG", value, .001)
        assert result["hurdle_relation"] == relation
        assert result["policy_net_simple_return"] == pytest.approx(expm1(value) - .001)
    assert demo["assess_outcome"]("NO_NEW_POSITION", -.001, .001)["policy_net_simple_return"] == 0


def test_regime_uses_only_last_24_completed_returns(demo):
    assert demo["classify_regime"]([100.] * 36 + [-.001] * 24) == "nonpositive-trailing-24h"
    assert demo["classify_regime"]([-100.] * 36 + [.001] * 24) == "positive-trailing-24h"
    assert demo["classify_regime"]([0.] * 60) == "nonpositive-trailing-24h"


def test_documented_mcp_summary_call(demo, tmp_path, capsys):
    output = tmp_path / "mcp"
    demo["main"](output)
    capsys.readouterr()
    evidence = json.loads((output / "evidence.json").read_text())
    examples = re.findall(r"```json\n(.*?)\n```", (SKILL / "references/trade-lifecycle.md").read_text(), re.S)
    call = json.loads(examples[0])
    call["arguments"]["execution_id"] = evidence["intent"]["execution_id"]
    # Use the recorded fixture decision time to validate the documented tool payload.
    origin = datetime.fromisoformat(evidence["review"]["summary"]["origin"])
    ledger = TemporalLedger(output / "ledger.db", clock=FixedClock(origin), create=False)
    config = tmp_path / "providers.toml"
    config.write_text("schema_version = 1\nallow_outcome_writes = true\n")
    with GnomonSession.from_config(config, ledger=ledger) as session:
        reply = session.call(call["name"], call["arguments"], compact=False)
    assert reply["status"] == "ok"
    stored = ledger.decision(reply["result"]["decision_id"])
    assert stored["inputs"]["rationale"] == call["arguments"]["rationale"]
    assert stored["execution_ids"] == [evidence["intent"]["execution_id"]]


def test_reusable_module_never_touches_the_ledger_clock():
    tree = ast.parse(DECISIONS.read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "FixedClock" not in names
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        assert not any(isinstance(t, ast.Attribute) and t.attr == "clock" for t in targets)
        if isinstance(node, ast.Call):
            assert not any(k.arg == "clock" for k in node.keywords)


def _decision_count(path):
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]


def _recorded_forecast_request():
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=4)
    return dict(
        history=[.001, -.002, .0015, .0005], horizon=1, series_id="test/log-return/1h",
        unit="log_return", timestamps=[(start + timedelta(hours=i)).isoformat() for i in range(4)],
        future_timestamps=[(start + timedelta(hours=4)).isoformat()],
    )


def _recorded_forecast(ledger):
    with GnomonSession.from_config(ledger=ledger) as session:
        reply = session.forecast("last_value", _recorded_forecast_request())
    assert reply["status"] == "ok", reply
    return reply


def _decide(demo, ledger, journal, forecast, **overrides):
    overrides.setdefault("mode", "paper")
    return demo["record_trade_decision"](
        ledger, journal, forecast=forecast, action="NO_NEW_POSITION", account="acct",
        leg="entry-0", event_id="event-1", policy_revision="test-v1", rationale="test",
        assumptions=["test"], invalidation_conditions=["test"], **overrides,
    )


def test_record_trade_decision_links_journal_with_system_clock(demo, tmp_path):
    ledger = TemporalLedger(tmp_path / "ledger.db")
    forecast = _recorded_forecast(ledger)
    intent = _decide(demo, ledger, tmp_path / "intent.json", forecast)
    assert json.loads((tmp_path / "intent.json").read_text()) == intent
    assert intent["client_order_id"] == demo["client_order_id"]("acct", intent["decision_id"], "entry-0")
    assert ledger.decision(intent["decision_id"])["execution_ids"] == [forecast["execution_id"]]
    assert intent["quantity"] is None and intent["submitted"] is False


def test_record_trade_decision_refuses_fixture_clock_without_writing(demo, tmp_path):
    ledger = TemporalLedger(tmp_path / "ledger.db")
    forecast = _recorded_forecast(ledger)
    ledger.clock = FixedClock(datetime.now(timezone.utc))
    with pytest.raises(ValueError, match="system clock"):
        _decide(demo, ledger, tmp_path / "intent.json", forecast)
    assert not (tmp_path / "intent.json").exists()
    assert _decision_count(tmp_path / "ledger.db") == 0


def test_existing_journal_blocks_a_second_decision(demo, tmp_path):
    ledger = TemporalLedger(tmp_path / "ledger.db")
    forecast = _recorded_forecast(ledger)
    journal = tmp_path / "intent.json"
    _decide(demo, ledger, journal, forecast)
    saved = journal.read_bytes()
    with pytest.raises(FileExistsError):
        _decide(demo, ledger, journal, forecast)
    assert journal.read_bytes() == saved
    assert _decision_count(tmp_path / "ledger.db") == 1


def _mode_labels(ledger, decision_id):
    return [c for c in ledger.decision(decision_id)["inputs"]["context"] if c["key"] == "trading_mode"]


def test_mode_is_stored_in_the_ledger(demo, decisions, tmp_path):
    ledger = TemporalLedger(tmp_path / "paper.db")
    intent = _decide(demo, ledger, tmp_path / "intent.json", _recorded_forecast(ledger))
    [label] = _mode_labels(ledger, intent["decision_id"])
    assert (label["value"], label["source_ref"]) == ("paper", "operator-mode:paper")
    assert intent["mode"] == "paper" and intent["authorization_ref"] is None
    assert decisions["ledger_modes"](ledger) == {"paper"}


@pytest.mark.parametrize("mode", ["synthetic", "live"])
def test_invalid_mode_or_unpromoted_live_writes_nothing(demo, tmp_path, mode):
    ledger = TemporalLedger(tmp_path / "live.db")
    forecast = _recorded_forecast(ledger)
    with pytest.raises(ValueError, match="mode must be one of" if mode != "live" else "promotion_record"):
        _decide(demo, ledger, tmp_path / "intent.json", forecast, mode=mode)
    assert not (tmp_path / "intent.json").exists()
    assert _decision_count(tmp_path / "live.db") == 0


def _paper_with_outcomes(decisions, tmp_path, count):
    paper = TemporalLedger(tmp_path / "paper.db")
    targets = set()
    for n in range(count):
        forecast = _recorded_forecast(paper)
        decisions["record_trade_decision"](
            paper, tmp_path / f"paper-{n}.json", mode="paper", forecast=forecast, action="NO_NEW_POSITION",
            account="acct", leg="entry-0", event_id=f"paper-{n}", policy_revision="test-v1",
            rationale="test", assumptions=["test"], invalidation_conditions=["test"])
        request = paper.execution(forecast["execution_id"])["request"]
        targets.add((request["series_id"], request["unit"], request["future_timestamps"][0]))
    now = datetime.now(timezone.utc).isoformat()
    for series_id, unit, valid_time in sorted(targets):
        paper.append_actual(series_id=series_id, unit=unit, valid_time=valid_time, value=.001,
                            source_available_at=now, source_ref="test:close")
    return paper


def _promotion(decisions, paper, path, *, minimum=2, approved_by="the user", **changes):
    now = datetime.now(timezone.utc).isoformat()
    review = decisions["promotion_review"](paper, source_as_of=now, recorded_as_of=now)
    record = dict(review={**review, **changes}, min_complete_decisions=minimum,
                  criteria="test criteria", approved_by=approved_by, approved_at=now)
    path.write_text(json.dumps(record))
    return path


def test_live_decision_requires_a_verified_promotion(decisions, tmp_path):
    paper = _paper_with_outcomes(decisions, tmp_path, 2)
    live = TemporalLedger(tmp_path / "live.db")
    forecast = _recorded_forecast(live)
    record = _promotion(decisions, paper, tmp_path / "promotion.json")
    intent = _decide(decisions, live, tmp_path / "intent.json", forecast, mode="live",
                     promotion_record=record, paper_ledger=paper)
    [label] = _mode_labels(live, intent["decision_id"])
    assert label["value"] == "live"
    assert label["source_ref"] == intent["authorization_ref"]
    assert intent["authorization_ref"].startswith("promotion:promotion.json:sha256:")


@pytest.mark.parametrize("promotion, message", [
    (dict(minimum=3), "Paper minimum not met"),
    (dict(approved_by=" "), "no approver"),
    (dict(complete_decisions=5), "no longer matches"),
])
def test_live_refuses_an_invalid_promotion(decisions, tmp_path, promotion, message):
    paper = _paper_with_outcomes(decisions, tmp_path, 2)
    live = TemporalLedger(tmp_path / "live.db")
    forecast = _recorded_forecast(live)
    record = _promotion(decisions, paper, tmp_path / "promotion.json", **promotion)
    with pytest.raises(ValueError, match=message):
        _decide(decisions, live, tmp_path / "intent.json", forecast, mode="live",
                promotion_record=record, paper_ledger=paper)
    assert not (tmp_path / "intent.json").exists()
    assert _decision_count(tmp_path / "live.db") == 0


def test_promotion_review_rejects_a_mixed_ledger(decisions, tmp_path):
    ledger = TemporalLedger(tmp_path / "backtest.db")
    _decide(decisions, ledger, tmp_path / "b.json", _recorded_forecast(ledger), mode="backtest")
    now = datetime.now(timezone.utc).isoformat()
    with pytest.raises(ValueError, match="paper-only ledger"):
        decisions["promotion_review"](ledger, source_as_of=now, recorded_as_of=now)


def test_ledger_refuses_a_second_mode(demo, tmp_path):
    ledger = TemporalLedger(tmp_path / "evidence.db")
    forecast = _recorded_forecast(ledger)
    _decide(demo, ledger, tmp_path / "backtest.json", forecast, mode="backtest")
    with pytest.raises(ValueError, match="separate ledger for paper"):
        _decide(demo, ledger, tmp_path / "paper.json", forecast, mode="paper")
    assert not (tmp_path / "paper.json").exists()
    assert _decision_count(tmp_path / "evidence.db") == 1


def test_demo_records_backtest_mode(demo, decisions, tmp_path, capsys):
    output = tmp_path / "mode"
    demo["main"](output)
    capsys.readouterr()
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["intent"]["mode"] == "backtest"
    ledger = TemporalLedger(output / "ledger.db", create=False)
    assert decisions["ledger_modes"](ledger) == {"backtest"}


def test_documented_mode_config_labels_mcp_decisions(tmp_path):
    [toml] = re.findall(r"```toml\n(.*?)\n```", (SKILL / "references/trade-lifecycle.md").read_text(), re.S)
    config = tmp_path / "paper.toml"
    config.write_text(toml)
    with GnomonSession.from_config(config) as session:
        assert session.capabilities()["ledger"]["decision_context"] == {"trading_mode": "paper"}
        forecast = session.forecast("last_value", _recorded_forecast_request())
        reply = session.call("gnomon_ledger", dict(
            operation="record_decision_summary", execution_id=forecast["execution_id"], rationale="r",
            assumptions=["a"], invalidation_conditions=["i"], context=[]), compact=False)
    ledger = TemporalLedger(tmp_path / "paper.db", create=False)
    [label] = ledger.decision(reply["result"]["decision_id"])["inputs"]["context"]
    assert (label["key"], label["value"]) == ("trading_mode", "paper")


@pytest.mark.parametrize('field,value', [
    ('criteria', None), ('criteria', ''), ('criteria', []),
    ('approved_at', None), ('approved_at', 'not-a-date'),
    ('approved_at', '2026-01-01T00:00:00'),
    ('approved_at', '1900-01-01T00:00:00Z'),
    ('approved_at', '2999-01-01T00:00:00Z'),
    ('min_complete_decisions', None), ('min_complete_decisions', -5),
    ('min_complete_decisions', 0), ('min_complete_decisions', True),
    ('min_complete_decisions', 1.5), ('min_complete_decisions', '2'),
])
def test_malformed_approval_cannot_record_live_intent(decisions, tmp_path, field, value):
    paper = _paper_with_outcomes(decisions, tmp_path, 2)
    live = TemporalLedger(tmp_path / 'live.db')
    forecast = _recorded_forecast(live)
    path = _promotion(decisions, paper, tmp_path / 'promotion.json')
    record = json.loads(path.read_text())
    if value is None:
        record.pop(field)
    else:
        record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        _decide(decisions, live, tmp_path / 'intent.json', forecast, mode='live',
                promotion_record=path, paper_ledger=paper)
    assert not (tmp_path / 'intent.json').exists()
    assert _decision_count(tmp_path / 'live.db') == 0
