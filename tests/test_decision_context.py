"""Operator-configured decision labels (decision_context) on CLI/MCP summary writes."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.contracts import GnomonError
from gnomon.forecast_adapter import ForecastAdapterError


def _config(tmp_path, ledger_path, mode, source_ref="operator-config", name="gnomon.toml"):
    path = tmp_path / name
    path.write_text("schema_version = 1\nallow_outcome_writes = true\n"
                    f"ledger_path = {json.dumps(str(ledger_path))}\n"
                    f"[decision_context.trading_mode]\nvalue = {json.dumps(mode)}\n"
                    f"source_ref = {json.dumps(source_ref)}\n")
    return path


def _request():
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=4)
    return dict(history=[1., 2., 3., 4.], horizon=1, series_id="ctx/series", unit="u",
                timestamps=[(start + timedelta(hours=i)).isoformat() for i in range(4)],
                future_timestamps=[(start + timedelta(hours=4)).isoformat()])


def _summary(execution_id, context=()):
    return dict(operation="record_decision_summary", execution_id=execution_id, rationale="r",
                assumptions=["a"], invalidation_conditions=["i"], context=list(context))


def _record(config, context=()):
    with GnomonSession.from_config(config) as session:
        execution_id = session.forecast("last_value", _request())["execution_id"]
        return session, session.call("gnomon_ledger", _summary(execution_id, context), compact=False)


def test_operator_label_is_added_and_visible(tmp_path):
    config = _config(tmp_path, tmp_path / "paper.db", "paper")
    session, reply = _record(config)
    assert reply["status"] == "ok", reply
    assert session.capabilities()["ledger"]["decision_context"] == {"trading_mode": "paper"}
    ledger = TemporalLedger(tmp_path / "paper.db", create=False)
    [label] = ledger.decision(reply["result"]["decision_id"])["inputs"]["context"]
    assert (label["key"], label["value"], label["source_ref"]) == (
        "trading_mode", "paper", "operator:operator-config")
    assert label["valid_from"] < label["valid_to"] and label["source_available_at"] <= label["recorded_at"]


def test_caller_cannot_set_an_operator_key(tmp_path):
    config = _config(tmp_path, tmp_path / "paper.db", "paper")
    now = datetime.now(timezone.utc)
    forged = dict(key="trading_mode", value="live", valid_from=(now - timedelta(hours=1)).isoformat(),
                  valid_to=(now + timedelta(hours=1)).isoformat(),
                  source_available_at=(now - timedelta(hours=1)).isoformat(), source_ref="agent")
    with pytest.raises(GnomonError, match="operator configuration"):
        _record(config, [forged])


def test_ledger_refuses_a_second_configured_value(tmp_path):
    ledger_path = tmp_path / "shared.db"
    assert _record(_config(tmp_path, ledger_path, "paper", name="paper.toml"))[1]["status"] == "ok"
    live = _config(tmp_path, ledger_path, "live", "promotion-review:1", name="live.toml")
    with pytest.raises(GnomonError, match="separate ledger"):
        _record(live)
    with GnomonSession.from_config(live) as session:
        assert session.call("gnomon_ledger", {"operation": "search"}, compact=False)["status"] == "ok"
    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1


@pytest.mark.parametrize("decision_context", [
    {f"k{i}": {"value": "v", "source_ref": "s"} for i in range(5)},
    {"trading_mode": {"value": "paper"}},
    {"trading_mode": {"value": "paper", "source_ref": "s", "extra": "x"}},
    {"trading_mode": {"value": "", "source_ref": "s"}},
])
def test_invalid_decision_context_is_rejected(decision_context):
    with pytest.raises(ForecastAdapterError):
        GnomonSession(decision_context=decision_context)


def test_cli_ledger_write_applies_operator_label(tmp_path):
    config = _config(tmp_path, tmp_path / "live.db", "live", "promotion-review:7")
    with GnomonSession.from_config(config) as session:
        execution_id = session.forecast("last_value", _request())["execution_id"]
    cli = subprocess.run([sys.executable, "-m", "gnomon", "ledger", "--providers-config", str(config),
                          "--arguments", json.dumps(_summary(execution_id))],
                         text=True, capture_output=True, timeout=30,
                         env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert cli.returncode == 0, cli.stderr
    printed = json.loads(cli.stdout)
    decision_id = printed["result"]["decision_id"]
    ledger = TemporalLedger(tmp_path / "live.db", create=False)
    [label] = ledger.decision(decision_id)["inputs"]["context"]
    assert (label["value"], label["source_ref"]) == ("live", "operator:promotion-review:7")
