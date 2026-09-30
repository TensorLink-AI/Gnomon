"""Run the examples in every packaged skill against the real contract, and lint the set."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import pytest

from gnomon import GnomonSession, TemporalLedger
from gnomon.ids import FixedClock
from gnomon.recovery import _matches
from gnomon.session import configuration_schema


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
SKILL_FILES = sorted(SKILLS.glob("*/SKILL.md"))


def _blocks(skill, language):
    return re.findall(rf"```{language}\n(.*?)\n```", (SKILLS / skill / "SKILL.md").read_text(), re.S)


def _frontmatter(path):
    import yaml
    return yaml.safe_load(path.read_text().split("---")[1])


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_skill_metadata_and_local_links(path):
    meta = _frontmatter(path)
    assert meta["name"] == path.parent.name
    assert "Use when" in meta["description"]
    for target in re.findall(r"\]\((?!https?://)([^)#]+)", path.read_text()):
        assert (path.parent / target).resolve().exists(), f"{path.parent.name}: broken link {target}"


def test_skill_names_and_descriptions_are_distinct():
    descriptions = [_frontmatter(p)["description"] for p in SKILL_FILES]
    assert len(set(descriptions)) == len(descriptions)


def _run(code, cwd):
    completed = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True,
                               timeout=60, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_forecast_skill_file_example_final_check_and_custom_model(tmp_path):
    save, check, custom = _blocks("forecast-with-gnomon", "python")
    (tmp_path / "history.csv").write_text(
        "timestamp,value\n" + "".join(f"2026-01-{d:02d}T00:00:00Z,{10 + d}\n" for d in range(1, 11)))
    (tmp_path / "future.csv").write_text("timestamp\n2026-01-11T00:00:00Z\n2026-01-12T00:00:00Z\n")
    (tmp_path / "task.json").write_text('{"series_id": "s", "unit": "u", "horizon": 2}')
    assert "final method: last_value" in _run(save + "\n" + check, tmp_path)
    saved = json.loads((tmp_path / "forecast.json").read_text())
    assert saved["point"] == [20.0, 20.0]
    out = _run(save + "\npredictions = [1.0, 2.0]\n" + custom + "\nprint(reply['result']['point'])", tmp_path)
    assert "(1.0, 2.0)" in out


def test_forecast_skill_final_check_rejects_an_unsaved_forecast(tmp_path):
    save, check, _ = _blocks("forecast-with-gnomon", "python")
    (tmp_path / "history.csv").write_text("timestamp,value\n2026-01-01T00:00:00Z,1\n2026-01-02T00:00:00Z,2\n")
    (tmp_path / "future.csv").write_text("timestamp\n2026-01-03T00:00:00Z\n")
    (tmp_path / "task.json").write_text('{"series_id": "s", "unit": "u", "horizon": 1}')
    _run(save, tmp_path)
    saved = json.loads((tmp_path / "forecast.json").read_text())
    (tmp_path / "forecast.json").write_text(json.dumps({**saved, "point": [99.0]}))
    completed = subprocess.run([sys.executable, "-c", "import json\nfrom pathlib import Path\n" + check],
                               cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert completed.returncode != 0 and "AssertionError" in completed.stderr


def test_ephemeris_setup_toml_and_tool_call_match_the_contract():
    [toml] = _blocks("setup-gnomon-ephemeris", "toml")
    config = tomllib.loads(toml)
    assert _matches(config, configuration_schema())
    assert set(config["providers"]) == {"ephemeris"}
    calls = [json.loads(b) for b in _blocks("setup-gnomon-ephemeris", "json")]
    tool_call = next(c for c in calls if c.get("name") == "gnomon_forecast")
    assert tool_call["arguments"]["provider"] == "ephemeris"
    # The same request is valid for an offline provider: only the provider differs.
    with GnomonSession.from_config() as session:
        request = {k: v for k, v in tool_call["arguments"]["request"].items() if k != "quantiles"}
        assert session.call("gnomon_forecast", {"provider": "last_value", "request": request},
                            compact=False)["status"] == "ok"


def _ledger_session(ledger, tmp_path):
    config = tmp_path / "ledger.toml"
    config.write_text("schema_version = 1\nallow_outcome_writes = true\n")
    return GnomonSession.from_config(config, ledger=ledger)


def test_ledger_skill_worked_calls_run_on_a_real_ledger(tmp_path):
    calls = {c["arguments"].get("operation"): c for c in map(json.loads, _blocks("use-gnomon-ledger", "json"))
             if c["name"] == "gnomon_ledger"}
    search, compare, review = calls["search"], calls["compare_history"], calls["review_decision"]
    origin = datetime(2026, 1, 10, tzinfo=timezone.utc)
    ledger = TemporalLedger(tmp_path / "ledger.db", clock=FixedClock(origin))
    request = dict(history=[1.0, 2.0, 3.0], horizon=1, series_id="SERIES_ID", unit="UNIT",
                   timestamps=[(origin - timedelta(days=3 - i)).isoformat() for i in range(3)],
                   future_timestamps=[(origin + timedelta(days=1)).isoformat()])
    with _ledger_session(ledger, tmp_path) as session:
        runs = [session.forecast(p, request) for p in ("last_value", "historical_mean")]
        decision = ledger.record_decision_summary(
            execution_id=runs[0]["execution_id"], rationale="r", assumptions=["a"],
            invalidation_conditions=["i"], context=[])
        ledger.clock = FixedClock(origin + timedelta(days=2))
        ledger.append_actual(series_id="SERIES_ID", unit="UNIT", valid_time=request["future_timestamps"][0],
                             value=2.5, source_available_at=(origin + timedelta(days=2)).isoformat(),
                             source_ref="test")
        cutoff = (origin + timedelta(days=2)).isoformat()

        found = session.call(search["name"], search["arguments"], compact=False)
        assert found["status"] == "ok" and len(found["result"]["items"]) == 2

        forecast_origin = request["timestamps"][-1]
        arguments = {**compare["arguments"], "start": forecast_origin, "end": forecast_origin,
                     "source_as_of": cutoff, "recorded_as_of": cutoff,
                     "providers": {r["provider"]: r["revision"] for r in runs}}
        assert set(arguments) == set(compare["arguments"])
        ranked = session.call(compare["name"], arguments, compact=False)["result"]
        assert ranked["matched_origins"] == 1 and ranked["metric"] == "mae"

        reviewed = session.call(review["name"], {**review["arguments"], "decision_id": decision["decision_id"],
                                                 "source_as_of": cutoff, "recorded_as_of": cutoff}, compact=False)
        assert reviewed["result"]["review_ready"]


def test_ledger_skill_rescore_call_matches_the_evaluate_schema():
    from gnomon.recovery import _matches
    calls = [json.loads(b) for b in _blocks("use-gnomon-ledger", "json")]
    rescore = next(c for c in calls if c["arguments"].get("operation") == "rescore")
    assert rescore["name"] == "gnomon_evaluate"
    with GnomonSession.from_config() as session:
        schema = next(t for t in session.tools() if t["name"] == "gnomon_evaluate")["inputSchema"]
        inspect = next(t for t in session.tools() if t["name"] == "gnomon_inspect")["inputSchema"]
    assert _matches(rescore["arguments"], schema)
    assert _matches(next(c for c in calls if c["name"] == "gnomon_inspect")["arguments"], inspect)



@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_every_skill_declares_an_agent_interface(path):
    text = (path.parent / "agents/openai.yaml").read_text()
    assert "display_name:" in text and "short_description:" in text
    assert f"${path.parent.name}" in text
