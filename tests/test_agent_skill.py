"""Execute packaged skill examples against the actual agent contract."""

import json
from pathlib import Path
import re

from gnomon import GnomonSession


SKILL = Path(__file__).resolve().parents[1] / "skills/use-gnomon/SKILL.md"


def test_skill_forecast_example_uses_exposed_tool_and_executes_offline():
    examples = re.findall(r"```json\n(.*?)\n```", SKILL.read_text(), re.S)
    assert examples, "The skill must retain its runnable tool-call example"
    with GnomonSession.from_config() as session:
        exposed = {tool["name"] for tool in session.tools()}
        for example in examples:
            call = json.loads(example)
            assert call["name"] in exposed
            result = session.call(call["name"], call["arguments"], compact=False)
            assert result["status"] == "ok"
            assert result["provider"] == "last_value"
            assert list(result["result"]["point"]) == [11.0, 11.0]
            assert result["action_authorized"] is False
            assert result["recorded"] is False


def test_ledger_skill_rescore_example_uses_revised_observations(tmp_path):
    from gnomon.ids import FixedClock
    from gnomon.temporal_store import TemporalObservation, TemporalStore
    from test_backtesting import at, configured
    skill = SKILL.parent.parent / 'use-gnomon-ledger/SKILL.md'
    calls = [json.loads(v) for v in re.findall(r'```json\n(.*?)\n```', skill.read_text(), re.S)]
    inspect = next(c for c in calls if c['name'] == 'gnomon_inspect')
    rescore = next(c for c in calls if c['arguments'].get('operation') == 'rescore')
    session, _ = configured(tmp_path, ledger=True)
    with session:
        path = tmp_path / 'vintages.db'
        store = TemporalStore(path)
        for day in range(1, 31):
            store.ingest_rows('sales', [TemporalObservation('a','value',at(day),at(day),day)],
                              source_fingerprint=str(day), clock=FixedClock(at(day)))
        old = session.data.inspect('store:sales', store_path=str(path), unit='UNIT',
            as_of=at(30).isoformat(), recorded_as_of=at(30).isoformat())
        original = session.evaluate(old['data_ref'], candidates=['trend'], baseline='last_value', horizon=2)
        saved = session.ledger.study(original['study_id'])
        for day in (29, 30):
            store.ingest_rows('sales', [TemporalObservation('a','value',at(day),at(30),500)],
                              source_fingerprint='revision', clock=FixedClock(at(34)))
        args = inspect['arguments'] | {'input':'store:sales', 'store_path':str(path),
            'as_of':at(30).isoformat(), 'recorded_as_of':at(35).isoformat()}
        revised = session.call(inspect['name'], args, compact=False)
        result = session.call(rescore['name'], rescore['arguments'] | {
            'study_id':original['study_id'], 'data_ref':revised['data_ref'],
            'source_as_of':at(30).isoformat(), 'recorded_as_of':at(35).isoformat()}, compact=False)
        assert result['usage']['provider_calls'] == 0
        assert result['scores']['trend']['mae'] > saved['scores']['trend']['mae']
        assert session.ledger.study(original['study_id']) == saved


def test_ledger_skill_schema_recovery_works_without_cli(tmp_path):
    from gnomon import TemporalLedger
    with GnomonSession.from_config(ledger=TemporalLedger(tmp_path / 'ledger.db')) as session:
        result = session.call('gnomon_capabilities', {'schema_tool':'gnomon_ledger',
            'schema_variant':'review_decision'}, compact=False)
        assert 'decision_id' in result['schema']['required']
