from datetime import datetime, timedelta, timezone
import math

import pytest

from gnomon import ForecastResult, GnomonSession, InferenceEngine, TemporalLedger
from gnomon.evidence_summary import comparison_summary, rmsle
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock
from gnomon.session import ledger_schema


def test_metric_domain_and_explicit_clipping():
    assert rmsle([(0, 0), (3, 1)])[0] == pytest.approx(math.log(2) / math.sqrt(2))
    with pytest.raises(ForecastAdapterError, match='negative_prediction'):
        rmsle([(-1, 0)])
    assert rmsle([(-1, 0)], 'clip_zero') == (0, 1)
    with pytest.raises(ForecastAdapterError, match='nonnegative_actuals'):
        rmsle([(1, -1)], 'clip_zero')


def test_ranks_ties_zero_denominator_and_recent_disagreement():
    origins = [dict(origin=str(i), n=1, actual_ids=[str(i)],
                    models=[dict(provider='a', mae=a), dict(provider='b', mae=b)])
               for i, (a, b) in enumerate(((0, 3), (0, 3), (2, 0)))]
    summary = comparison_summary(origins, ['a', 'b'], recent_origins=1)
    assert summary['lifetime']['ranking'][0]['provider'] == 'a'
    assert summary['recent']['ranking'][0]['provider'] == 'b'
    assert summary['recent_lifetime_disagreement']
    assert summary['recent']['pairwise_differences'][0]['relative_status'] == 'undefined_zero_reference'
    origins[-1]['models'][0]['mae'] = 0
    tie = comparison_summary(origins[-1:], ['a', 'b'])
    assert [m['rank'] for m in tie['lifetime']['ranking']] == [1, 1]
    assert tie['lifetime']['ties'] == [['a', 'b']]


def test_real_ledger_metric_reversal_visibility_and_no_calls(tmp_path):
    origin = datetime(2026, 1, 2, tzinfo=timezone.utc)
    ledger = TemporalLedger(tmp_path / 'test.db', clock=FixedClock(origin))
    engine = InferenceEngine(ledger=ledger)
    calls = []
    def provider(name, predictions):
        def forecast(r):
            calls.append(name)
            return ForecastResult(point=(predictions[int(r.history[-1] == 1000)],),
                series_id=r.series_id, unit=r.unit, timestamps=r.future_timestamps)
        return forecast
    # A loses on MAE, B loses on RMSLE because relative errors differ by level.
    for name, points in [('a', (0, 1100)), ('b', (1, 1000))]:
        engine.register(name, provider(name, points), revision='v1', deterministic=True, lifecycle='stateless')
    revisions = {'a': 'v1', 'b': 'v1'}
    for i, actual in enumerate((0, 1000)):
        now = origin + timedelta(days=2*i)
        ledger.clock = FixedClock(now)
        request = dict(history=[actual], horizon=1, series_id='sales', unit='widgets',
            timestamps=[now.isoformat()], cutoff=now.isoformat(), frequency='D',
            future_timestamps=[(now+timedelta(days=1)).isoformat()])
        for p in revisions:
            engine.forecast(p, request)
        ledger.clock = FixedClock(now+timedelta(days=1))
        ledger.append_actual(series_id='sales', unit='widgets', valid_time=request['future_timestamps'][0],
                             source_available_at=ledger._now(), value=actual)
    query = dict(series_id='sales', unit='widgets', horizon=1, providers=revisions,
                 start=origin.isoformat(), end=(origin+timedelta(days=2)).isoformat(),
                 source_as_of=ledger._now(), recorded_as_of=ledger._now(), recent_origins=1)
    count, writes = len(calls), ledger._committed_row_writes
    mae = ledger.compare_history(**query)
    log = ledger.compare_history(**query, metric='rmsle')
    assert mae['evidence_summary']['lifetime']['ranking'][0]['provider'] == 'b'
    assert log['evidence_summary']['lifetime']['ranking'][0]['provider'] == 'a'
    assert log['models'][0]['rmsle'] == pytest.approx(math.log(1101/1001)/2)
    assert log['matched_origins'] == 2
    assert len(log['origins'][0]['models'][0]['scored_pairs_sha256']) == 64
    assert len(calls) == count and ledger._committed_row_writes == writes
    early = ledger.compare_history(**{**query, 'recorded_as_of': (origin+timedelta(days=2)).isoformat()}, metric='rmsle')
    assert early['matched_origins'] == 1
    assert early['excluded'][0]['reason'] == 'incomplete_actuals'
    session = GnomonSession(engine=engine, ledger=ledger)
    reply = session.call('gnomon_ledger', {'operation': 'compare_history', **query, 'metric': 'rmsle'}, compact=False)
    assert reply['status'] == 'ok'
    assert reply['result']['evidence_summary']['metric'] == 'rmsle'


def test_comparison_schema_exposes_optional_metric_controls():
    schema = ledger_schema(allow_outcome_writes=True)
    for operation in ('compare_history', 'compare_context'):
        branch = next(b for b in schema['oneOf'] if b['properties']['operation']['const'] == operation)
        assert branch['properties']['metric']['default'] == 'mae'
        assert 'metric' not in branch['required']
