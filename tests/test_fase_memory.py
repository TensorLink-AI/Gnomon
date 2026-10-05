"""Numerical and causal checks for the optional FASE-style memory."""
import cmath
import math

import pytest

from gnomon import fase_memory as fase
from gnomon.episodic_memory import compute_features, episodic_evidence, validate_memory
from gnomon.adaptive_router import replay_router, validate_policy
from gnomon.forecast_adapter import ForecastAdapterError


def row(i, value, losses=None, served='base', receipt=()):
    return {'series_id': 's', 'origin': f'2026-01-{i:02d}T00:00:00+00:00',
            'completed_at': f'2026-01-{i+1:02d}T00:00:00+00:00',
            'features': {'trend': value}, 'losses': losses or {'base': 1., 'a': .5},
            'served_provider': served, 'retention_receipt': list(receipt)}


def policy(**extra):
    return validate_policy({'baseline': 'base', 'candidates': ['a'], 'min_origins': 1,
                            'memory': {'features': ['trend'], 'distance': 'fase', 'retention': 'fase',
                                       'k': 2, 'min_effective_n': 1, 'recent_capacity': 2,
                                       'long_term_capacity': 2, **extra}})


def test_group_distance_and_constant_feature_limit():
    a = {'fase:missing_ratio': 0., 'fase:zero_ratio': 0., 'fase:change_strength': 0.}
    b = dict(zip(a, [1., 3., 1.]))
    scales = {n: (0., 1.) for n in a}
    assert fase.distance(a, b, list(a), scales) == pytest.approx(((.5+.75)/2+.5)/2)
    assert fase.distance({'x': 0}, {'x': 1e100}, ['x'], {'x': (0, 1)}) <= 1
    assert fase.distance({'x': 0}, {'x': 1}, ['x'], {'x': (0, 0)}) == 1
    assert fase.distance({'x': 0}, {'x': 0}, ['x'], {'x': (0, 0)}) == 0
    assert fase.distance({'x': None}, {'x': 0}, ['x'], {'x': (0, 1)}) is None


@pytest.mark.parametrize('n', [3, 8, 11, 56])
def test_pure_python_spectrum_matches_direct_dft(n):
    xs = [math.sin(i*.73)+math.cos(i*.29) for i in range(n)]
    expected = [abs(sum(x*cmath.exp(-2j*math.pi*k*t/n) for t,x in enumerate(xs)))**2
                for k in range(1,n//2+1)]
    assert fase._power(xs) == pytest.approx(expected, abs=1e-9)


def test_profile_has_18_features_and_covariates_use_history():
    spec = validate_memory({'profile': 'fase', 'long_window': 56})
    xs = [math.sin(i*.3)+i*.02 for i in range(56)]
    out = compute_features(spec, xs, past_covariates=[[x] for x in xs], past_covariate_names=['market'])
    assert len(out) == 18
    assert out['fase:strongest_covariate_relation'] == pytest.approx(1.)
    assert out['fase:covariate_relation_stability'] == pytest.approx(1.)
    assert 0 <= out['fase:spectral_entropy'] <= 1
    assert out['fase:missing_ratio'] == 0
    assert out['fase:longest_missing_block'] is None
    assert all(v is None or math.isfinite(v) for v in out.values())


def test_masked_values_do_not_change_features():
    spec = validate_memory({'profile': 'fase', 'long_window': 56, 'mask_covariate': 'observed'})
    xs = [math.sin(i) for i in range(56)]
    mask = [[0 if 12<=i<17 else 1] for i in range(56)]
    changed = [999999 if not mask[i][0] else v for i,v in enumerate(xs)]
    a = compute_features(spec, xs, past_covariates=mask, past_covariate_names=['observed'])
    b = compute_features(spec, changed, past_covariates=mask, past_covariate_names=['observed'])
    assert a == b
    assert a['fase:longest_missing_block'] == 5
    assert a['fase:spectral_entropy'] is None
    assert a['fase:strongest_covariate_relation'] is None


def test_empty_constant_and_short_histories():
    spec = validate_memory({'profile': 'fase'})
    for xs in ([], [0], [2]*100, [None]*5):
        out = compute_features(spec, xs)
        assert all(v is None or math.isfinite(v) for v in out.values())
    assert compute_features(spec,[None]*5)['fase:longest_missing_block'] == 5


def test_distinctiveness_delayed_update_and_promotion():
    pool = fase.Pool(1, 1)
    pool.complete(row(1,0))
    receipt = pool.receipt({'trend': 0}, ['trend'], 1, ['base','a'])
    assert receipt[0]['contribution'] == 1
    assert pool.values[('s',row(1,0)['origin'])] == (0,0)  # reading cannot teach
    pool.complete(row(2,1,receipt=receipt))
    assert pool.values[('s',row(1,0)['origin'])] == (1,1)
    pool.complete(row(3,1))
    assert [r['origin'] for r in pool.long_term] == [row(1,0)['origin']]
    assert [r['origin'] for r in pool.recent] == [row(3,0)['origin']]


def test_partitions_use_invoked_provider_and_all_tied_winners():
    pool = fase.Pool(10,0)
    pool.complete(row(1,0))
    pool.complete(row(2,0,served='a'))
    pool.complete(row(3,0,losses={'base':1.,'a':1.}))
    receipt = pool.receipt({'trend':0}, ['trend'], 3, ['base','a'])
    assert [r['contribution'] for r in receipt] == [1,1,1]


def test_incremental_pool_matches_receipt_reconstruction():
    p = policy()
    index, rows = fase.ReplayIndex(p), []
    for i in range(1,12):
        q = {'trend': float(i%3)}
        expected = episodic_evidence(rows,p,'s',q)
        assert index.evidence('s',q,None) == expected
        r = row(i,float(i%3),receipt=expected.get('retention_receipt',[]))
        rows.append(r)
        index.add(r)


def test_retention_validation_rejects_inconsistent_retrieval():
    for opts in ({'retention':'fase'}, {'retention':'fase','distance':'fase','dedupe_seconds':1}):
        with pytest.raises(ForecastAdapterError):
            validate_memory(opts)


def test_replay_never_uses_unmatured_losses():
    p = policy()
    folds = [{'series_id':'s', 'origin':row(i,0)['origin'], 'target_time':row(i+3,0)['origin'],
              'actual': 10., 'points': {'base': 12., 'a': 10.}} for i in range(1,8)]
    history = {'s': [(row(i,0)['origin'], float(i%3)) for i in range(1,8)]}
    a = replay_router(folds,p,history,return_decisions=True)
    altered = [{**f,'actual': -999.} if i>=3 else f for i,f in enumerate(folds)]
    b = replay_router(altered,p,history,return_decisions=True)
    assert a['decisions'][:6] == b['decisions'][:6]


def test_accelerated_group_distances_match_pure_with_missing_and_constants():
    np = pytest.importorskip('numpy')
    from gnomon.episodic_memory_fast import bounded_distances
    names = ['fase:zero_ratio','fase:missing_ratio','fase:change_strength','context:mkt','context:vol']
    vectors = [dict(zip(names,v)) for v in ([0,None,1,2,3],[0,1,0,None,3],[0,2,3,1,3])]
    query = dict(zip(names,[1,None,2,3,3]))
    expected = [fase.distance(query,v,names,fase.scales(vectors,names)) for v in vectors]
    actual = bounded_distances([[v[n] for n in names] for v in vectors],[query[n] for n in names],names)
    assert list(actual) == pytest.approx(expected)


@pytest.mark.parametrize('retention', ['window','fase'])
def test_accelerated_replay_matches_pure(retention):
    pytest.importorskip('numpy')
    from datetime import datetime, timedelta, timezone
    start = datetime(2026,1,1,tzinfo=timezone.utc)
    stamp = lambda i: (start+timedelta(days=i)).isoformat()
    series = ['a','b']
    hist = {sid:[(stamp(i), math.sin(i*.23+j)*(.8+i*.003)) for i in range(85)] for j,sid in enumerate(series)}
    cov = {sid:{'mkt':[(stamp(i),math.cos(i*.17)) for i in range(85)]} for sid in series}
    folds = [{'series_id':sid,'origin':stamp(i),'target_time':stamp(i+2),'actual': 2+math.sin(i*.3),
              'points':{'base':2.2,'a':2+math.cos(i*.17)}} for i in range(35,80) for sid in series]
    p = validate_policy({'baseline':'base','candidates':['a'],'pool':{'series':series},'min_origins':2,
        'memory':{'profile':'fase','long_window':32,'distance':'fase','retention':retention,
                  'recent_capacity':5,'long_term_capacity':9,'k':5,'min_effective_n':2}})
    a = replay_router(folds,p,hist,covariates=cov,return_decisions=True,accelerate=False)
    b = replay_router(folds,p,hist,covariates=cov,return_decisions=True,accelerate=True)
    assert a['decisions'] == b['decisions']
    assert a['router_score'] == b['router_score']


from test_adaptive_router import live, _drive  # noqa: E402,F401


def test_live_retention_records_receipts_and_reads_beyond_lookback(live):
    ledger, session = live({'candidates':['last_value'],'baseline':'historical_mean',
        'min_origins':2,'recent_origins':3,'lookback_seconds':1,
        'memory':{'profile':'fase','short_window':2,'long_window':14,'distance':'fase','retention':'fase',
                  'k':3,'min_effective_n':1,'recent_capacity':2,'long_term_capacity':2}})
    replies = _drive(ledger,session,9)
    latest = replies[-1]['routing']
    assert latest['evidence_level']=='memory'
    decision = ledger.decision(latest['routing_decision_id'])
    selection = decision['inputs']['selection']
    assert selection['retention_pool']=={'recent':2,'long_term':2}
    assert len(selection['retention_receipt'])==3
    assert decision['inputs']['retention_policy']['memory']['retention']=='fase'
    from datetime import datetime, timedelta
    cutoff = datetime.fromisoformat(decision['inputs']['origin'])-timedelta(seconds=1)
    assert all(datetime.fromisoformat(n['origin']) < cutoff for n in latest['memory_neighbours'])
