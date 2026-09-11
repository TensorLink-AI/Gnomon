from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json

import pytest

from benchmarks.ledger_optimization.m5_candidates import PROVIDERS, compute_case, load_panel, metric, request
from benchmarks.ledger_optimization.audits.m5_candidates_015 import check_request, score


def rows():
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    return [{'unique_id': 'synthetic', 'ds': (start + timedelta(days=i)).isoformat(),
             'source_date': (start + timedelta(days=i - 1)).date().isoformat(), 'y': (i % 7) + 1,
             'onpromotion': 0} for i in range(730)]


def test_request_never_reads_future_targets_and_preserves_identity():
    data = rows()
    expected = request(data, 365)
    for row in data[366:]:
        row['y'] = object()
        row['onpromotion'] = object()
    assert request(data, 365) == expected
    assert expected.series_id == 'synthetic' and expected.unit == 'unit_sales'
    assert len(expected.history) == 366 and len(expected.future_timestamps) == 14
    assert expected.cutoff == expected.known_time_cutoff == expected.recorded_time_cutoff
    assert expected.timestamps[-1] == expected.cutoff < expected.future_timestamps[0]


def test_future_sales_change_scores_not_forecasts_or_cv_cards():
    data = rows()
    calls = []

    def predict(provider, req):
        calls.append((provider, req))
        return tuple(req.history[-7:]) * 2, {'fallback_used': False}

    original = deepcopy(data)
    first = compute_case(data, 0, predict)
    assert data == original
    assert len(calls) == len(PROVIDERS) * 3
    assert all(req.future_timestamps[-1] <= first['origin'] for _, req in calls if req.cutoff != first['origin'])
    for row in data[366:380]:
        row['y'] += 100
    second = compute_case(data, 0, predict)
    assert first['request'] == second['request']
    assert first['predictions'] == second['predictions']
    assert first['current_card'] == second['current_card']
    assert first['scores'] != second['scores']


def test_bad_provider_output_is_not_silently_excluded():
    def wrong(provider, req):
        return (1,), {'fallback_used': False}

    with pytest.raises(ValueError, match='complete'):
        compute_case(rows(), 0, wrong)
    with pytest.raises(ValueError, match='finite'):
        metric([float('nan')], [1])
    with pytest.raises(ValueError, match='nonnegative'):
        metric([1], [-1])


def test_unregistered_manifest_rejected_before_data_read(tmp_path):
    p = tmp_path / 'manifest.json'
    p.write_text(json.dumps({'development': {'path': 'reserved-secret-data'}}))
    with pytest.raises(ValueError, match='registered'):
        load_panel(p)


def test_known_seasonal_and_metric_arithmetic():
    req = request(rows(), 365)
    point = req.history[-7:] * 2
    actual = [r['y'] for r in rows()[366:380]]
    assert point == tuple(actual)
    assert metric(point, actual) == {'mae': 0, 'rmsle': 0}
    assert metric([-1, 3], [0, 3])['rmsle'] == 0  # pinned clipping convention


def test_independent_audit_rejects_changed_cutoff_and_identity():
    data = rows()
    req = json.loads(json.dumps(asdict(request(data, 365))))
    check_request(req, data, 365)
    for field, value in [('series_id', 'unrelated'), ('recorded_time_cutoff', req['future_timestamps'][0])]:
        altered = dict(req, **{field: value})
        with pytest.raises(AssertionError):
            check_request(altered, data, 365)
    with pytest.raises(AssertionError):
        score([-1] * 14, [1] * 14)  # pinned provider must have clipped before returning
