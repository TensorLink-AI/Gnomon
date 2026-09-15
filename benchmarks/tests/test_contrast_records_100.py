from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import unittest

from benchmarks.ledger_optimization.contrast_records_100 import current_runs


def fixture():
    task = {'series_id': 'sales', 'unit': 'widgets', 'horizon': 2, 'origin': '2026-03-01T00:00:00+00:00'}
    requests = []
    rows = []
    for i in range(3):
        origin = datetime(2026, 2, 1+i*7, tzinfo=timezone.utc)
        request = {'cutoff': origin.isoformat(), 'series_id': task['series_id'], 'unit': task['unit'],
                   'horizon': 2, 'history': [1.], 'timestamps': [origin.isoformat()],
                   'future_timestamps': [(origin+timedelta(days=d)).isoformat() for d in (1, 2)]}
        requests.append(request)
        for model in ('left', 'right'):
            config = {'model': model}
            cid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
            point = [1., 1.] if model == 'left' else [2., 2.]
            actual = [2., 2.]
            score = abs(math.log1p(point[0])-math.log1p(actual[0]))
            rows.append({'task_origin': task['origin'], 'event': 'result', 'kind': 'backtest',
                         'config': config, 'config_id': cid, 'request': deepcopy(request),
                         'point': point, 'actual': actual, 'metrics': {'n': 2, 'rmsle': score},
                         'execution': {'status': 'ok', 'execution_id': f'{model}-{i}',
                                       'provider': model+'_'+cid, 'revision': 'ml-lab-v1:'+cid,
                                       'result': {'point': point, 'timestamps': request['future_timestamps'],
                                                  'series_id': task['series_id'], 'unit': task['unit']}}})
    return task, rows, requests


def render(t, r, q, arm='gnomon'):
    return current_runs(t, r, q, deepcopy, arm=arm, expected_actuals={x['cutoff']: [2., 2.] for x in q})


class ContrastRecordTests(unittest.TestCase):
    def test_scores_are_recomputed_without_modifying_records(self):
        t, r, q = fixture(); before = deepcopy((t, r, q))
        result = render(t, r, q)
        self.assertEqual(result['verified_fold_executions'], 6)
        self.assertEqual(len(result['runs']), 2)
        self.assertEqual(result['excluded'], [])
        self.assertEqual(result['provider_calls'], 0)
        self.assertEqual((t, r, q), before)

    def test_production_and_other_task_values_are_not_accessed(self):
        t, r, q = fixture()
        r += [{'task_origin': t['origin'], 'event': 'result', 'kind': 'forecast'},
              {'task_origin': '2027-01-01T00:00:00+00:00', 'event': 'result', 'kind': 'backtest'},
              {'task_origin': t['origin'], 'event': 'attempt', 'kind': 'backtest'}]
        self.assertEqual(render(t, r, q)['verified_fold_executions'], 6)

    def test_plain_identity_comes_from_exact_host_request_not_typed_result(self):
        t, r, q = fixture()
        for row in r:
            row['execution'].pop('status')
            row['execution']['result'] = {'point': row['point']}
        self.assertEqual(render(t, r, q, arm='plain')['verified_fold_executions'], 6)
        with self.assertRaises(ValueError): render(t, r, q, arm='gnomon')
        with self.assertRaises(ValueError): render(t, r, q, arm='unknown')
        r[0]['execution']['status'] = 'error'
        with self.assertRaises(ValueError): render(t, r, q, arm='plain')

    def test_internally_consistent_wrong_actuals_still_rejected(self):
        t, r, q = fixture()
        r[0]['actual'] = [1., 1.]
        r[0]['metrics']['rmsle'] = 0.
        with self.assertRaises(ValueError): render(t, r, q)

    def test_unitless_typed_result_still_requires_explicit_identity_fields(self):
        t, r, q = fixture(); t['unit'] = None
        for request in q: request['unit'] = None
        for row in r:
            row['request']['unit'] = None
            row['execution']['result']['unit'] = None
        self.assertEqual(render(t, r, q)['verified_fold_executions'], 6)
        r[0]['execution']['result'].pop('unit')
        with self.assertRaises(ValueError): render(t, r, q)

    def test_incomplete_run_is_explicitly_excluded_not_filled(self):
        t, r, q = fixture(); r.pop()
        result = render(t, r, q)
        self.assertEqual(len(result['runs']), 1)
        self.assertEqual(result['excluded'][0]['completed_folds'], 2)
        self.assertEqual(result['excluded'][0]['missing_origins'], [q[-1]['cutoff']])

    def test_duplicate_origins_and_execution_ids_fail_closed(self):
        for same_id in (True, False):
            t, r, q = fixture(); duplicate = deepcopy(r[0])
            if not same_id: duplicate['execution']['execution_id'] = 'different'
            r.append(duplicate)
            with self.subTest(same_id=same_id), self.assertRaises(ValueError): render(t, r, q)

    def test_identity_and_score_tampering_rejected(self):
        changes = [lambda r: r['request'].update(unit='kg'),
                   lambda r: r['request']['history'].append(10.),
                   lambda r: r.update(config_id='invented'),
                   lambda r: r['execution'].update(revision='other'),
                   lambda r: r['execution'].update(status='error'),
                   lambda r: r['execution']['result'].update(series_id='other'),
                   lambda r: r['execution']['result'].update(timestamps=[]),
                   lambda r: r['metrics'].update(rmsle=.9),
                   lambda r: r['metrics'].update(n=True),
                   lambda r: r['actual'].__setitem__(0, -1.),
                   lambda r: r['point'].__setitem__(0, float('nan'))]
        for change in changes:
            t, r, q = fixture(); change(r[0])
            with self.subTest(change=change), self.assertRaises(ValueError): render(t, r, q)

    def test_expected_history_endpoints_and_completed_targets_are_required(self):
        changes = [lambda q: q.pop(), lambda q: q.append(deepcopy(q[0])),
                   lambda q: q[0].update(cutoff='2026-01-31T00:00:00+00:00'),
                   lambda q: q[0].update(unit='kg'),
                   lambda q: q[0]['future_timestamps'].__setitem__(1, '2026-04-01T00:00:00+00:00')]
        for change in changes:
            t, r, q = fixture(); change(q)
            with self.subTest(change=change), self.assertRaises(ValueError): render(t, r, q)


if __name__ == '__main__': unittest.main()
