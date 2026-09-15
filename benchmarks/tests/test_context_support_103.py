from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from benchmarks.ledger_optimization.context_support_103 import context, cohorts


def request(day=100, values=None, promotion=0):
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {'history': values or [10] * 28,
            'timestamps': [(origin - timedelta(days=27-i)).isoformat() for i in range(28)],
            'cutoff': origin.isoformat(), 'future_timestamps': [(origin + timedelta(days=1)).isoformat()],
            'horizon': 1, 'series_id': 's', 'unit': 'widgets',
            'future_covariate_names': ['onpromotion'], 'future_covariates': [[promotion]]}


def episodes():
    return [{'origin': request(day)['cutoff'], 'source_as_of': request(100)['cutoff'],
             'recorded_as_of': request(100)['cutoff'],
             'models': [{'provider': p, 'revision': 'v1', 'execution_id': f'{day}-{p}',
                         'request': request(day, promotion=int(day < 5)),
                         'rmsle': score} for p, score in [('a', 1), ('b', 2)]]}
            for day in range(1, 7)]


class ContextSupportTests(unittest.TestCase):
    def run_cohorts(self, rows):
        return cohorts(request(), {'a': 'v1', 'b': 'v1'}, rows, {'a': 2, 'b': 1})

    def test_count_gate_not_score_selection(self):
        rows = episodes()
        first = self.run_cohorts(rows)
        self.assertEqual([r['matched_origins'] for r in first['cohorts']], [2, 2, 6, 6, 6])
        self.assertEqual(first['selected_filter_index'], 2)
        self.assertTrue(first['cohorts'][2]['disagrees_with_current_cv'])
        for row in rows:
            row['models'][0]['rmsle'], row['models'][1]['rmsle'] = 3, 0
        second = self.run_cohorts(rows)
        self.assertEqual(first['selected_filter_index'], second['selected_filter_index'])
        self.assertEqual(second['cohorts'][2]['lowest_mean_providers'], ['b'])
        self.assertFalse(second['cohorts'][2]['disagrees_with_current_cv'])

    def test_empty_and_exact_ties(self):
        empty = self.run_cohorts([])
        self.assertIsNone(empty['selected_filter_index'])
        self.assertTrue(all(r['scores'] == {} for r in empty['cohorts']))
        rows = episodes()
        for r in rows:
            for m in r['models']:
                m['rmsle'] = 0
        tied = self.run_cohorts(rows)
        self.assertEqual(tied['cohorts'][2]['lowest_mean_providers'], ['a', 'b'])

    def test_origin_maturity_and_visible_cutoffs(self):
        for mutation in ('future', 'recording', 'source', 'duplicate', 'timezone'):
            rows = episodes()
            if mutation == 'future':
                for m in rows[0]['models']:
                    m['request']['future_timestamps'] = request(101)['future_timestamps']
            elif mutation in ('recording', 'source'):
                rows[0]['recorded_as_of' if mutation == 'recording' else 'source_as_of'] = request(101)['cutoff']
            elif mutation == 'duplicate':
                rows.append(deepcopy(rows[0]))
            else:
                rows[0]['origin'] = '2020-01-02'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_cohorts(rows)

    def test_identity_and_context_mismatch(self):
        for field, value in [('unit', 'kg'), ('series_id', 'other'), ('horizon', 2),
                             ('history', [20] * 28), ('future_covariates', [[0]])]:
            rows = episodes()
            rows[0]['models'][0]['request'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_cohorts(rows)
        rows = episodes()
        rows[0]['models'][0]['revision'] = 'v2'
        with self.assertRaises(ValueError):
            self.run_cohorts(rows)

    def test_observed_labels_and_unknown_promotion(self):
        r = request(values=[0] * 28)
        self.assertEqual(context(r), {'sparsity': 'high', 'trend': 'stable', 'volatility': 'stable', 'promotion': 'none'})
        r['future_covariate_names'] = []
        r['future_covariates'] = []
        self.assertEqual(context(r)['promotion'], 'unknown')
        self.assertEqual(context(request(values=[10]*14 + [20]*14))['trend'], 'rising')
        self.assertEqual(context(request(values=[20]*14 + [10]*14))['trend'], 'falling')
        for value in (True, float('nan'), -1):
            r = request(); r['history'][-1] = value
            with self.assertRaises(ValueError):
                context(r)

    def test_does_not_mutate_inputs(self):
        rows = episodes(); before = deepcopy(rows)
        self.run_cohorts(rows)
        self.assertEqual(rows, before)


if __name__ == '__main__':
    unittest.main()
