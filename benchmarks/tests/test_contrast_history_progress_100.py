from copy import deepcopy
import unittest

from benchmarks.ledger_optimization.contrast_history_progress_100 import summarize, ARMS


def rows():
    return [{'arm': a, 'series_id': 's', 'round': n, 'prior_outcomes': n,
             'origin': str(n), 'rmsle': score, 'valid': a != 'ledger',
             'workflow_complete': a != 'ledger', 'fallback_used': a == 'ledger'}
            for n, scores in enumerate(((2., 2., 1.), (2., 2., 3.)))
            for a, score in zip(ARMS, scores)]


class HistoryProgressTests(unittest.TestCase):
    def test_all_case_mean_and_failures_stay_in_primary(self):
        out = summarize(rows())
        self.assertEqual(out['primary_all_matched']['matched_cases'], 2)
        self.assertEqual(out['primary_all_matched']['mean_rmsle']['ledger'], 2.)
        self.assertEqual(out['primary_all_matched']['ledger_relative_improvement'], 0.)
        self.assertEqual(out['primary_all_matched']['fallbacks']['ledger'], 2)
        self.assertEqual([r['ledger_relative_improvement'] for r in out['by_prior_outcomes']], [.5, -.5])
        self.assertIsNone(out['cold_mature_threshold'])
        self.assertFalse(out['threshold_predeclared'])
        self.assertFalse(out['target_established'])

    def test_incomplete_group_is_pending_not_an_unequal_arm_comparison(self):
        out = summarize(rows()[:-1])
        self.assertEqual(out['observed_sessions'], 5)
        self.assertEqual(out['primary_all_matched']['matched_cases'], 1)
        self.assertEqual(out['pending'], [{'series_id': 's', 'round': 1, 'missing_arms': ['ledger']}])

    def test_cumulative_results_are_per_series_and_order_invariant(self):
        data = rows()
        extra = deepcopy(data[:3])
        for row in extra: row.update(series_id='other', rmsle=0.)
        data += extra
        out = summarize(data)
        self.assertEqual(out, summarize(list(reversed(data))))
        self.assertEqual(out['per_series_cumulative']['s'][-1]['ledger_relative_improvement'], 0.)
        self.assertIsNone(out['per_series_cumulative']['other'][0]['ledger_relative_improvement'])
        self.assertEqual(out['per_series_cumulative']['other'][0]['relative_status'], 'undefined_zero_control_mean')

    def test_bad_or_duplicate_rows_and_origin_mismatch_reject(self):
        for field, value in (('prior_outcomes', 2), ('round', True), ('rmsle', float('nan')),
                             ('rmsle', -1.), ('valid', 1), ('arm', '1.1.9'), ('origin', 'different')):
            data = rows(); data[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): summarize(data)
        data = rows()
        with self.assertRaises(ValueError): summarize(data+data[:1])
        self.assertEqual(summarize([])['primary_all_matched']['relative_status'], 'no_matched_cases')

    def test_absent_origins_do_not_form_a_matched_task(self):
        for value in (None, '', ' '):
            data = rows()
            for row in data: row['origin'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): summarize(data)
        data = rows()[:1]; data[0].pop('origin')
        with self.assertRaises(ValueError): summarize(data)


if __name__ == '__main__':
    unittest.main()
