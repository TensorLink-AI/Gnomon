from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from benchmarks.ledger_optimization.training_inputs_102 import ALGORITHM_SHA256, compare_inputs, describe_inputs


def request():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    times = [(start+timedelta(days=n)).isoformat() for n in range(102)]
    names = ['onpromotion', 'dow_sin', 'dow_cos']
    return {'history': list(range(100)), 'timestamps': times[:100], 'horizon': 2,
            'future_timestamps': times[100:], 'cutoff': times[99], 'series_id': 's', 'unit': 'widgets',
            'past_covariates': [[0, 1, 0] for _ in range(100)], 'future_covariates': [[0, 1, 0]]*2,
            'past_covariate_names': names, 'future_covariate_names': names}


def compare(a, b, **changes):
    return compare_inputs(a, b, {'model': 'ridge', 'window': 90, 'lags': 7, 'alpha': 1., **changes},
                          algorithm_sha256=ALGORITHM_SHA256)


class TrainingInputsTests(unittest.TestCase):
    def test_prefix_outside_used_window_does_not_change_effective_inputs(self):
        a = request(); b = deepcopy(a)
        for k in ('history', 'timestamps', 'past_covariates'): b[k] = b[k][-90:]
        out = compare(a, b)
        self.assertTrue(out['same_effective_model_inputs'])
        self.assertEqual(out['left']['available_history_rows'], 100)
        self.assertEqual(out['right']['available_history_rows'], 90)
        self.assertEqual(out['left']['supervised_training_rows'], 83)
        self.assertFalse(out['recording_evidence_equivalence_asserted'])
        self.assertFalse(compare(a, b, window=100)['same_effective_model_inputs'])

    def test_identity_future_features_and_used_history_remain_distinct(self):
        for mutate in (lambda r:r.update(series_id='other'), lambda r:r.update(unit='kg'),
                       lambda r:r['future_covariates'][0].__setitem__(0, 1),
                       lambda r:r['history'].__setitem__(-1, 0),
                       lambda r:r['past_covariates'][-1].__setitem__(0, 1)):
            a = request(); b = deepcopy(a); mutate(b)
            with self.subTest(mutate=mutate): self.assertFalse(compare(a, b)['same_effective_model_inputs'])

    def test_numeric_timestamp_normalization_and_provenance_not_conflated(self):
        a = request(); b = deepcopy(a); b['history'] = [float(v) for v in b['history']]
        b['timestamps'] = [v.replace('+00:00', 'Z') for v in b['timestamps']]
        b['future_timestamps'] = [v.replace('+00:00', 'Z') for v in b['future_timestamps']]
        b['recorded_time_cutoff'] = '2026-01-01T00:00:00Z'
        out = compare(a, b)
        self.assertTrue(out['same_effective_model_inputs'])
        self.assertNotEqual(out['left']['temporal_provenance'], out['right']['temporal_provenance'])
        self.assertFalse(out['recording_evidence_equivalence_asserted'])

    def test_seasonal_uses_only_its_season_and_no_training_covariates(self):
        a = request(); b = deepcopy(a); b['history'][0] = 9000; b['future_covariates'] = []
        out = compare_inputs(a, b, {'model':'seasonal','season':7}, algorithm_sha256=ALGORITHM_SHA256)
        self.assertTrue(out['same_effective_model_inputs'])
        self.assertEqual(out['left']['effective_history_rows'], 7)
        self.assertEqual(out['left']['supervised_training_rows'], 0)

    def test_unsupported_algorithm_or_invalid_dimensions_reject(self):
        config = {'model':'ridge','window':90,'lags':7,'alpha':1.}
        with self.assertRaises(ValueError): describe_inputs(request(),config,algorithm_sha256='unknown')
        for field,value in [('timestamps',[]),('horizon',True),('cutoff','2025-01-01'),
                            ('past_covariates',[]),('history',[float('nan')]*100)]:
            r = request(); r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): compare(r,r)


if __name__ == '__main__':
    unittest.main()
