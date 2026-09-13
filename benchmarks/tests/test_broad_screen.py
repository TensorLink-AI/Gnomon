from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch
from benchmarks.ledger_optimization.broad_screen import select, compute_case, MODELS
from benchmarks.ledger_optimization.hourly_numerical import predict


class BroadScreenTest(unittest.TestCase):
    def test_visibility_and_cold_start(self):
        cv = {m: float(i) for i, m in enumerate(MODELS)}
        history = [{'origin': f'2020-01-0{i}T00:00:00Z',
                    'last_target': f'2020-01-0{i+1}T00:00:00Z',
                    'outcome_recorded_at': f'2020-01-0{i+1}T00:00:00Z',
                    'scores': {m: float(6-j) for j, m in enumerate(MODELS)}} for i in (1, 2, 3)]
        now = '2020-01-05T00:00:00Z'
        self.assertEqual(select(cv, history[:2], now)['past'], 'daily')
        expected = select(cv, history, now)
        self.assertEqual(expected['past'], 'forest')
        future = deepcopy(history[0]); future['outcome_recorded_at'] = '2021-01-01T00:00:00Z'
        self.assertEqual(select(cv, history+[future], now), expected)
        future['outcome_recorded_at'] = now; future['last_target'] = '2020-01-06T00:00:00Z'
        self.assertEqual(select(cv, history+[future], now), expected)
        with self.assertRaises(ValueError):
            select(cv, history+history, now)

    def test_backtest_prefix_and_production_targets_do_not_affect_choice(self):
        span = {'start_label': '2020-01-01T00:00:00', 'values': list(range(4954))}
        calls = []

        def forecast(history, labels, future, model):
            calls.append((len(history), history[-1], labels[-1], future[0]))
            return [float(history[-1])]*24

        with patch('benchmarks.ledger_optimization.broad_screen.numerical.predict', forecast):
            result = compute_case(span, 0, [])
            changed = deepcopy(span); changed['values'][730:] = [99999]*4224
            altered = compute_case(changed, 0, [])
        self.assertEqual(result['selection'], altered['selection'])
        self.assertEqual(result['point'], altered['point'])
        self.assertNotEqual(result['scores'], altered['scores'])
        for n, last, at, future in calls:
            self.assertIn(n, (658, 682, 706, 730))
            self.assertEqual(last, n-1)
            self.assertEqual(future-at, timedelta(hours=1))

    def test_hourly_baselines_at_shortest_cv_origin(self):
        n = 658
        labels = [datetime(2020, 1, 1, tzinfo=timezone.utc)+timedelta(hours=i) for i in range(n+24)]
        history = list(range(n))
        self.assertEqual(predict(history, labels[:n], labels[n:], 'daily'), list(range(n-24, n)))
        self.assertEqual(predict(history, labels[:n], labels[n:], 'weekly'), list(range(n-168, n-144)))
        self.assertEqual(predict(history, labels[:n], labels[n:], 'weekly_mean'),
                         [sum(n-168*k+i for k in (1, 2, 3))/3 for i in range(24)])

    def test_all_recipes_preserve_constant_level_at_each_origin(self):
        for n in (658, 682, 706, 730):
            labels = [datetime(2020, 1, 1, tzinfo=timezone.utc)+timedelta(hours=i) for i in range(n+24)]
            for model in MODELS:
                point = predict([37.0]*n, labels[:n], labels[n:], model)
                self.assertEqual(len(point), 24)
                for value in point:
                    self.assertAlmostEqual(value, 37.0, places=10)


if __name__ == '__main__':
    unittest.main()
