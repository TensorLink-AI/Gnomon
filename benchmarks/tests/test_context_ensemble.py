import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from benchmarks.ledger_optimization.context_ensemble import retrieve


class ContextRetrievalTest(unittest.TestCase):
    def fixture(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        rows = []
        for t in range(10):
            for s in range(4):
                rows.append({'series_id': f'electricity:{s}', 'domain': 'electricity',
                    'origin': (start+timedelta(days=t*7)).isoformat(),
                    'last_target': (start+timedelta(days=t*7+1)).isoformat(),
                    'outcome_recorded_at': (start+timedelta(days=t*7+1)).isoformat(),
                    'features': [float(s)]*12})
        current = {'domain': 'electricity', 'origin': (start+timedelta(days=70)).isoformat(), 'features': [0.]*12}
        return current, rows

    def test_visible_eight_origin_pool_and_nearest(self):
        current, rows = self.fixture(); result = retrieve(current, rows)
        self.assertEqual(len(result['candidates']), 32)
        self.assertEqual(len(result['selected']), 16)
        self.assertEqual({r['series_id'] for r in result['selected']}, {'electricity:0', 'electricity:1'})

    def test_recording_closure_origin_and_domain_boundaries(self):
        current, rows = self.fixture(); baseline = retrieve(current, rows)
        extras = []
        for field in ('origin', 'last_target', 'outcome_recorded_at'):
            row = deepcopy(rows[-1]); row['series_id'] = field
            row[field] = '2021-01-01T00:00:00+00:00'; extras.append(row)
        row = deepcopy(rows[-1]); row['origin'] = current['origin']; extras.append(row)
        row = deepcopy(rows[-1]); row['domain'] = 'pedestrian'; extras.append(row)
        self.assertEqual(retrieve(current, rows+extras), baseline)

    def test_current_and_future_outcomes_cannot_affect_retrieval(self):
        current, rows = self.fixture(); baseline = retrieve(current, rows)
        current['actual'] = [-999.]*24
        for r in rows:r.update(actual=[999.]*24, scores={'future_oracle': 0.})
        self.assertEqual(retrieve(current, rows), baseline)

    def test_duplicates_rejected_and_cold_start_disclosed(self):
        current, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            retrieve(current, rows+[rows[-1]])
        self.assertFalse(retrieve(current, rows[:8])['ready'])
        self.assertEqual(retrieve(current, [])['selected'], [])


if __name__ == '__main__':unittest.main()
