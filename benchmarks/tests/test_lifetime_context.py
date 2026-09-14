from copy import deepcopy
import unittest
from benchmarks.ledger_optimization.lifetime_context import retrieve
from benchmarks.tests.test_context_ensemble import ContextRetrievalTest

class LifetimeMemoryTest(unittest.TestCase):
    def test_retains_all_visible_origins(self):
        current,rows=ContextRetrievalTest().fixture();result=retrieve(current,rows)
        self.assertEqual(len(result['candidates']),40);self.assertEqual(result['distinct_origins'],10)
        self.assertEqual(result['candidates_outside_recent_eight'],8)
        self.assertGreater(result['selected_outside_recent_eight'],0);self.assertEqual(len(result['selected']),16)
    def test_future_recording_still_excluded(self):
        current,rows=ContextRetrievalTest().fixture();baseline=retrieve(current,rows)
        extra=deepcopy(rows[0]);extra.update(series_id='electricity:future',outcome_recorded_at='2030-01-01T00:00:00+00:00')
        self.assertEqual(retrieve(current,rows+[extra]),baseline)
        current['actual']=[999.]*24
        for row in rows:row['scores']={'oracle':0}
        self.assertEqual(retrieve(current,rows),baseline)
    def test_cold_start_and_duplicate_safety(self):
        current,rows=ContextRetrievalTest().fixture()
        self.assertFalse(retrieve(current,rows[:8])['ready'])
        self.assertEqual(retrieve(current,[])['selected_outside_recent_eight'],0)
        with self.assertRaises(ValueError):retrieve(current,rows+[rows[0]])

if __name__=='__main__':unittest.main()
