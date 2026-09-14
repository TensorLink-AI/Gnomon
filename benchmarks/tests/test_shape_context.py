from datetime import datetime,timedelta,timezone
import unittest
import numpy as np
from benchmarks.ledger_optimization.shape_context import profile,retrieve
from benchmarks.tests.test_series_first_context import fixture


class ShapeContextTest(unittest.TestCase):
    def test_complete_groups_and_observed_endpoint(self):
        values=[float(10+i%24) for i in range(730)];r=profile(values,'2020-02-01T00:00:01+00:00')
        self.assertEqual(len(r['vector']),31);self.assertEqual(r['hour_counts'],[28]*24);self.assertEqual(r['weekday_counts'],[96]*7)
        self.assertEqual(r['history_end'],'2020-01-31T23:00:01+00:00')
    def test_invariant_to_log_level_shift(self):
        x=[float(10+i%24) for i in range(730)];y=[(v+1)*3-1 for v in x]
        a=profile(x,'2020-02-01T00:00:00+00:00');b=profile(y,'2020-02-01T00:00:00+00:00');self.assertTrue(np.allclose(a['vector'],b['vector']))
    def test_late_profile_is_never_requested(self):
        c,rows=fixture();late={**rows[0],'origin':'2020-02-01T00:00:00+00:00','last_target':'2020-02-02T00:00:00+00:00','outcome_recorded_at':'2030-01-01T00:00:00+00:00'}
        seen=[]
        def get(r):
            seen.append((r['series_id'],r['origin']));self.assertNotEqual(r['origin'],late['origin'])
            return {'vector':[0.]*31,'input_sha256':r['origin']}
        result=retrieve(c,rows+[late],get);self.assertEqual(len(seen),25);self.assertEqual(len(result['selected']),16)
        for r in result['candidates']:self.assertAlmostEqual(r['distance'],r['coarse_distance']/12+r['shape_distance']/31)
    def test_invalid_history_or_naive_origin_rejected(self):
        for x,o in [([1.]*729,'2020-01-01T00:00:00+00:00'),([1.]*730,'2020-01-01'),([float('nan')]*730,'2020-01-01T00:00:00+00:00')]:
            with self.assertRaises(ValueError):profile(x,o)


if __name__=='__main__':unittest.main()
