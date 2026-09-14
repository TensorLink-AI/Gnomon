import math
import unittest
from copy import deepcopy
from benchmarks.ledger_optimization.error_profile import profile,retrieve_profile,MODELS
from benchmarks.tests.test_context_ensemble import ContextRetrievalTest

class ErrorProfileTest(unittest.TestCase):
    def folds(self):return {m:[{'end':end,'point':[3.]*6+[1.]*18,'actual':[1.]*24} for end in (658,682,706)] for m in MODELS}
    def test_signed_shape_and_rms(self):
        values=profile(self.folds());self.assertEqual(len(values),48)
        for i in range(6):
            self.assertAlmostEqual(values[i*8],math.log(2));self.assertAlmostEqual(values[i*8+1],math.log(2))
            self.assertEqual(values[i*8+2:i*8+8],[0.]*6)
    def test_unmatched_actuals_rejected(self):
        folds=self.folds();folds[MODELS[-1]][0]['actual'][0]=999
        with self.assertRaises(ValueError):profile(folds)
    def fixture(self):
        current,rows=ContextRetrievalTest().fixture();current['error_profile']=[0.]*48
        for r in rows:r['error_profile']=[float(r['series_id'].split(':')[-1])]*48
        return current,rows
    def test_visibility_and_production_outcome_independence(self):
        current,rows=self.fixture();expected=retrieve_profile(current,rows)
        current['actual']=[999.]*24
        for r in rows:r.update(actual=[999.]*24,scores={'oracle':0.})
        future=deepcopy(rows[0]);future.update(series_id='electricity:future',origin='2030-01-01T00:00:00+00:00',error_profile=[float('nan')]*48)
        self.assertEqual(retrieve_profile(current,rows+[future]),expected)
        self.assertEqual(len(expected['selected']),16)
    def test_family_sizes_do_not_multiply_relative_weight(self):
        current,rows=self.fixture();r=retrieve_profile(current,rows)
        # Repeated scalar features across each family: components must match.
        for c in r['candidates']:self.assertAlmostEqual(c['legacy_component'],c['profile_component'])
        self.assertFalse(retrieve_profile(current,rows[:4])['ready'])

if __name__=='__main__':unittest.main()
