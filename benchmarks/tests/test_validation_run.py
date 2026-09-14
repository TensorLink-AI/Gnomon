"""Validation adapter tests use synthetic data, never validation targets."""
import unittest
from unittest.mock import patch
import numpy as np

from benchmarks.ledger_optimization.validation_run import resample_indices, uncertainty, fit_case, MODELS

class ValidationRunTests(unittest.TestCase):
    def test_paired_interval_constant_relative_gain(self):
        rows = [{'series_id':d+':'+str(s),'round':t,'scores':{'global_cv':2.+s+t/26,'block_cv':2.+s+t/26,'block_ledger':.8*(2.+s+t/26)}}
            for d in ('electricity','pedestrian') for s in range(8) for t in range(26)]
        result, si, oi, reductions = uncertainty(rows)
        self.assertEqual(si.shape,(10000,2,8));self.assertEqual(oi.shape,(10000,2,26))
        np.testing.assert_allclose(reductions,.2,atol=1e-14)
        for x in result['interval_95'].values():np.testing.assert_allclose(x,[.2,.2])
        si2,oi2 = resample_indices(np.random.default_rng(20260914))
        np.testing.assert_array_equal(si,si2);np.testing.assert_array_equal(oi,oi2)
        for i in range(0,24,4):np.testing.assert_array_equal((oi[...,i+1:i+4]-oi[...,i:i+3])%26,np.ones((10000,2,3)))

    def test_missing_scored_task_rejects(self):
        rows = [{'series_id':d+':'+str(s),'round':t,'scores':dict.fromkeys(('global_cv','block_cv','block_ledger'),1.)}
            for d in ('electricity','pedestrian') for s in range(8) for t in range(26)]
        with self.assertRaises(ValueError):uncertainty(rows[:-1])

    def test_cold_start_uses_identical_control_without_current_outcomes(self):
        point = {m:[1.]*24 for m in MODELS}
        source = {'origin':'2020-01-01T00:00:00+00:00','point':point,
            'folds':{m:[{'end':e,'point':[1.]*24,'actual':[1.]*24} for e in (658,682,706)] for m in MODELS}}
        current = {'origin':source['origin'],'domain':'electricity','features':[0.]*12}
        # No current actual/scores keys exist. No past evidence is available.
        def anchor(pairs,masses):return {'weights':[1/6]*6}
        with patch('benchmarks.ledger_optimization.validation_run.block_fit',return_value={'weights':[[1/6]*6]*4}) as mocked:
            out = fit_case(source,current,[],{},anchor)
        self.assertEqual(mocked.call_count,1)
        self.assertEqual(out['point']['block_cv'],out['point']['block_ledger'])
        self.assertEqual(out['fallback'],'insufficient_visible_context')
        self.assertEqual(out['weight_fits'],2)

if __name__ == '__main__':unittest.main()
