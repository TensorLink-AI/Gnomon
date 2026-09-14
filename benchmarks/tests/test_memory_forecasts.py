import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmarks.ledger_optimization import memory_forecasts as module

class HistoricalWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for name in ('raw','contexts','jobs'):(self.root/name).mkdir()
        self.old_spans,self.old_output=module._SPANS,module._OUTPUT
        module._OUTPUT=self.root;module._SPANS={'main':{'electricity:test':{'values':[1.]*754}}}
        self.task=('electricity:test',0,0,'main',None)
    def tearDown(self):
        module._SPANS,module._OUTPUT=self.old_spans,self.old_output;self.temp.cleanup()
    def test_worker_counts_and_preserves_evidence_role(self):
        def compute(span,index,history,charge):
            models=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
            for m in models:
                for _ in range(4):charge(m)
            return {'origin':'2020-01-01T00:00:00+00:00','last_target':'2020-01-02T00:00:00+00:00','history_indices':[0,730],'cv':dict.fromkeys(models,0.)}
        with patch.object(module,'compute_case',side_effect=compute):result=module.worker(self.task)
        self.assertEqual(result['status'],'complete');self.assertEqual(result['forecast_computations_started'],24);self.assertEqual(result['estimator_fits_started'],12)
        row=json.loads(next((self.root/'raw').glob('*.json')).read_text());self.assertFalse(row['included_in_scored_denominator'])
        context=json.loads(next((self.root/'contexts').glob('*.json')).read_text());self.assertEqual(len(context['features']),12);self.assertNotIn('actual',context)
    def test_failure_retains_started_costs(self):
        def failed(span,index,history,charge):charge('daily');charge('forest');raise ValueError('synthetic failure')
        with patch.object(module,'compute_case',side_effect=failed):result=module.worker(self.task)
        self.assertEqual(result['status'],'failed');self.assertEqual(result['forecast_computations_started'],2);self.assertEqual(result['estimator_fits_started'],1)
        self.assertEqual(list((self.root/'raw').glob('*.json')),[])
        self.assertEqual(json.loads(next((self.root/'jobs').glob('*.json')).read_text())['message'],'synthetic failure')

if __name__=='__main__':unittest.main()
