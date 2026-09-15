from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math
import unittest
import json
from pathlib import Path
import tempfile

from benchmarks.ledger_optimization.m5_ml_search_screen_105 import CONFIGS,choose,requests_for_job,rmsle
from benchmarks.ledger_optimization.run_m5_ml_search_screen_105 import call_totals


def job():
    start=datetime(2020,1,1,tzinfo=timezone.utc)
    times=[(start+timedelta(days=i)).isoformat() for i in range(744)]
    r={'history':[float(i%7) for i in range(730)],'timestamps':times[:730],
       'future_timestamps':times[730:],'past_covariates':[[float(i%7)] for i in range(730)],
       'future_covariates':[[float(i%7)] for i in range(730,744)],
       'past_covariate_names':['weekday'],'future_covariate_names':['weekday'],
       'horizon':14,'series_id':'s','unit':'widgets','cutoff':times[729]}
    return {'series_id':'s','origin':times[729],'request':r,'actual':[999.]*14}


def history(count=4):
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    return [{'series_id':'s','origin':(start+timedelta(days=14*i)).isoformat(),
             'target_end':(start+timedelta(days=14*(i+1))).isoformat(),
             'recorded_at':(start+timedelta(days=14*(i+1))).isoformat(),
             'scores':[2.,1.]+[3.]*9} for i in range(count)]


class SearchScreenTests(unittest.TestCase):
    def test_production_targets_cannot_change_requests_or_cv_actuals(self):
        j=job();before=deepcopy(j);a=requests_for_job(j)
        j['actual']=[123456.]*14;b=requests_for_job(j)
        self.assertEqual(a,b)
        self.assertEqual(a[688]['actual'],before['request']['history'][688:702])
        self.assertEqual(a[716]['request']['future_timestamps'],before['request']['timestamps'][716:730])
        self.assertIsNone(a[730]['actual'])
        self.assertEqual(a[730]['request']['unit'],'widgets')
        self.assertEqual(len(a[688]['request']['history']),688)
        self.assertEqual(a[730]['request']['future_covariates'],before['request']['future_covariates'])

    def test_request_identity_and_order_rejected(self):
        for change in ('series','time'):
            j=job()
            if change=='series':j['series_id']='other'
            else:j['request']['timestamps'][0]=j['request']['timestamps'][1]
            with self.assertRaises(ValueError):requests_for_job(j)

    def test_cold_start_and_maturity_boundary(self):
        h=history();cv=[.1,.2]+[1.]*9
        self.assertEqual(choose(cv,h[:3],h[-1]['target_end'],series_id='s')['choices'],
                         dict.fromkeys(('current_cv','recent_history','lifetime_history','lifetime_support'),0))
        r=choose(cv,h,h[-1]['target_end'],series_id='s')
        self.assertEqual(r['matched_origins'],4)
        self.assertEqual(r['choices']['current_cv'],0)
        self.assertEqual(r['choices']['lifetime_support'],1)

    def test_source_recording_and_forecast_origin_boundaries(self):
        h=history();origin=h[-1]['target_end'];h[0]['recorded_at']='2099-01-01T00:00:00+00:00'
        r=choose([.1,.2]+[1.]*9,h,origin,series_id='s')
        self.assertEqual(r['matched_origins'],3);self.assertEqual(r['exclusions']['recording'],1)
        self.assertEqual(r['choices']['lifetime_history'],0)
        r=choose([.1,.2]+[1.]*9,history(),history()[2]['origin'],series_id='s')
        self.assertEqual(r['matched_origins'],2)
        self.assertEqual(r['exclusions']['source'],2)
        self.assertEqual(r['exclusions']['not_prior'],2)

    def test_history_order_input_immutability_and_ties(self):
        h=history();original=deepcopy(h);cv=[1.]*11
        a=choose(cv,h,h[-1]['target_end'],series_id='s')
        self.assertEqual(a,choose(cv,list(reversed(h)),h[-1]['target_end'],series_id='s'))
        self.assertEqual(h,original);self.assertEqual(a['choices']['current_cv'],0)
        self.assertEqual(len(CONFIGS),11)

    def test_support_requires_paired_wins_not_only_better_mean(self):
        h=history()
        for i,r in enumerate(h):r['scores']=[10. if i==0 else 1.,0. if i==0 else 2.]+[20.]*9
        r=choose([.1,.2]+[1.]*9,h,h[-1]['target_end'],series_id='s')
        self.assertEqual(r['choices']['lifetime_history'],1)
        self.assertEqual(r['choices']['lifetime_support'],0)

    def test_incomplete_or_cross_series_history_rejected(self):
        h=history()
        for items in (h+[h[0]], [{**h[0],'series_id':'other'}], [{**h[0],'scores':[1.]}]):
            with self.assertRaises(ValueError):choose([1.]*11,items,h[-1]['target_end'],series_id='s')
        with self.assertRaises(ValueError):choose([float('nan')]*11,h,h[-1]['target_end'],series_id='s')

    def test_metric(self):
        self.assertAlmostEqual(rmsle([3.]*14,[1.]*14),math.log(2))
        self.assertEqual(rmsle([-5.]*14,[0.]*14),0.)
        for p,a in [([1.]*13,[1.]*14),([True]*14,[1.]*14),([1.]*14,[-1.]*14)]:
            with self.assertRaises(ValueError):rmsle(p,a)

    def test_interrupted_worker_attempts_are_not_lost_from_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'series/s/round-00/calls.jsonl';p.parent.mkdir(parents=True)
            event={'event':'attempt','configuration_index':0,'end':688,'config':{'model':'ridge'}}
            p.write_text(json.dumps(event)+'\n')
            result=call_totals(directory)
            self.assertEqual(result['numerical_attempts'],1)
            self.assertEqual(len(result['unmatched_attempts']),1)
            self.assertIsNone(result['estimator_fits'])
            p.write_text(json.dumps(event)+'\n'+json.dumps({**event,'event':'result'})+'\n')
            result=call_totals(directory)
            self.assertEqual(result['estimator_fits'],1)
            self.assertEqual(result['numerical_successes'],1)


if __name__=='__main__':unittest.main()
