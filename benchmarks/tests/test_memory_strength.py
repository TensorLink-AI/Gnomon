import copy
from datetime import datetime,timedelta,timezone
import math
import unittest
from benchmarks.ledger_optimization.memory_strength import KEYS,blend_inputs,choose


def fixture():
    current={'series_id':'electricity:q','arm':'ledger','domain':'electricity','origin':'2020-03-01T00:00:00+00:00'};records=[]
    for i in range(16):
        origin=datetime(2020,1,1,tzinfo=timezone.utc)+timedelta(days=2*i);close=(origin+timedelta(days=1)).isoformat();tid=str(i)
        records.append({'task_id':tid,'series_id':'electricity:s','arm':'ledger','domain':'electricity','origin':origin.isoformat(),'last_target':close,'source_available_at':close,'recorded_at':close,'actual':[1.]*24,
            'candidates':{k:{'task_id':tid,'arm':'ledger','requested_strength':float(k),'execution_id':tid+':'+k,'forecast_recorded_at':origin.isoformat(),'point':[1.+abs(float(k)-.75)]*24} for k in KEYS}})
    return current,records,[{'series_id':r['series_id'],'origin':r['origin']} for r in records]


class MemoryStrengthTest(unittest.TestCase):
    def test_masses_and_zero_strength_remove_unused_evidence(self):
        current=[{'kind':'current'}]*3;past=[{'kind':'past'}]*16
        for strength in (0.,.25,.5,.75,1.):
            r=blend_inputs(current,past,strength);self.assertAlmostEqual(sum(r['masses']),1.)
            self.assertAlmostEqual(sum(m for p,m in zip(r['pairs'],r['masses']) if p['kind']=='past'),strength)
        self.assertEqual(blend_inputs(current,[],1.)['effective_strength'],0.)
        with self.assertRaises(ValueError):blend_inputs(current,past,True)
    def test_selects_actual_matched_risk_without_stored_score_trust(self):
        c,r,n=fixture();out=choose(c,r,n);self.assertTrue(out['ready']);self.assertEqual(out['requested_strength'],.75)
        self.assertAlmostEqual(out['candidate_score_means']['0'],math.log(2.75)-math.log(2))
    def test_exact_ties_preserve_half_strength(self):
        c,r,n=fixture()
        for row in r:
            for v in row['candidates'].values():v['point']=[1.]*24
        self.assertEqual(choose(c,r,n)['requested_strength'],.5)
    def test_unavailable_labels_are_not_read(self):
        c,r,n=fixture();baseline=choose(c,r,n)
        for kind in ('source_available_at','recorded_at','origin'):
            late=copy.deepcopy(r[0]);late.update(series_id='electricity:late',candidates=object(),actual=object());late[kind]='2030-01-01T00:00:00+00:00'
            self.assertEqual(choose(c,r+[late],n),baseline)
    def test_current_outcomes_and_other_arm_never_enter_selection(self):
        c,r,n=fixture();baseline=choose(c,r,n);c['actual']=object();other={**r[0],'arm':'control','candidates':object(),'actual':object()}
        self.assertEqual(choose(c,r+[other],n),baseline)
    def test_missing_record_does_not_score_a_smaller_subset(self):
        c,r,n=fixture();out=choose(c,r[:-1],n);self.assertFalse(out['ready']);self.assertIsNone(out['candidate_score_means']);self.assertEqual(len(out['missing_trial_records']),1)
    def test_missing_or_mismatched_candidate_is_rejected(self):
        c,r,n=fixture();del r[0]['candidates']['1']
        with self.assertRaises(ValueError):choose(c,r,n)
        c,r,n=fixture();r[0]['candidates']['1']['task_id']='wrong'
        with self.assertRaises(ValueError):choose(c,r,n)
    def test_late_candidate_recording_and_equivalent_origin_duplicates(self):
        c,r,n=fixture();r[0]['candidates']['1']['forecast_recorded_at']=r[0]['last_target']
        with self.assertRaises(ValueError):choose(c,r,n)
        c,r,n=fixture();duplicate=copy.deepcopy(r[0]);duplicate['origin']=duplicate['origin'].replace('+00:00','Z')
        with self.assertRaises(ValueError):choose(c,r+[duplicate],n)
    def test_duplicate_evidence_and_inconsistent_chronology_rejected(self):
        c,r,n=fixture()
        with self.assertRaises(ValueError):choose(c,r+[r[0]],n)
        r[0]['recorded_at']=r[0]['origin']
        with self.assertRaises(ValueError):choose(c,r,n)


if __name__=='__main__':unittest.main()
