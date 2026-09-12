from copy import deepcopy
from datetime import datetime,timedelta,timezone
import math
import unittest

from benchmarks.ledger_optimization.ml_review_coverage import check_pair


class ReviewCoverageTest(unittest.TestCase):
    def setUp(self):
        self.origin=datetime(2026,1,1,tzinfo=timezone.utc)
        self.at=self.origin+timedelta(days=14)
        times=[(self.origin+timedelta(days=i)).isoformat() for i in range(1,15)]
        self.events=[]
        for cid,point in [('a',1),('b',2)]:
            self.events.extend([
                {'event':'result','kind':'forecast','execution':{'execution_id':cid},
                 'config_id':cid,'point':[point]*14,
                 'request':{'cutoff':self.origin.isoformat(),'future_timestamps':times}},
                {'event':'matured','execution_id':cid,'point':[point]*14,'actual':[1]*14,
                 'task_origin':self.at.isoformat(),'outcome_recorded_at':self.at.isoformat()}])
        self.pair={'n':1,'matched_origins':[self.origin.isoformat()],
                   'left':'a','right':'b','left_rmsle':0,'right_rmsle':math.log(1.5)}

    def test_independent_known_values(self):
        check_pair(self.pair,self.events,self.at)

    def test_wrong_reported_metric(self):
        self.pair['right_rmsle']=0
        with self.assertRaises(AssertionError):check_pair(self.pair,self.events,self.at)

    def test_wrong_origin(self):
        self.pair['matched_origins']=['2025-12-31T00:00:00+00:00']
        with self.assertRaises(KeyError):check_pair(self.pair,self.events,self.at)

    def test_future_recording_excluded(self):
        self.events[1]['outcome_recorded_at']=(self.at+timedelta(days=1)).isoformat()
        with self.assertRaises(AssertionError):check_pair(self.pair,self.events,self.at)

    def test_unmatured_target_excluded(self):
        self.events[0]['request']['future_timestamps']=deepcopy(self.events[0]['request']['future_timestamps'])
        self.events[0]['request']['future_timestamps'][-1]=(self.at+timedelta(days=1)).isoformat()
        with self.assertRaises(AssertionError):check_pair(self.pair,self.events,self.at)

    def test_duplicate_matched_origin(self):
        self.pair['matched_origins']*=2;self.pair['n']=2
        with self.assertRaises(AssertionError):check_pair(self.pair,self.events,self.at)


if __name__=='__main__':unittest.main()
