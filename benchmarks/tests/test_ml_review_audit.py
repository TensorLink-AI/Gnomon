from copy import deepcopy
import unittest
from benchmarks.ledger_optimization.ml_review_audit import coverage

class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.task={'origin':'2026-02-01T00:00:00Z','series_id':'s','unit':'units'}
        req={'series_id':'s','unit':'units','horizon':1,'cutoff':'2026-01-01T00:00:00Z',
             'future_timestamps':['2026-01-02T00:00:00Z']}
        self.events=[{'event':'result','kind':'forecast','task_origin':req['cutoff'],
                      'request':req,'point':[2],'config_id':c,'execution':{'execution_id':c}}
                     for c in ['baseline','ml']]
        self.events.append({'event':'matured','execution_id':'ml'})
        self.prior=[{'origin':req['cutoff'],'future_timestamps':req['future_timestamps'],
                     'outcome_recorded_at':'2026-01-03T00:00:00Z','actual':[3]}]
    def test_unselected_is_eligible_when_identical_horizon_matures(self):
        rows=coverage(self.events,self.prior,self.task)
        self.assertEqual(rows[0]['unscored_executions'],1)
        self.assertEqual(rows[0]['distinct_configs'],2)
    def test_late_recording_is_not_visible(self):
        self.prior[0]['outcome_recorded_at']='2026-03-01T00:00:00Z'
        self.assertEqual(coverage(self.events,self.prior,self.task),[])
    def test_wrong_unit_series_horizon_and_origin_are_excluded(self):
        for field,value in [('unit','other'),('series_id','other'),
                            ('future_timestamps',['2026-01-03T00:00:00Z']),
                            ('cutoff','2025-12-31T00:00:00Z')]:
            events=deepcopy(self.events);events[0]['request'][field]=value
            # The fixture deliberately shares a request; both rows must be excluded.
            self.assertEqual(coverage(events,self.prior,self.task),[])
    def test_retrospective_is_not_production(self):
        for e in self.events[:2]:e['kind']='backtest'
        self.assertEqual(coverage(self.events,self.prior,self.task),[])
    def test_ambiguous_outcomes_rejected(self):
        with self.assertRaises(AssertionError):coverage(self.events,self.prior*2,self.task)
    def test_horizon_not_yet_arrived(self):
        self.task['origin']='2026-01-01T12:00:00Z'
        self.prior[0]['outcome_recorded_at']='2026-01-01T12:00:00Z'
        self.assertEqual(coverage(self.events,self.prior,self.task),[])
