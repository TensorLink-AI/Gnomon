import unittest
from copy import deepcopy
from benchmarks.ledger_optimization.breadth_screen import merge_contexts

class MemoryMergeTests(unittest.TestCase):
    def row(self,name):return {'series_id':'electricity:'+name,'domain':'electricity','origin':'2020-01-01T00:00:00+00:00',
        'last_target':'2020-01-02T00:00:00+00:00','outcome_recorded_at':'2020-01-02T00:00:00+00:00','features':[0.]*12}
    def test_only_predecision_fields_survive_merge(self):
        old,new=self.row('old'),self.row('new');old['scores']={'oracle':0.};new['actual']=[999.]*24
        keys={(r['series_id'],r['origin']) for r in (old,new)}
        out=merge_contexts([old],[new],keys,{'electricity:new'})
        self.assertEqual(len(out),2);self.assertNotIn('actual',out[1]);self.assertNotIn('scores',out[0])
    def test_unapproved_or_overlapping_or_missing_identity_rejected(self):
        old,new=self.row('old'),self.row('new');keys={(r['series_id'],r['origin']) for r in (old,new)}
        with self.assertRaises(ValueError):merge_contexts([old],[new],keys,{'electricity:other'})
        with self.assertRaises(ValueError):merge_contexts([old],[old],keys,{'electricity:old'})
        with self.assertRaises(ValueError):merge_contexts([old],[new],set(),{'electricity:new'})
        bad=deepcopy(new);bad['domain']='pedestrian'
        with self.assertRaises(ValueError):merge_contexts([old],[bad],keys,{'electricity:new'})

if __name__=='__main__':unittest.main()
