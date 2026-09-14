import copy
from datetime import datetime,timedelta,timezone
import unittest
from benchmarks.ledger_optimization.series_first_context import retrieve
from benchmarks.ledger_optimization.lifetime_context import retrieve as baseline


def fixture():
    current={'series_id':'electricity:self','domain':'electricity','origin':'2020-03-01T00:00:00+00:00','features':[0.]*12};rows=[]
    for i in range(24):
        at=datetime(2020,1,1,tzinfo=timezone.utc)+timedelta(days=i);close=(at+timedelta(days=1)).isoformat();same=i<8
        rows.append({'series_id':current['series_id'] if same else 'electricity:other','domain':'electricity','origin':at.isoformat(),'last_target':close,'outcome_recorded_at':close,'features':[100. if same else 0.]*12})
    return current,rows


class SeriesFirstTest(unittest.TestCase):
    def test_prioritizes_eight_own_records_and_borrows_exactly_eight(self):
        c,rows=fixture();r=retrieve(c,rows)
        self.assertEqual(r['selected_same_series'],8);self.assertEqual(len(r['selected']),16)
        self.assertTrue(all(x['series_id']==c['series_id'] for x in r['selected'][:8]))
        old=baseline(c,rows);self.assertEqual(r['candidates'],old['candidates']);self.assertEqual(r['scale'],old['scale'])
        self.assertEqual(r['selected_outside_recent_eight'],16)
    def test_enough_own_history_uses_no_other_series(self):
        c,rows=fixture()
        for row in rows[:18]:row['series_id']=c['series_id']
        r=retrieve(c,rows);self.assertEqual(r['selected_same_series'],16)
    def test_same_identity_does_not_bypass_recording_visibility(self):
        c,rows=fixture();late=copy.deepcopy(rows[0]);late.update(origin='2020-02-01T00:00:00+00:00',last_target='2020-02-02T00:00:00+00:00',outcome_recorded_at='2030-01-01T00:00:00+00:00',features=object())
        self.assertEqual(retrieve(c,rows),retrieve(c,rows+[late]))
    def test_cold_start_and_identity_requirement(self):
        c,rows=fixture();r=retrieve(c,rows[:8]);self.assertFalse(r['ready']);self.assertEqual(r['selected'],[])
        del c['series_id']
        with self.assertRaises(ValueError):retrieve(c,rows)


if __name__=='__main__':unittest.main()
