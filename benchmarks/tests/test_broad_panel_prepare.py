from datetime import datetime,timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from benchmarks.ledger_optimization.broad_panel_prepare import eligible,layout,numbers,partition,prepare,source_rows


class BroadPanelPrepareTest(unittest.TestCase):
    def test_later_values_do_not_change_eligibility_or_split(self):
        start=datetime(2014,1,1);end=start+timedelta(hours=4954)
        first=['1']*4954;changed=['1']*730+['invalid_future_value']*(4954-730)
        left=[];right=[]
        for i in range(40):
            a,reason=eligible(str(i),start,first,end);self.assertIsNone(reason)
            b,reason=eligible(str(i),start,changed,end);self.assertIsNone(reason)
            left.append(a);right.append(b)
        self.assertEqual(left,right)
        self.assertEqual(partition('electricity',left),partition('electricity',list(reversed(right))))
        dev,reserved=partition('electricity',left)
        self.assertEqual((len(dev),len(reserved)),(8,16))
        self.assertFalse({r['series_name'] for r in dev}&{r['series_name'] for r in reserved})

    def test_period_end_layout_has_exact_required_span(self):
        start=datetime(2014,1,1);end=start+timedelta(hours=4954)
        self.assertEqual(layout(start,end),(0,730,4954))
        self.assertEqual(start+timedelta(hours=730+25*168+24),end)

    def test_ineligible_prefix_and_duplicate_ids_rejected(self):
        start=datetime(2014,1,1);end=start+timedelta(hours=4954)
        self.assertEqual(eligible('a',start,['0']*4954,end)[1],'fewer_than_28_nonzero_initial_observations')
        self.assertEqual(eligible('a',start,['?']*4954,end)[1],'invalid_initial_history')
        with self.assertRaises(ValueError):partition('x',[{'series_name':'same'}]*24)
        with self.assertRaises(ValueError):partition('x',[{'series_name':str(i)} for i in range(23)])
        for values in (['nan'],['-1'],['?']):
            with self.assertRaises(ValueError):numbers(values)

    def test_existing_output_cannot_be_overwritten(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):prepare(d,d)

    def test_source_identity_and_header_validation(self):
        head='@attribute series_name string\n@attribute start_timestamp date\n@frequency hourly\n@data\n'
        with TemporaryDirectory() as d:
            path=Path(d)/'source.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('data.tsf',head+'a:2014-01-01 00-00-00:?,1,2\n')
            row=list(source_rows(path))[0]
            self.assertEqual(row[2],['?','1','2']) # Strings, not numeric future observations.
            with zipfile.ZipFile(path,'w') as z:z.writestr('data.tsf',head+('a:2014-01-01 00-00-00:1,2\n'*2))
            with self.assertRaises(ValueError):list(source_rows(path))


if __name__=='__main__':unittest.main()
