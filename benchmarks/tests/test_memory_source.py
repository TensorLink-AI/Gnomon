import hashlib,json,unittest
from datetime import datetime
from benchmarks.ledger_optimization.memory_source import split

class MemorySourceTests(unittest.TestCase):
    def fixture(self):
        values=[float(i%7) for i in range(6130)]
        prefix=hashlib.sha256(json.dumps(values[1344:2074],separators=(',',':')).encode()).hexdigest()
        return values,prefix
    def test_main_stops_before_unused_final_origin(self):
        values,prefix=self.fixture();values[0]=None
        main,warm=split(values,datetime(2020,1,1),'sha','widgets',prefix)
        self.assertEqual(len(main['values']),4786);self.assertEqual(len(warm['values']),2074)
        self.assertEqual(warm['values'][-730:],main['values'][:730]);self.assertIsNone(warm['values'][0])
        self.assertEqual(len(main['values'][730+24*168:730+24*168+24]),24)
        self.assertEqual(main['values'][730+25*168:730+25*168+24],[])
        self.assertFalse(main['included_in_scored_denominator'])
    def test_missing_main_or_extra_tail_rejected(self):
        values,prefix=self.fixture()
        with self.assertRaises(ValueError):split(values+[1.],datetime(2020,1,1),'sha','widgets',prefix)
        values[-1]=None
        with self.assertRaises(ValueError):split(values,datetime(2020,1,1),'sha','widgets',prefix)

if __name__=='__main__':unittest.main()
