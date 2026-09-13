from datetime import datetime
import unittest
from benchmarks.ledger_optimization.pedestrian_timestamp_coverage import timestamp,START,END


class TimestampCoverageTest(unittest.TestCase):
    def row(self):
        return {'Year':'2019','Month':'November','Mdate':'1','Time':'17','Day':'Friday',
                'Date_Time':'November 01, 2019 05:00:00 PM'}

    def test_timestamp_requires_no_count_field(self):
        self.assertEqual(timestamp(self.row()),datetime(2019,11,1,17))
        self.assertEqual((END-START).total_seconds()/3600,4954)

    def test_redundant_fields_must_agree(self):
        for key,value in [('Time','18'),('Day','Monday')]:
            row=self.row();row[key]=value
            with self.assertRaises(ValueError):timestamp(row)


if __name__=='__main__':unittest.main()
