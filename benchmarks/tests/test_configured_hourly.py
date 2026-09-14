from datetime import datetime,timedelta,timezone
import math
import unittest
from benchmarks.ledger_optimization.configured_hourly import catalog,configuration,config_id,predict,validate_labels
from benchmarks.ledger_optimization.hourly_numerical import RECIPES,predict as original_predict


class ConfiguredHourlyTest(unittest.TestCase):
    def test_catalogue_identity_and_original_recipes(self):
        rows=catalog();self.assertEqual(len(rows),78);self.assertEqual(len({r['config_id'] for r in rows}),78)
        self.assertEqual([r['config'] for r in rows[:6]],[configuration(r) for r in RECIPES.values()])
        self.assertEqual(sum(r['config']['kind']=='ridge' for r in rows),63)
        self.assertEqual(sum(r['config']['kind']=='forest' for r in rows),12)
        for row in rows:self.assertEqual(row['config_id'],config_id(row['config']))
        self.assertEqual(config_id({'kind':'ridge','window':336,'lags':48,'alpha':10}),config_id({'alpha':10.,'lags':48.,'window':336.,'kind':'ridge'}))

    def test_rejected_schema_inputs(self):
        invalid=[[],{'kind':'bad'}, {'kind':'weekly_mean','alpha':10}, {'kind':'ridge','window':336,'lags':48},
            {'kind':'ridge','window':True,'lags':48,'alpha':10},{'kind':'ridge','window':336,'lags':48,'alpha':float('nan')},
            {'kind':'ridge','window':336,'lags':48,'alpha':2}]
        for value in invalid:
            with self.assertRaises(ValueError):configuration(value)

    def test_hour_phase_and_explicit_coordinate(self):
        start=datetime(2020,1,1,0,0,1,tzinfo=timezone.utc);labels=[start+timedelta(hours=i) for i in range(682)]
        validate_labels(labels[:658],labels[658:])
        with self.assertRaises(ValueError):validate_labels([t.replace(tzinfo=None) for t in labels[:658]],labels[658:])
        with self.assertRaises(ValueError):validate_labels(labels[:658],[t+timedelta(seconds=1) for t in labels[658:]])

    def test_all_original_predictions_unchanged(self):
        start=datetime(2020,1,1,0,0,1,tzinfo=timezone.utc)
        for n in (658,730):
            values=[20+4*math.sin(2*math.pi*i/24)+2*math.cos(2*math.pi*i/168)+i/1000 for i in range(n)]
            before=values.copy();labels=[start+timedelta(hours=i) for i in range(n+24)]
            for name,config in RECIPES.items():
                expected=original_predict(values,labels[:n],labels[n:],name)
                self.assertEqual(predict(values,labels[:n],labels[n:],config),expected)
                self.assertEqual(values,before)


if __name__=='__main__':unittest.main()
