import copy
import math
import unittest
from benchmarks.ledger_optimization.configuration_diagnostic import best,case,summary


def fixture():
    task={'task_id':'t','series_id':'electricity:s','round':0,'origin':'2020-01-01T00:00:00+00:00','actual':[2.,2.],
        'point':{'strong_block_cv':[2.,2.],'lifetime_ledger':[1.,1.]}}
    produced={str(i):[float(i)]*2 for i in range(7)}
    backtests=[{'config_id':str(i),'cv_rmsle':float(i+1)} for i in range(17)];backtests[6]['cv_rmsle']=.5
    decisions={a:{'backtests':copy.deepcopy(backtests),'selected':{'config_id':'6'},'point':produced['6']} for a in ('control','ledger')}
    return task,decisions,produced


class DiagnosticTest(unittest.TestCase):
    def test_hindsight_kept_separate_from_actual_selection(self):
        task,decisions,produced=fixture();r=case(task,decisions,produced)
        self.assertEqual(r['hindsight_choices']['union']['selected'],'2')
        self.assertEqual(r['arms']['ledger']['selected_config_id'],'6')
        self.assertEqual(r['arms']['ledger']['unobserved_production_configurations'],10)
        self.assertEqual(r['scores']['oracle_union'],0.)
        self.assertAlmostEqual(r['scores']['ledger'],math.log(7)-math.log(3))
        self.assertTrue(r['diagnostic_only']);self.assertTrue(r['uses_current_future_actuals'])
    def test_unexecuted_production_cannot_enter_ceiling(self):
        task,decisions,produced=fixture();produced['7']=[2.,2.]
        with self.assertRaises(ValueError):case(task,decisions,produced)
    def test_changed_selection_rejected(self):
        task,decisions,produced=fixture();decisions['ledger']['selected']['config_id']='5'
        with self.assertRaises(ValueError):case(task,decisions,produced)
    def test_exact_oracle_ties(self):
        self.assertEqual(best({'b':.1,'a':.1,'c':.2}),{'score':.1,'selected':'a','ties':['a','b']})
        with self.assertRaises(ValueError):best({'bad':float('nan')})


if __name__=='__main__':unittest.main()
