import copy
from datetime import datetime,timedelta,timezone
import math
import unittest
from benchmarks.ledger_optimization.configuration_search import admit_backtest,posterior,suggest,select,checked_backtests,vector
from benchmarks.ledger_optimization.configured_hourly import catalog,revision


def fixture():
    configs=catalog();backtests=[{**r,'cv_rmsle':.1*(i+1)} for i,r in enumerate(configs[:6])]
    current={'series_id':'electricity:query','domain':'electricity','arm':'ledger','origin':'2020-02-01T00:00:00+00:00','features':[0.]*12}
    history=[];start=datetime(2020,1,1,tzinfo=timezone.utc)
    for i in range(16):
        origin=(start+timedelta(days=i)).isoformat()
        history.append({'series_id':'electricity:past','domain':'electricity','arm':'ledger','origin':origin,
            'source_available_at':origin,'recorded_at':origin,'evidence_kind':'executed_backtests','revision':revision(),
            'features':[i/20]*12,'backtests':copy.deepcopy(backtests)+[{**configs[6],'cv_rmsle':.2+i/100}],
            'synthetic_fixture':True})
    return current,backtests,history


class ConfigurationSearchTest(unittest.TestCase):
    def test_budget_reserves_final_fit(self):
        used=24
        for _ in range(11):admit_backtest(used);used+=3
        self.assertEqual(used,57)
        with self.assertRaises(ValueError):admit_backtest(used)
        self.assertEqual(used+1,58)
        for invalid in (-1,True,59):
            with self.assertRaises(ValueError):admit_backtest(invalid)

    def test_independent_two_point_kernel_posterior(self):
        c=[[1.]+[0.]*8,[0.,1.]+[0.]*7];x=[[0.]*12]*2;y=[.5,-1.];q=[.4,.6]
        mean,spread=posterior(c,x,y,q,[c[0]],[0.]*12,[1.]*12)
        off=math.exp(-4);a=1+.05/.4;b=1+.05/.6;det=a*b-off*off
        expected=(b*y[0]-off*y[1]+off*(-off*y[0]+a*y[1]))/det
        variance=1-(b-2*off*off+a*off*off)/det
        self.assertAlmostEqual(mean[0],expected,places=12);self.assertAlmostEqual(spread[0],math.sqrt(variance),places=12)

    def test_cold_start_identical_acquisition(self):
        current,backtests,history=fixture();empty=suggest(current,backtests,[]);insufficient=suggest(current,backtests,history[:8])
        self.assertEqual(empty['ranking'],insufficient['ranking'])
        self.assertEqual(empty['next_config_id'],insufficient['next_config_id'])
        self.assertEqual(insufficient['prior_episodes'],0)

    def test_unavailable_backtests_are_never_read(self):
        current,backtests,history=fixture();baseline=suggest(current,backtests,history)
        late=copy.deepcopy(history[0]);late.update(series_id='electricity:late',recorded_at='2030-01-01T00:00:00+00:00',backtests=object())
        simultaneous=copy.deepcopy(history[0]);simultaneous.update(series_id='electricity:same',origin=current['origin'],source_available_at=current['origin'],recorded_at=current['origin'],backtests=object())
        future=copy.deepcopy(history[0]);future.update(series_id='electricity:future',origin='2030-01-01T00:00:00+00:00',source_available_at='2030-01-01T00:00:00+00:00',recorded_at='2030-01-01T00:00:00+00:00',backtests=object())
        self.assertEqual(baseline,suggest(current,backtests,history+[late,simultaneous,future]))
        self.assertEqual(baseline['training_records'],6+16*7)

    def test_production_actuals_and_other_arm_records_do_not_enter_search(self):
        current,backtests,history=fixture();baseline=suggest(current,backtests,history)
        current['production_actuals']=object()
        for row in history:row['production_actuals']=object();row['production_scores']=object()
        other=copy.deepcopy(history[0]);other.update(arm='control',backtests=object())
        self.assertEqual(baseline,suggest(current,backtests,history+[other]))

    def test_revisions_duplicates_and_untested_configs(self):
        current,backtests,history=fixture()
        with self.assertRaises(ValueError):suggest(current,backtests,history+[history[0]])
        history[0]['revision']='different'
        with self.assertRaises(ValueError):suggest(current,backtests,history)
        with self.assertRaises(ValueError):checked_backtests(backtests+[backtests[0]])
        proposal=suggest(current,backtests)
        self.assertNotIn(proposal['next_config_id'],{r['config_id'] for r in backtests})
        self.assertEqual(len(proposal['ranking']),72)

    def test_final_selection_is_observed_cv_with_catalogue_ties(self):
        _,backtests,_=fixture();backtests[2]['cv_rmsle']=.01;backtests[3]['cv_rmsle']=.01
        self.assertEqual(select(list(reversed(backtests)))['config_id'],catalog()[2]['config_id'])

    def test_input_vectors_and_masses(self):
        for row in catalog():
            v=vector(row['config']);self.assertEqual(len(v),9);self.assertTrue(all(0<=x<=1 for x in v));self.assertEqual(sum(v[:4]),1.)
        with self.assertRaises(ValueError):posterior([[0.]*9],[[0.]*12],[1.],[0.],[[0.]*9],[0.]*12,[1.]*12)


if __name__=='__main__':unittest.main()
