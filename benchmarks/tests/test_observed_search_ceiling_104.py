import copy
import math
import unittest

from benchmarks.ledger_optimization.observed_search_ceiling_104 import rmsle, summarize


def rows():
    return [{'arm': a, 'series_id': 's', 'round': 0, 'origin': '2026-01-01',
             'actual': [1., 3.], 'fallback_point': [0., 0.], 'selected_point': [2., 2.],
             'valid': True, 'workflow_complete': True,
             'forecasts': [{'execution_id': a+'1', 'config_id': 'c1', 'point': [2., 2.]},
                           {'execution_id': a+'2', 'config_id': 'c2', 'point': [1., 3.]}]}
            for a in ('plain', 'gnomon', 'ledger')]


class SearchCeilingTests(unittest.TestCase):
    def test_independent_score_and_hindsight(self):
        data = rows(); original = copy.deepcopy(data)
        result = summarize(data)['all_matched']
        expected = math.sqrt(((math.log(3)-math.log(2))**2+(math.log(3)-math.log(4))**2)/2)
        self.assertAlmostEqual(result['selected_mean_rmsle']['gnomon'], expected)
        self.assertEqual(result['union_hindsight_mean_rmsle'], 0)
        self.assertEqual(result['union_reduction_vs_actual_no_ledger'], 1)
        self.assertEqual(data, original)

    def test_failed_case_and_pending_remain(self):
        data = rows();data[0].update(valid=False, workflow_complete=False,
                                    selected_point=[0.,0.], forecasts=[])
        extra = copy.deepcopy(data[0]);extra['round']=1;data.append(extra)
        result = summarize(data)
        self.assertEqual(result['observed_sessions'],4)
        self.assertEqual(result['all_matched']['matched_cases'],1)
        self.assertEqual(result['all_matched']['valid']['plain'],0)
        self.assertGreater(result['all_matched']['selected_mean_rmsle']['plain'],0)
        self.assertEqual(result['pending'][0]['missing_arms'],['gnomon','ledger'])

    def test_identity_and_execution_rejections(self):
        for key, value in [('origin','other'),('actual',[2.,3.]),('fallback_point',[1.,1.])]:
            data=rows();data[0][key]=value
            with self.assertRaises(ValueError):summarize(data)
        data=rows();data[0]['forecasts']=[]
        with self.assertRaises(ValueError):summarize(data)
        data=rows();data[0]['forecasts']*=2
        with self.assertRaises(ValueError):summarize(data)
        with self.assertRaises(ValueError):summarize(rows()+rows())

    def test_zero_denominator_and_order(self):
        data=rows()
        for r in data:r['selected_point']=[1.,3.]
        result=summarize(data)
        self.assertIsNone(result['all_matched']['union_reduction_vs_actual_no_ledger'])
        self.assertEqual(result,summarize(list(reversed(data))))
        self.assertFalse(result['executable_policy'])

    def test_clipping_and_invalid_vectors(self):
        self.assertEqual(rmsle([-1.],[0.]),0)
        for p,a in [([],[]),([1.],[1.,2.]),([True],[1.]),([1.],[-1.]),([float('nan')],[1.])]:
            with self.assertRaises(ValueError):rmsle(p,a)

    def test_union_can_choose_different_arms_per_case_and_means_are_not_pooled(self):
        data=[]
        for number,errors in enumerate(((4.,2.,1.),(.5,1.,3.))):
            for template,error in zip(rows(),errors,strict=True):
                template.update(round=number,actual=[0.],fallback_point=[math.expm1(9.)],
                                selected_point=[math.expm1(error)])
                template['forecasts']=[{'execution_id':template['arm']+str(number),
                    'config_id':'c', 'point':template['selected_point']}]
                data.append(template)
        result=summarize(data)['all_matched']
        self.assertAlmostEqual(result['own_hindsight_mean_rmsle']['plain'],2.25)
        self.assertAlmostEqual(result['own_hindsight_mean_rmsle']['gnomon'],1.5)
        self.assertAlmostEqual(result['own_hindsight_mean_rmsle']['ledger'],2.)
        self.assertAlmostEqual(result['union_hindsight_mean_rmsle'],.75)
        self.assertAlmostEqual(result['union_reduction_vs_actual_no_ledger'],.5)


if __name__=='__main__':unittest.main()
