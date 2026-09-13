from copy import deepcopy
import unittest

from benchmarks.ledger_optimization.ml_cohort_screen import select, fold_request


class CohortSelectionTest(unittest.TestCase):
    def history(self):
        return [{'origin':f'2026-01-0{i}T00:00:00Z',
                 'last_target':f'2026-01-0{i+1}T00:00:00Z',
                 'outcome_recorded_at':f'2026-01-0{i+1}T00:00:00Z',
                 'scores':{'seasonal':3,'ridge':1,'random_forest':2}} for i in (1,2,3)]

    def test_cold_start_and_common_history(self):
        cv={'seasonal':0,'ridge':2,'random_forest':1}
        self.assertEqual(select(cv,[], '2026-01-05T00:00:00Z')['past'],'seasonal')
        r=select(cv,self.history(),'2026-01-05T00:00:00Z')
        self.assertEqual((r['cv'],r['past'],r['matched_origins']),('seasonal','ridge',3))

    def test_future_outcomes_cannot_change_selection(self):
        cv={'seasonal':0,'ridge':2,'random_forest':1};history=self.history()
        future=deepcopy(history[-1]);future['outcome_recorded_at']='2027-01-01T00:00:00Z'
        future['scores']={'seasonal':0,'ridge':100000,'random_forest':0}
        expected=select(cv,history,'2026-01-05T00:00:00Z')
        self.assertEqual(select(cv,history+[future],'2026-01-05T00:00:00Z'),expected)
        future['outcome_recorded_at']='2026-01-05T00:00:00Z'
        future['last_target']='2026-01-06T00:00:00Z'
        self.assertEqual(select(cv,history+[future],'2026-01-05T00:00:00Z'),expected)

    def test_no_duplicate_or_mixed_cohort_ranking(self):
        cv={'seasonal':0,'ridge':2,'random_forest':1};history=self.history()
        with self.assertRaises(ValueError):select(cv,history+history,'2026-01-05T00:00:00Z')
        history[0]['scores'].pop('ridge')
        with self.assertRaises(ValueError):select(cv,history,'2026-01-05T00:00:00Z')

    def test_backtest_has_only_prefix_history_and_correct_target_covariates(self):
        req={'history':list(range(40)), 'timestamps':[str(i) for i in range(40)],
             'past_covariates':[[i] for i in range(40)],'horizon':14}
        original=deepcopy(req);fold,actual=fold_request(req,20)
        self.assertEqual(fold['history'],list(range(20)))
        self.assertEqual(actual,list(range(20,34)))
        self.assertEqual(fold['future_covariates'],[[i] for i in range(20,34)])
        self.assertEqual(fold['cutoff'],'19')
        self.assertEqual(req,original)


if __name__=='__main__':unittest.main()
