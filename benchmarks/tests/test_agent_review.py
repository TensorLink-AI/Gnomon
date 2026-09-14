import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.ledger_optimization.agent_review import brief, review, compact_bytes, WINDOWS


class AgentReviewTest(unittest.TestCase):
    def setUp(self):
        self.task={'series_id':'sales','unit':'widgets','horizon':2,'origin':'2026-01-10T00:00:00+00:00'}
        window={'status':'ok','matched_origins':1,'n':2,'start':'2026-01-01T00:00:00+00:00',
                'end':'2026-01-01T00:00:00+00:00','models':[{'provider':'a','revision':'r1','rmsle':.2},
                {'provider':'b','revision':'r2','rmsle':.3}],
                'origins':[{'origin':'2026-01-01T00:00:00+00:00','actual_ids':['x','y'],'models':[]}]}
        self.full={'metric':'rmsle','as_of':self.task['origin'],'status':'evidence_available',
            'configuration_index':{'ca':{'provider':'a','revision':'r1','config':{}},'cb':{'provider':'b','revision':'r2','config':{}}},
            'cards':[{'left_config_id':'ca','right_config_id':'cb','providers':{'a':'r1','b':'r2'},
                'windows':{k:copy.deepcopy(window) for k in WINDOWS},'recent_lifetime_disagreement':False,'excluded':[]}],
            'offset':0,'next_offset':None,'total_pairs':1,'catalog_excluded':[],
            'global_past_origins':[window['start']],'window_semantics':'Global past-origin windows',
            'pair_order':'Recency then identity','ledger_queries':3,'provider_calls':0}

    def render(self):return brief(self.full,self.task,'full.json','0'*64)

    def test_exact_facts_aliases_and_no_mutation(self):
        before=copy.deepcopy(self.full);r=self.render();w=r['cards'][0]['windows']
        self.assertEqual(w['last_4_origins']['scores'],{'ca':.2,'cb':.3})
        self.assertEqual(w['last_4_origins']['lowest_error_config_ids'],['ca'])
        self.assertEqual(w['lifetime']['same_as'],'last_4_origins')
        self.assertEqual(r['configuration_index'],self.full['configuration_index'])
        self.assertEqual(before,self.full)
        self.assertTrue(r['pagination']['all_pairs_included'])

    def test_equal_scores_different_evidence_not_aliased(self):
        self.full['cards'][0]['windows']['lifetime']['origins'][0]['actual_ids']=['revision-x','revision-y']
        self.assertNotIn('same_as',self.render()['cards'][0]['windows']['lifetime'])

    def test_tie_is_preserved(self):
        for w in self.full['cards'][0]['windows'].values():w['models'][1]['rmsle']=.2
        self.assertEqual(self.render()['cards'][0]['windows']['last_4_origins']['lowest_error_config_ids'],['ca','cb'])

    def test_pagination_not_completion(self):
        self.full.update(total_pairs=20,next_offset=1)
        r=self.render();self.assertFalse(r['pagination']['all_pairs_included'])
        self.assertEqual(r['pagination']['next_call']['arguments'],{'offset':1,'limit':1})
        self.assertEqual(r['pagination']['next_call']['valid_for_origin'],self.task['origin'])
        self.assertFalse(r['forecast_selection_made'])
        self.full['next_offset']=12
        with self.assertRaises(ValueError):self.render()

    def test_missing_support_is_not_zero_error(self):
        for w in self.full['cards'][0]['windows'].values():
            w.update(status='insufficient_evidence',matched_origins=0,n=0,start=None,end=None,models=[],origins=[])
        self.full['cards'][0].pop('recent_lifetime_disagreement')
        r=self.render();w=r['cards'][0]['windows']['last_4_origins']
        self.assertEqual(w['scores'],{})
        self.assertEqual(w['lowest_error_config_ids'],[])
        self.assertIsNone(r['cards'][0]['recent_lifetime_disagreement'])

    def test_cold_start_has_no_invented_comparison(self):
        self.full.update(status='insufficient_evidence',configuration_index={},cards=[],
                         global_past_origins=[],total_pairs=0)
        result=self.render()
        self.assertEqual(result['status'],'insufficient_evidence')
        self.assertEqual(result['cards'],[])
        self.assertTrue(result['pagination']['all_pairs_included'])
        self.assertFalse(result['forecast_selection_made'])

    def test_reject_invalid_identity_counts_values_cutoff_and_execution(self):
        original=copy.deepcopy(self.full)
        for mutate in (lambda r:r.update(as_of='2026-01-11T00:00:00+00:00'),
                       lambda r:r.update(provider_calls=1),
                       lambda r:r['cards'][0]['providers'].update(a='wrong'),
                       lambda r:r['cards'][0]['windows']['last_4_origins'].update(n=1),
                       lambda r:r['cards'][0]['windows']['last_4_origins']['models'][0].update(rmsle=float('nan')),
                       lambda r:r['cards'][0]['windows']['last_4_origins']['origins'][0].update(origin=self.task['origin'])):
            self.full=copy.deepcopy(original);mutate(self.full)
            with self.assertRaises(ValueError):self.render()

    def test_saved_evidence_bytes_match_hash_and_existing_file_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp, patch('benchmarks.ledger_optimization.agent_review.development_cards',return_value=self.full) as query:
            path=Path(tmp)/'full.json';r=review(None,[],self.task,None,path,pair=['ca','cb'])
            self.assertEqual(path.read_bytes(),compact_bytes(self.full))
            self.assertEqual(r['full_evidence']['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(query.call_args.kwargs,{'pair':['ca','cb']})
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):review(None,[],self.task,None,path)
            self.assertEqual(path.read_bytes(),before)
