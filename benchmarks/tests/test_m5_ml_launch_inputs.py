from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import m5_ml_launch_inputs as m


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj))


class LaunchInputsTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.parent=self.root/'parent.json';self.capsule=self.root/'capsule';self.worker=self.root/'worker'
        self.plan=self.root/'plan.json';self.registry=self.root/'registry'
        dump(self.parent,{'agent':{'model':'deepseek-v4.1-flash'},'gnomon':{'version':'1.2.0'},'budgets':{'numerical_attempts':60},'comparison':{'include_failures':True},'final_target':{'relative_rmsle_reduction':.2}})
        p=self.capsule/'benchmarks/hermes_ml_checkpoint_v6/run.py';p.parent.mkdir(parents=True);p.write_text('# synthetic\n')
        dump(self.capsule/'cohort-contract.json',{'synthetic':True})
        self.sources={'run.py':m.sha(p)}
        dump(self.capsule/'capsule.json',{'requested_seed':19,'sources':self.sources,'cohort_contract_sha256':m.sha(self.capsule/'cohort-contract.json')})
        rows=[{'arm':a,'series_id':'synthetic-collection-worker','round':n,'valid':True,'workflow_complete':True} for a in ('plain','gnomon','ledger') for n in (0,1)]
        dump(self.worker/'passed.json',{'passed':True,'engy_calls':0,'requested_seed':19,'sources':self.sources,'checks':[{'passed':True}],'numerical_attempts':48,'scripted_model_responses':30,'independent_audit_checks':3859})
        dump(self.worker/'report.json',{'rows':rows,'audit_checks':3859,'audit_failures':[],'shutdown_record_gaps':[]})
        dump(self.worker/'manifest.json',{'requested_seed':19,'sources':self.sources,'build':{'package_version':'1.2.0','source_sha256':m.BUILD_SHA},'inventory':{'synthetic':True}})
        identities={19:{'capsule.json':m.sha(self.capsule/'capsule.json'),**{name:m.sha(self.worker/name) for name in ('passed.json','report.json','manifest.json')}}}
        self.addCleanup(patch.stopall)
        patch.object(m,'IDENTITIES',identities).start()
        patch.object(m,'PARENT_PLAN_SHA',m.sha(self.parent)).start()
        patch.object(m,'authenticated_contract',return_value={'cases':[{'series_id':'synthetic','round':0}]}).start()

    def args(self):return (self.parent,self.capsule,self.worker,b'synthetic',b'synthetic')
    def kwargs(self):return {'seed':19,'registry':self.registry}
    def freeze(self):return m.freeze_plan(self.plan,*self.args(),**self.kwargs())

    def test_plan_matches_source_seed_budget_and_no_dispatch(self):
        p=self.freeze();self.assertFalse(p['execution_authorized']);self.assertEqual(p['planned']['total_sessions'],624)
        self.assertEqual(p,m.verify_plan(self.plan,*self.args(),**self.kwargs()))
        self.assertEqual(p['budgets'],{'numerical_attempts':60})
        self.assertFalse(self.registry.exists())
        with self.assertRaisesRegex(ValueError,'Fresh'):self.freeze()

    def test_changed_plan_source_or_proof_fails(self):
        p=self.freeze();p['budgets']['numerical_attempts']=61;dump(self.plan,p)
        with self.assertRaisesRegex(ValueError,'Prospective plan'):m.verify_plan(self.plan,*self.args(),**self.kwargs())
        source=self.capsule/'benchmarks/hermes_ml_checkpoint_v6/run.py';source.write_text('# changed')
        with self.assertRaisesRegex(ValueError,'sources changed'):m.prepare_plan(*self.args(),**self.kwargs())
        source.write_text('# synthetic\n');proof=self.worker/'passed.json';proof.write_text('{}')
        with self.assertRaisesRegex(ValueError,'frozen input'):m.prepare_plan(*self.args(),**self.kwargs())

    def test_unknown_seed_and_relative_registry(self):
        for seed in (7,True,3):
            with self.assertRaises(ValueError):m.prepare_plan(*self.args(),seed=seed,registry=self.registry)
        with self.assertRaisesRegex(ValueError,'absolute'):m.prepare_plan(*self.args(),seed=19,registry=Path('relative'))

    def test_concurrent_reservation_allows_only_one_even_with_different_outputs(self):
        p=self.freeze()
        def call(i):
            try:return m.reserve_stage(self.plan,p,stage='pilot',output=self.root/f'output-{i}',controller=self.root/f'controller-{i}')
            except FileExistsError:return None
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(call,(1,2)))
        self.assertEqual(sum(r is not None for r in results),1)
        self.assertEqual(len(list(self.registry.glob('*.json'))),1)
        self.assertFalse((self.root/'output-1').exists())
        # Reservation survives even though no child has been started.
        self.assertIsNone(call(3))

    def test_stage_isolation_and_existing_outputs_rejected(self):
        p=self.freeze()
        first=m.reserve_stage(self.plan,p,stage='pilot',output=self.root/'a',controller=self.root/'b')
        second=m.reserve_stage(self.plan,p,stage='complete',output=self.root/'c',controller=self.root/'d')
        self.assertNotEqual(first,second)
        (self.root/'existing').mkdir()
        with self.assertRaisesRegex(ValueError,'Fresh'):
            m.reserve_stage(self.plan,p,stage='pilot',output=self.root/'existing',controller=self.root/'e')

    def test_changed_verified_plan_does_not_reserve(self):
        p=self.freeze();different=deepcopy(p);different['requested_seed']=7
        with self.assertRaisesRegex(ValueError,'changed before'):
            m.reserve_stage(self.plan,different,stage='pilot',output=self.root/'a',controller=self.root/'b')
        self.assertFalse(self.registry.exists())


if __name__=='__main__':unittest.main()
