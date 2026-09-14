import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmarks.ledger_optimization.configuration_search_run import Execution,run_case,domain_run,digest
from benchmarks.ledger_optimization.configured_hourly import catalog,revision


def fixture():
    task={'task_id':'synthetic','series_id':'electricity:a','domain':'electricity','round':-8,'origin':'2020-01-31T00:00:00+00:00',
        'history':[1.]*730,'features':[0.]*12,'unit':'synthetic','source_sha256':'synthetic'}
    starters={r['config_id']:{'production':[2.]*24,'folds':{str(e):[2.]*24 for e in (658,682,706)},'source_ref':{'synthetic':True}} for r in catalog()[:6]}
    return task,starters


class ConfigurationSearchRunTest(unittest.TestCase):
    def test_full_logical_budget_and_cross_arm_cache(self):
        task,starters=fixture();calls=[]
        def predictor(values,labels,future,config):
            calls.append((len(values),config));self.assertEqual(len(future),24);self.assertTrue(all(v==1 for v in values));return [1.]*24
        with tempfile.TemporaryDirectory() as temp:
            a=run_case(temp,task,starters,'control',[],predictor);b=run_case(temp,task,starters,'ledger',[],predictor)
            self.assertEqual(a['logical_attempts'],58);self.assertEqual(b['logical_attempts'],58)
            self.assertEqual(a['physical_computations'],34);self.assertEqual(b['physical_computations'],0);self.assertEqual(len(calls),34)
            self.assertEqual(a['point'],b['point']);self.assertEqual(a['selected'],b['selected'])
            self.assertEqual(len(a['backtests']),17);self.assertEqual(len(a['proposal_refs']),11)
            for arm in ('control','ledger'):
                logs=list((Path(temp)/'cases/synthetic'/arm).glob('attempt-*.json'));self.assertEqual(len(logs),58)
                self.assertTrue(all(json.loads(p.read_text())['completed'] for p in logs))

    def test_failure_is_charged_and_preserved(self):
        task,starters=fixture()
        def fail(*args):raise RuntimeError('synthetic numerical failure')
        with tempfile.TemporaryDirectory() as temp:
            executor=Execution(temp,task,starters,'control',fail)
            with self.assertRaises(RuntimeError):executor.call(catalog()[6]['config'],658)
            self.assertEqual(len(executor.events),1);self.assertTrue(executor.events[0]['physical_started']);self.assertFalse(executor.events[0]['completed'])
            saved=json.loads((executor.home/'attempt-01.json').read_text());self.assertEqual(saved['error'],'RuntimeError')

    def test_cache_metadata_tampering_rejected(self):
        task,starters=fixture()
        with tempfile.TemporaryDirectory() as temp:
            first=Execution(temp,task,starters,'control');_,event=first.call(catalog()[0]['config'],730)
            path=Path(temp)/event['cache_ref'];cached=json.loads(path.read_text());cached['request']['input_sha256']='wrong';path.write_text(json.dumps(cached))
            second=Execution(temp,task,starters,'ledger')
            with self.assertRaisesRegex(ValueError,'identity'):second.call(catalog()[0]['config'],730)
            self.assertEqual(len(second.events),1);self.assertFalse(second.events[0]['physical_started'])

    def test_same_origin_history_barrier(self):
        task,starters=fixture();seen=[];items=[]
        for i,(series,origin) in enumerate((('electricity:a','2020-01-01T00:00:00+00:00'),('electricity:b','2020-01-01T00:00:00+00:00'),('electricity:a','2020-01-08T00:00:00+00:00'))):
            inp={**task,'task_id':str(i),'series_id':series,'origin':origin,'round':-8+(i==2)}
            items.append({'input':inp,'starters':starters,'target':[1.]*24,'guards':{},'last_target':origin})
        def fake(root,inp,prior,arm,history):
            seen.append((inp['task_id'],arm,len(history)));self.assertTrue(all(r['origin']<inp['origin'] for r in history))
            return {'point':[1.]*24,'selected':{'config_id':'synthetic'},'study':{'origin':inp['origin']},'logical_attempts':0,'physical_computations':0,
                'physical_estimator_fits':0,'reused_computations':0,'surrogate_solves':0,'seconds':0.,'cpu_seconds':0.}
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'outcomes').mkdir()
            with patch('benchmarks.ledger_optimization.configuration_search_run.run_case',fake):domain_run('electricity',items,temp)
        self.assertEqual([n for task,arm,n in seen if task in ('0','1')],[0,0,0,0])
        self.assertEqual([n for task,arm,n in seen if task=='2'],[2,2])


if __name__=='__main__':unittest.main()
