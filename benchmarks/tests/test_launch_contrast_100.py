import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization.launch_contrast_100 import normalized_preflight, verify_inputs, pilot_gate, fresh_outputs, main, run_pilot
from benchmarks.tests.test_contrast_plan_100 import PARENT,CAPSULE,WORKER,TASKS

PLAN=Path('results/contrast-100-prospective-plan-002/plan.json')


@unittest.skipUnless(PLAN.exists() and WORKER.exists(),'requires frozen candidate evidence')
class LaunchContrastTests(unittest.TestCase):
    def setUp(self):self.plan=json.loads(PLAN.read_text())
    def report(self):
        return {'complete':True,'audit_failures':[],'shutdown_record_gaps':[],
            'rows':[{'arm':a,**c,'valid':True,'workflow_complete':True,'rmsle':1000000} for a in self.plan['arms'] for c in self.plan['cases'] if c['stage']=='pilot']}

    def test_preflight_adaptation_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'preflight.json';proof=normalized_preflight(self.plan,WORKER);p.write_text(json.dumps(proof))
            verify_inputs(PARENT,CAPSULE,WORKER,TASKS,PLAN,p)
            proof['tested_sources']['core.py']='wrong';p.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'normalized worker proof'):verify_inputs(PARENT,CAPSULE,WORKER,TASKS,PLAN,p)

    def test_gate_counts_workflows_not_accuracy(self):
        report=self.report();self.assertTrue(pilot_gate(report,self.plan)['passed'])
        report['rows'][0]['workflow_complete']=False
        self.assertTrue(pilot_gate(report,self.plan)['passed'])
        report['rows'][1]['workflow_complete']=False
        self.assertFalse(pilot_gate(report,self.plan)['passed'])
        for row in report['rows']:row['rmsle']=0
        self.assertFalse(pilot_gate(report,self.plan)['passed'])
        report=self.report();report['rows'][0]['valid']=False
        self.assertFalse(pilot_gate(report,self.plan)['passed'])

    def test_gate_rejects_wrong_cohort_and_audit_failure(self):
        report=self.report();report['rows'][0]['origin']='2099-01-01'
        with self.assertRaisesRegex(ValueError,'cohort'):pilot_gate(report,self.plan)
        report=self.report();report['audit_failures']=['changed input']
        self.assertFalse(pilot_gate(report,self.plan)['passed'])

    def test_output_cannot_modify_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(output=root/'new',predecessor_audit=root/'audit',previous_root=root/'old',previous_launch=root/'old-launch',previous_capsule=root/'old-code',capsule=root/'new-code',worker_proof_root=root/'proof',runtime=root/'runtime')
            fresh_outputs(args)
            args.output=args.previous_root/'new'
            with self.assertRaisesRegex(ValueError,'overlap'):fresh_outputs(args)
            args.output=root/'new';args.predecessor_audit=args.output/'audit'
            with self.assertRaisesRegex(ValueError,'separate'):fresh_outputs(args)


    def test_runner_retains_gate_and_does_not_start_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'pilot';credentials=root/'synthetic-credentials'
            credentials.write_text('ENGY_API_KEY=SYNTHETIC_NOT_A_REAL_KEY\n')
            proof=root/'proof.json';proof.write_text('{}')
            args=SimpleNamespace(output=output,plan=PLAN,preflight=proof,credentials_file=credentials)
            def dump(path,value):path.write_text(json.dumps(value))
            def fake_main():
                output.mkdir();self.assertEqual(run.key(),'SYNTHETIC_NOT_A_REAL_KEY')
            run=SimpleNamespace(main=fake_main,dump=dump)
            with patch('sys.argv',['test']), patch('builtins.print'):
                run_pilot(run,lambda path:self.report(),args,self.plan,{'synthetic':True})
            self.assertTrue(json.loads((output/'GATE.json').read_text())['passed'])
            self.assertEqual(json.loads((output/'runner-exit.json').read_text()),{'exit_status':0,'continuation_launched':False})
            self.assertNotIn('SYNTHETIC_NOT_A_REAL_KEY',(output/'accepted-launch.json').read_text())
            self.assertEqual((output/'pilot-plan.json').read_bytes(),PLAN.read_bytes())

    def test_failed_runner_is_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'pilot';args=SimpleNamespace(output=output,plan=PLAN,preflight=root/'proof',credentials_file=root/'missing')
            calls=[]
            def fake_main():output.mkdir();calls.append(1);raise RuntimeError('synthetic stop')
            run=SimpleNamespace(main=fake_main,dump=lambda p,v:p.write_text(json.dumps(v)))
            with patch('sys.argv',['test']):
                with self.assertRaisesRegex(RuntimeError,'synthetic stop'):run_pilot(run,None,args,self.plan,{})
            self.assertEqual(calls,[1])
            self.assertEqual(json.loads((output/'runner-exit.json').read_text())['exit_status'],1)
            self.assertFalse((output/'accepted-launch.json').exists())

    def test_live_predecessor_stops_before_proof_credentials_or_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fields=('parent-plan','capsule','worker-proof-root','plan','preflight','task-source','runtime','previous-root','previous-launch','previous-capsule','predecessor-audit','output')
            argv=['launcher']
            for field in fields:argv+=['--'+field,str(root/field)]
            argv+=['--credentials-file',str(root/'missing-secret')]
            with patch('sys.argv',argv), patch('benchmarks.ledger_optimization.launch_contrast_100.verify_predecessor',side_effect=ValueError('Predecessor process is still live')), patch('benchmarks.ledger_optimization.launch_contrast_100.verify_inputs') as inputs, patch('benchmarks.ledger_optimization.launch_contrast_100.subprocess.run') as child:
                with self.assertRaisesRegex(ValueError,'still live'):main()
                inputs.assert_not_called();child.assert_not_called()
            self.assertFalse((root/'output').exists());self.assertFalse((root/'predecessor-audit').exists())


if __name__=='__main__':unittest.main()
