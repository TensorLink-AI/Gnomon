import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from benchmarks.ledger_optimization.launch_collection_096 import verify_predecessors, verify_inputs


class CollectionLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.proc=self.root/'proc'
        p=self.proc/'sys/kernel/random';p.mkdir(parents=True);(p/'boot_id').write_text('boot')
        self.paid=self.root/'paid';self.seed=self.root/'seed';self.paid.mkdir();self.seed.mkdir()
        for root,pid in ((self.paid,101),(self.seed,102)):
            self.write(root/'launch.json',{'pid':pid,'start_ticks':'20','boot_id':'boot'})
            self.write(root/'FINISHED.json',{})
        self.write(self.paid/'development-process.json',{'pid':103,'start_ticks':'20'})
        self.write(self.paid/'development-exit.json',{'exit_status':0})
        self.write(self.paid/'AUDITED.json',{'complete':True})
        results=[]
        for number,pid in ((7,104),(19,105)):
            self.write(self.seed/f'process-{number}.json',{'pid':pid,'start_ticks':'20'})
            self.write(self.seed/f'exit-{number}.json',{'exit_code':0})
            p=self.seed/f'integration-{number}/passed.json';self.write(p,{'passed':True})
            results.append({'seed':number,'passed_sha256':self.sha(p)})
        self.write(self.seed/'passed.json',{'passed':True,'engy_calls':0,'results':results})
        self.write(self.seed/'FINISHED.json',{'passed_sha256':self.sha(self.seed/'passed.json')})

    def write(self,p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

    def sha(self,p):return hashlib.sha256(p.read_bytes()).hexdigest()

    def live(self,pid,ticks='20'):
        p=self.proc/str(pid);p.mkdir();fields=['S']+['0']*18+[ticks]+['0']*3
        (p/'stat').write_text(f'{pid} (worker) '+' '.join(fields))

    def verify(self):return verify_predecessors(self.paid,self.seed,self.proc)

    def test_complete_terminal_predecessors(self):
        self.assertEqual(len(self.verify()),4)

    def test_live_controller_despite_completion_marker_rejected(self):
        self.live(101)
        with self.assertRaisesRegex(ValueError,'still live'):self.verify()

    def test_live_paid_child_rejected(self):
        self.live(103)
        with self.assertRaisesRegex(ValueError,'still live'):self.verify()

    def test_live_seed_child_rejected(self):
        self.live(105)
        with self.assertRaisesRegex(ValueError,'still live'):self.verify()

    def test_pid_reuse_does_not_block_completed_predecessor(self):
        self.live(101,ticks='21');self.verify()

    def test_incomplete_or_changed_evidence_rejected(self):
        self.write(self.paid/'INCOMPLETE.json',{})
        with self.assertRaisesRegex(ValueError,'cleanly'):self.verify()
        (self.paid/'INCOMPLETE.json').unlink()
        self.write(self.seed/'integration-19/passed.json',{'passed':False})
        with self.assertRaisesRegex(ValueError,'evidence changed'):self.verify()

    def test_nonzero_exit_rejected(self):
        self.write(self.seed/'exit-7.json',{'exit_code':1})
        with self.assertRaisesRegex(ValueError,'process failed'):self.verify()

    def test_source_identity_and_task_hash_checks(self):
        cap=self.root/'capsule';package=cap/'benchmarks/hermes_ml_checkpoint_v6'
        package.mkdir(parents=True);(package/'lab.py').write_text('source')
        sources={'lab.py':self.sha(package/'lab.py')};manifest={'sources':sources}
        self.write(cap/'capsule.json',manifest)
        task=self.root/'jobs.json';task.write_text('{}')
        plan=self.root/'plan.json';proof=self.root/'proof.json'
        self.write(plan,{'capsule_sha256':self.sha(cap/'capsule.json'),'capsule':manifest,
            'task_source_sha256':self.sha(task),'arms':['plain','gnomon','ledger'],
            'planned':{'pilot_sessions':36,'continuation_sessions':276,'total_sessions':312},'final_gate_opened':False})
        self.write(proof,{'passed':True,'engy_calls':0,'tested_sources':sources,'checks':[{'passed':True}]})
        verify_inputs(cap,plan,proof,task)
        task.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'development plan mismatch'):verify_inputs(cap,plan,proof,task)
        task.write_text('{}');(package/'lab.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'identities differ'):verify_inputs(cap,plan,proof,task)


    def test_module_launch_uses_capsule_despite_loaded_parent_package(self):
        cap=self.root/'capsule';package=cap/'benchmarks/hermes_ml_checkpoint_v6'
        package.mkdir(parents=True)
        (package/'run.py').write_text('from pathlib import Path\nHERE=Path(__file__).resolve().parent\ndef runtime_inventory(): return {}\n')
        (package/'analyze.py').write_text('def analyze(*args): raise AssertionError("check-only must not analyze")\n')
        script="""from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import benchmarks.ledger_optimization.launch_collection_096 as launch
import sys
root=Path(sys.argv[1])
args=SimpleNamespace(capsule=root/'capsule',runtime=root/'runtime',task_source=root/'jobs',
 plan=root/'plan',preflight=root/'proof',paid_predecessor=root/'paid',seed_predecessor=root/'seed',
 output=root/'never-created',check_only=True,credentials_file=None)
with patch.object(launch.argparse.ArgumentParser,'parse_args',return_value=args), \
     patch.object(launch,'verify_predecessors',return_value={}), \
     patch.object(launch,'verify_inputs',return_value=({}, {'tested_inventory':{}}, {})):
 launch.main()
assert not args.output.exists()
"""
        result=subprocess.run([sys.executable,'-c',script,str(self.root)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['checks_passed'])


if __name__=='__main__':unittest.main()
