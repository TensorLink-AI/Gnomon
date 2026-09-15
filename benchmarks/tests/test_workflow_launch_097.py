"""A new workflow pilot admits complete failed pilots, never live or partial ones."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.ledger_optimization.launch_workflow_097 import verify_predecessor, sha
from benchmarks.ledger_optimization.workflow_plan_097 import freeze

PARENT = Path('results/collection-096-prospective-plan-002/plan.json')
CAPSULE = Path('results/workflow-097-offline-002/capsule')
JOBS = Path('results/collection-096-dispatch-bundle-001/payload/host-jobs.json')


class WorkflowLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.pilot = self.root/'pilot'; self.launch = self.root/'launch'
        self.pilot.mkdir(); self.launch.mkdir(); self.proc = self.root/'proc'
        parent = json.loads(PARENT.read_text())
        (self.pilot/'pilot-plan.json').write_bytes(PARENT.read_bytes())
        rows = [{'arm':a,'series_id':c['series_id'],'round':c['round'],'valid':True,'workflow_complete':True}
                for c in parent['cases'] if c['stage']=='pilot' for a in ('plain','gnomon','ledger')]
        for r in rows:
            if r['arm']=='plain' and r['round']==0:
                r['workflow_complete']=False
        self.write(self.pilot/'report.json', {'complete':True,'audit_failures':[], 'shutdown_record_gaps':[], 'rows':rows})
        self.write(self.pilot/'GATE.json', {'passed':False,'accuracy_used_for_gate':False})
        self.write(self.pilot/'runner-exit.json', {'exit_status':0,'continuation_launched':False})
        self.write(self.pilot/'manifest.json', {'planned':36})
        for number,name in enumerate(('launch.json','pilot-process.json')):
            self.write(self.launch/name, {'pid':100+number,'start_ticks':'10','boot_id':'synthetic'})
        self.write(self.launch/'AUDITED.json', {'complete':True})
        self.write(self.launch/'pilot-exit.json', {'exit_status':0})
        (self.launch/'evidence.tar.gz').write_bytes(b'synthetic evidence')
        self.seal()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))

    def seal(self):
        inventory = {'pilot/'+p.name:sha(p) for p in self.pilot.iterdir()}
        inventory.update({'launch/'+name:sha(self.launch/name) for name in ('AUDITED.json','pilot-exit.json')})
        self.write(self.launch/'SHA256SUMS.json', inventory)
        self.write(self.launch/'FINISHED.json', {'complete':True,'continuation_gate_passed':False,
            'archive_sha256':sha(self.launch/'evidence.tar.gz'),'inventory_sha256':sha(self.launch/'SHA256SUMS.json')})

    def verify(self): return verify_predecessor(self.pilot,self.launch,self.proc)

    def test_completed_failed_gate_is_retained_not_retried(self):
        result=self.verify()
        self.assertFalse(result['previous_gate_passed'])
        self.assertFalse(result['accuracy_used_for_admission'])

    def test_live_child_rejected_even_with_terminal_markers(self):
        p=self.proc/'101/stat';p.parent.mkdir(parents=True)
        fields=['S']+['0']*18+['10'];p.write_text('101 (worker) '+' '.join(fields))
        p=self.proc/'sys/kernel/random/boot_id';p.parent.mkdir(parents=True);p.write_text('synthetic')
        with self.assertRaisesRegex(ValueError,'still live'):self.verify()

    def test_missing_terminal_receipt_rejected(self):
        (self.launch/'FINISHED.json').unlink()
        with self.assertRaises(FileNotFoundError):self.verify()

    def test_failed_archive_hash_rejected(self):
        (self.launch/'evidence.tar.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'archive'):self.verify()

    def test_incomplete_audit_cannot_promote(self):
        p=self.pilot/'report.json';value=json.loads(p.read_text());value['audit_failures']=['bad']
        self.write(p,value);self.seal()
        with self.assertRaisesRegex(ValueError,'cleanly'):self.verify()

    def test_repeated_case_does_not_replace_missing_case(self):
        p=self.pilot/'report.json';value=json.loads(p.read_text())
        value['rows'][3]=value['rows'][0];self.write(p,value);self.seal()
        with self.assertRaisesRegex(ValueError,'cohort'):self.verify()

    def test_gate_cannot_claim_pass_when_full_counts_fail(self):
        self.write(self.pilot/'GATE.json', {'passed':True,'accuracy_used_for_gate':False});self.seal()
        with self.assertRaisesRegex(ValueError,'gate disagrees'):self.verify()

    def test_plan_preserves_budget_population_and_gate(self):
        path=self.root/'plan.json';new=freeze(PARENT,CAPSULE,JOBS,path)
        old=json.loads(PARENT.read_text())
        for field in ('budgets','cases','pilot_gate','agent','gnomon','comparison','final_target'):
            self.assertEqual(new[field],old[field])
        self.assertTrue(new['amendment']['096_pilot_excluded_from_097_scores'])
        with self.assertRaisesRegex(ValueError,'Fresh'):freeze(PARENT,CAPSULE,JOBS,path)


if __name__=='__main__':unittest.main()
