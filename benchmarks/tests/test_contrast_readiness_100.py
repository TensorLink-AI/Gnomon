"""Synthetic terminal-gate tests; do not represent paid evaluation outcomes."""
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.ledger_optimization.contrast_plan_100 import sha
from benchmarks.ledger_optimization.contrast_readiness_100 import verify_predecessor
from benchmarks.ledger_optimization.control_continuation_097 import completion
from benchmarks.tests.test_contrast_plan_100 import PARENT


@unittest.skipUnless(PARENT.exists(), 'requires frozen parent plan')
class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.output=self.root/'run';self.launch=self.root/'launch';self.proc=self.root/'proc'
        for p in (self.output,self.launch,self.proc):p.mkdir()
        self.parent=json.loads(PARENT.read_text()); self.identity={'pid':123,'boot_id':'synthetic','start_ticks':42}
        for name in ('launch.json','development-process.json'):self.write(self.launch/name,self.identity)
        self.write(self.launch/'development-exit.json',{'exit_status':0})
        self.write(self.output/'manifest.json',{'planned':312,'pilot_sessions_retained':36,'new_sessions_planned':276,
            'accuracy_used_for_promotion':False,'sources':self.parent['capsule']['sources'],'source_jobs_sha256':self.parent['task_source_sha256']})
        self.write(self.output/'complete.json',{'completed':312,'planned':312,'retained_pilot_sessions':36,'new_sessions':276,'source_unchanged':True})
        self.write(self.output/'runner-exit.json',{'exit_status':0,'final_gate_opened':False})
        jobs={}
        for c in self.parent['cases']:jobs.setdefault(c['series_id'],[]).append(c)
        self.write(self.output/'host-jobs.json',jobs)
        self.write(self.output/'report.json',{'complete':True,'audit_failures':[],'shutdown_record_gaps':[],
            'rows':[{'arm':a,**c} for a in self.parent['arms'] for c in self.parent['cases']]})
        self.write(self.launch/'AUDITED.json',completion(self.output,0))
        self.write(self.launch/'costs.json',{'deduplicated_total':{'sessions':312},'stages':{'retained_pilot':{'sessions':36},'continuation':{'sessions':276}}})
        self.seal()

    def write(self,p,value):p.write_text(json.dumps(value))
    def seal(self):
        inventory={}
        for label,root in (('launch',self.launch),('pilot',self.output)):
            for p in root.iterdir():
                if p.name not in ('SHA256SUMS.json','evidence.tar.gz','FINISHED.json'):inventory[label+'/'+p.name]=sha(p)
        self.write(self.launch/'SHA256SUMS.json',inventory)
        (self.launch/'evidence.tar.gz').write_bytes(b'synthetic archive fixture; actual file hashes are checked separately')
        self.write(self.launch/'FINISHED.json',{**completion(self.output,0),'further_execution_launched':False,
            'archive_sha256':sha(self.launch/'evidence.tar.gz'),'inventory_sha256':sha(self.launch/'SHA256SUMS.json')})
    def verify(self):return verify_predecessor(self.output,self.launch,PARENT,self.proc)

    def test_clean_receipts_allow_read_only_gate(self):
        result=self.verify();self.assertTrue(result['predecessor_terminal_verified']);self.assertEqual(result['engy_calls'],0)
        self.assertFalse(result['accuracy_used_for_admission'])

    def test_live_process_overrides_finished_receipts(self):
        (self.proc/'123').mkdir();(self.proc/'sys/kernel/random').mkdir(parents=True)
        (self.proc/'sys/kernel/random/boot_id').write_text('synthetic')
        fields=['S']+['0']*18+['42'];(self.proc/'123/stat').write_text('123 (worker) '+' '.join(fields))
        with self.assertRaisesRegex(ValueError,'still live'):self.verify()

    def test_changed_evidence_rejected(self):
        p=self.output/'report.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Retained evidence changed'):self.verify()

    def test_wrong_cohort_rejected_even_when_resealed(self):
        for name in ('host-jobs.json','report.json'):
            p=self.output/name;p.write_text(p.read_text().replace('2016-08-16','2016-08-17'))
        self.seal()
        with self.assertRaisesRegex(ValueError,'Wrong predecessor cohort'):self.verify()

    def test_source_mismatch_rejected_even_when_resealed(self):
        p=self.output/'manifest.json';d=json.loads(p.read_text());d['sources']['core.py']='wrong';self.write(p,d);self.seal()
        with self.assertRaisesRegex(ValueError,'source or task'):self.verify()

    def test_missing_costs_or_incomplete_flag_reject(self):
        (self.launch/'costs.json').unlink()
        with self.assertRaises(OSError):self.verify()
        self.write(self.launch/'INCOMPLETE.json',{})
        with self.assertRaisesRegex(ValueError,'evidence incomplete'):self.verify()


if __name__=='__main__':unittest.main()
