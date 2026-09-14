import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from benchmarks.ledger_optimization.control_collection_096 import supervise


class CollectionControlTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.launch=self.root/'launch';self.output=self.root/'pilot'

    def run_child(self, tail='', gate=True, credentials_file=None):
        script='''import json,sys
from pathlib import Path
p=Path(sys.argv[1]);p.mkdir()
(p/'runner-exit.json').write_text(json.dumps({'exit_status':0,'continuation_launched':False}))
(p/'report.json').write_text(json.dumps({'complete':True,'audit_failures':[],'shutdown_record_gaps':[]}))
(p/'GATE.json').write_text(json.dumps({'passed':GATE,'accuracy_used_for_gate':False}))
print('retained stdout')
print('retained stderr',file=sys.stderr)
'''
        script=script.replace("'passed':GATE", "'passed':"+repr(gate))
        return supervise([sys.executable,'-c',script+tail,str(self.output)],self.launch,self.output,credentials_file)

    def verify_archive(self):
        inventory=json.loads((self.launch/'SHA256SUMS.json').read_text())
        finished=json.loads((self.launch/'FINISHED.json').read_text())
        archive=self.launch/'evidence.tar.gz'
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),finished['archive_sha256'])
        with tarfile.open(archive) as tar:
            for name,digest in inventory.items():
                self.assertEqual(hashlib.sha256(tar.extractfile(name).read()).hexdigest(),digest)
        self.assertFalse(finished['continuation_launched'])

    def test_success_records_process_exit_logs_and_verified_archive(self):
        self.assertTrue(self.run_child()['complete'])
        self.assertTrue(json.loads((self.launch/'pilot-process.json').read_text())['start_ticks'])
        self.assertEqual((self.launch/'pilot.stdout').read_text(),'retained stdout\n')
        self.assertEqual((self.launch/'pilot.stderr').read_text(),'retained stderr\n')
        self.assertFalse((self.launch/'INCOMPLETE.json').exists())
        self.verify_archive()

    def test_failed_completion_gate_still_preserves_complete_experiment(self):
        result=self.run_child(gate=False)
        self.assertTrue(result['complete']);self.assertFalse(result['continuation_gate_passed'])
        self.verify_archive()

    def test_failed_process_cannot_be_overridden_by_success_receipts(self):
        result=self.run_child('\nraise SystemExit(9)\n')
        self.assertFalse(result['complete']);self.assertEqual(result['exit_status'],9)
        self.assertTrue((self.launch/'INCOMPLETE.json').exists());self.verify_archive()

    def test_zero_exit_missing_receipts_remains_incomplete(self):
        result=self.run_child("\n(p/'report.json').unlink()\n")
        self.assertFalse(result['complete']);self.verify_archive()

    def test_audit_failure_prevents_success(self):
        result=self.run_child("\nr=json.loads((p/'report.json').read_text());r['audit_failures']=['mismatch'];(p/'report.json').write_text(json.dumps(r))\n")
        self.assertFalse(result['complete']);self.verify_archive()

    def test_spawn_failure_is_preserved_without_retry(self):
        result=supervise([str(self.root/'missing-executable')],self.launch,self.output)
        self.assertEqual(result['exit_status'],127)
        self.assertTrue((self.launch/'spawn-error.json').exists());self.verify_archive()

    def test_existing_or_nested_output_rejected_before_process(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError,'fresh'):
            self.run_child()
        self.assertFalse(self.launch.exists())
        self.output.rmdir()
        with self.assertRaisesRegex(ValueError,'fresh'):
            supervise([sys.executable,'-c','raise Exception()'],self.launch,self.launch/'nested')
        self.assertFalse(self.launch.exists())

    def test_outside_symlink_never_archived_or_marked_finished(self):
        with self.assertRaisesRegex(ValueError,'symlink'):
            self.run_child("\n(p/'outside').symlink_to('/etc/passwd')\n")
        self.assertFalse((self.launch/'FINISHED.json').exists())
        self.assertFalse((self.launch/'evidence.tar.gz').exists())

    def test_credentials_not_read_without_accepted_launch(self):
        result=self.run_child(credentials_file=self.root/'nonexistent-credentials')
        self.assertTrue(result['complete']);self.verify_archive()

    def test_credential_in_evidence_blocks_archive(self):
        key=self.root/'credentials';key.write_text('ENGY_API_KEY=synthetic-test-secret\n')
        with self.assertRaisesRegex(ValueError,'Credential scan failed'):
            self.run_child("\n(p/'accepted-launch.json').write_text('{}');(p/'leak').write_text('synthetic-test-secret')\n",credentials_file=key)
        self.assertFalse((self.launch/'evidence.tar.gz').exists())
        self.assertFalse((self.launch/'FINISHED.json').exists())

    def test_accepted_launch_scanned_without_copying_credentials(self):
        key=self.root/'credentials';key.write_text('ENGY_API_KEY=synthetic-test-secret\n')
        self.assertTrue(self.run_child("\n(p/'accepted-launch.json').write_text('{}')\n",credentials_file=key)['complete'])
        self.verify_archive()
        with tarfile.open(self.launch/'evidence.tar.gz') as archive:
            for item in archive.getmembers():
                self.assertNotIn(b'synthetic-test-secret',archive.extractfile(item).read())

    def test_missing_key_after_accepted_launch_withholds_archive(self):
        key=self.root/'credentials';key.write_text('ENGY_API_KEY=\n')
        with self.assertRaisesRegex(ValueError,'scan unavailable'):
            self.run_child("\n(p/'accepted-launch.json').write_text('{}')\n",credentials_file=key)
        self.assertFalse((self.launch/'evidence.tar.gz').exists())
        self.assertFalse((self.launch/'FINISHED.json').exists())


if __name__=='__main__':
    unittest.main()
