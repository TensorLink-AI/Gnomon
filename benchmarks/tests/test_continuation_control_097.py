"""Continuation supervision preserves failures/costs and never promotes partial state."""
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from benchmarks.ledger_optimization.control_continuation_097 import completion, supervise
from benchmarks.ledger_optimization.control_collection_096 import sha


def fixture(root):
    root.mkdir()
    jobs={f's{i}':[{'round':n} for n in range(26)] for i in range(4)}
    rows=[{'arm':a,'series_id':s,'round':n,'valid':False} for a in ('plain','gnomon','ledger')
          for s in jobs for n in range(26)]
    values={'host-jobs.json':jobs,
        'manifest.json':{'planned':312,'pilot_sessions_retained':36,'new_sessions_planned':276,
                         'accuracy_used_for_promotion':False},
        'complete.json':{'planned':312,'completed':312,'retained_pilot_sessions':36,
                         'new_sessions':276,'source_unchanged':True},
        'runner-exit.json':{'exit_status':0,'final_gate_opened':False},
        'report.json':{'complete':True,'audit_failures':[],'shutdown_record_gaps':[],'rows':rows}}
    for name,value in values.items():(root/name).write_text(json.dumps(value))
    return values


class ContinuationControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.output=self.root/'run';self.launch=self.root/'launch'

    def test_failed_forecasts_are_results_not_an_admission_filter(self):
        fixture(self.output)
        self.assertTrue(completion(self.output,0)['complete'])

    def test_exit_failure_cannot_be_hidden_by_complete_report(self):
        fixture(self.output)
        self.assertFalse(completion(self.output,2)['complete'])

    def test_duplicate_rows_do_not_replace_missing_cases(self):
        values=fixture(self.output);report=values['report.json'];report['rows'][0]=report['rows'][1]
        (self.output/'report.json').write_text(json.dumps(report))
        self.assertFalse(completion(self.output,0)['complete'])

    def test_unfinished_or_changed_prefix_counts_fail(self):
        values=fixture(self.output);values['complete.json']['retained_pilot_sessions']=35
        (self.output/'complete.json').write_text(json.dumps(values['complete.json']))
        self.assertFalse(completion(self.output,0)['complete'])

    def test_final_holdout_flag_rejected(self):
        fixture(self.output)
        (self.output/'runner-exit.json').write_text(json.dumps({'exit_status':0,'final_gate_opened':True}))
        self.assertFalse(completion(self.output,0)['complete'])

    def test_failed_launch_does_not_read_credentials_or_restart(self):
        result=supervise([sys.executable,'-c','raise SystemExit(2)'],self.launch,self.output,
                         self.root/'nonexistent-credential')
        self.assertFalse(result['complete'])
        self.assertTrue((self.launch/'INCOMPLETE.json').exists())
        self.assertFalse(self.output.exists())
        finished=json.loads((self.launch/'FINISHED.json').read_text())
        self.assertFalse(finished['further_execution_launched'])
        self.assertEqual(sha(self.launch/'evidence.tar.gz'),finished['archive_sha256'])
        with self.assertRaisesRegex(ValueError,'fresh'):
            supervise([sys.executable,'-c','pass'],self.launch,self.output)

    def test_costs_count_retained_prefix_once_and_preserve_api_failure(self):
        source=self.root/'source';fixture(source)
        for a in ('plain','gnomon','ledger'):
            for s in range(4):
                for n in range(26):
                    folder=source/a/f's{s}'/f'round-{n}';folder.mkdir(parents=True)
                    (folder/'api-01-forwarded.json').write_text('{}')
                    response={'usage':{'total_tokens':11}}
                    if (a,s,n)==('plain',0,0):response={'error':{'message':'synthetic'}}
                    (folder/'api-01-response.json').write_text(json.dumps(response))
        code='import shutil,sys;shutil.copytree(sys.argv[1],sys.argv[2]);print("one child")'
        result=supervise([sys.executable,'-c',code,str(source),str(self.output)],self.launch,self.output)
        self.assertTrue(result['complete'])
        cost=json.loads((self.launch/'costs.json').read_text())
        self.assertEqual(cost['deduplicated_total']['sessions'],312)
        self.assertEqual(cost['stages']['retained_pilot']['sessions'],36)
        self.assertEqual(cost['stages']['continuation']['sessions'],276)
        self.assertEqual(cost['deduplicated_total']['reported_tokens'],311*11)
        self.assertEqual(cost['deduplicated_total']['api_errors'],1)
        self.assertEqual(cost['deduplicated_total']['requests_without_reported_usage'],1)
        with tarfile.open(self.launch/'evidence.tar.gz') as tar:
            inventory=json.loads(tar.extractfile('SHA256SUMS.json').read())
            import hashlib
            for name,digest in inventory.items():
                self.assertEqual(hashlib.sha256(tar.extractfile(name).read()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
