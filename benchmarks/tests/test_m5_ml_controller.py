from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import m5_ml_controller as c
from benchmarks.ledger_optimization import m5_ml_stage_checks as stages
from benchmarks.ledger_optimization.m5_ml_terminal_prefix import archived_prefix
from benchmarks.tests import test_m5_ml_terminal_prefix as fixtures


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TerminalPrefixTests.setUpClass()

    def setUp(self):
        self.fixture = fixtures.TerminalPrefixTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.jobs = self.fixture.fixture.jobs
        self.cohort = self.fixture.fixture.cohort
        self.root, self.output = self.fixture.root, self.fixture.pilot
        fixtures.dump(self.output/'complete.json', {'completed': 72, 'planned': 72, 'source_unchanged': True})

    def complete(self, stage='pilot', status=0):
        with patch.object(stages, 'authenticated_contract', return_value=self.cohort):
            return c.completion(self.output, status, b'synthetic', json.dumps(self.jobs).encode(), stage=stage)

    def test_complete_pilot_vs_failed_quality_gate(self):
        good = self.complete()
        self.assertTrue(good['complete']);self.assertTrue(good['continuation_gate_passed'])
        rows=deepcopy(self.fixture.rows)
        for row in rows:
            if row['arm']=='ledger' and row['round']==0:
                row.update(valid=False,workflow_complete=False,fallback_used=True)
            fixtures.dump(self.output/row['arm']/row['series_id']/f"round-{row['round']}"/'grade.json',row)
        fixtures.dump(self.output/'report.json',self.fixture.fixture.report(rows))
        fixtures.dump(self.output/'GATE.json',{'passed':False,'accuracy_used_for_gate':False})
        result=self.complete()
        self.assertTrue(result['complete'])
        self.assertFalse(result['continuation_gate_passed'])
        self.assertEqual(result['stage_checks']['arms']['ledger']['fallbacks'],8)
        # A lying passed=true marker must not override the observed poor pilot.
        fixtures.dump(self.output/'GATE.json',{'passed':True,'accuracy_used_for_gate':False})
        self.assertFalse(self.complete()['complete'])

    def test_complete_624_keeps_failures_and_checks_retained_counts(self):
        rows, report=self.fixture.fixture.evidence('complete')
        rows[0].update(valid=False,workflow_complete=False,fallback_used=True)
        fixtures.dump(self.output/'report.json',self.fixture.fixture.report(rows))
        for row in rows:fixtures.dump(self.output/row['arm']/row['series_id']/f"round-{row['round']}"/'grade.json',row)
        manifest=c.read(self.output/'manifest.json');manifest.update(planned=624,pilot_sessions_retained=72,new_sessions_planned=552,accuracy_used_for_promotion=False)
        fixtures.dump(self.output/'manifest.json',manifest)
        fixtures.dump(self.output/'runner-exit.json',{'exit_status':0,'final_gate_opened':False})
        fixtures.dump(self.output/'complete.json',{'completed':624,'planned':624,'new_sessions':552,'retained_pilot_sessions':72,'source_unchanged':True})
        fixtures.dump(self.output/'host-jobs.json',self.jobs)
        result=self.complete('complete')
        self.assertTrue(result['complete']);self.assertEqual(result['sessions'],624)
        self.assertFalse(result['continuation_gate_passed']);self.assertFalse(result['final_gate_opened'])
        manifest['new_sessions_planned']=276;fixtures.dump(self.output/'manifest.json',manifest)
        self.assertFalse(self.complete('complete')['complete'])

    def test_missing_bad_scores_or_process_failures_retained(self):
        self.assertEqual(self.complete(status=9)['cause'],'stage_process_failed')
        grade=next(self.output.rglob('grade.json'));row=c.read(grade);row['rmsle']=1;fixtures.dump(grade,row)
        self.assertFalse(self.complete()['complete'])
        grade.unlink();self.assertFalse(self.complete()['complete'])

    def supervise(self, exit_status):
        launch=self.root/f'controller-{exit_status}';output=self.root/f'stage-{exit_status}';marker=self.root/f'calls-{exit_status}'
        script=self.root/'child.py'
        script.write_text('import shutil,sys\nfrom pathlib import Path\nshutil.copytree(sys.argv[1],sys.argv[2])\np=Path(sys.argv[3]);p.write_text(str(int(p.read_text())+1) if p.exists() else "1")\nraise SystemExit(int(sys.argv[4]))\n')
        command=[sys.executable,str(script),str(self.output),str(output),str(marker),str(exit_status)]
        with patch.object(stages,'authenticated_contract',return_value=self.cohort):
            result=c.supervise(command,launch,output,b'synthetic',json.dumps(self.jobs).encode(),stage='pilot')
        self.assertEqual(marker.read_text(),'1')
        self.assertTrue((launch/'FINISHED.json').exists())
        self.assertEqual(c.read(launch/'pilot-exit.json')['exit_status'],exit_status)
        self.assertEqual(c.read(launch/'command.json')['argv'],command)
        archived_prefix(output,launch)
        return result,launch,output

    def test_real_child_success_archived_and_no_second_launch(self):
        result,launch,output=self.supervise(0)
        self.assertTrue(result['complete']);self.assertFalse((launch/'INCOMPLETE.json').exists())
        account=c.read(launch/'costs.json')
        self.assertEqual(account['deduplicated_total']['sessions'],72)
        self.assertIsNone(account['billing_dollars'])
        with patch.object(stages,'authenticated_contract',return_value=self.cohort):
            with self.assertRaisesRegex(ValueError,'fresh'):
                c.supervise([sys.executable,'-c','raise AssertionError()'],launch,output,b'',json.dumps(self.jobs).encode(),stage='pilot')

    def test_real_child_failure_keeps_unreported_request_cost(self):
        folder=next(self.output.glob('*/*/round-*'))
        fixtures.dump(folder/'api-01-forwarded.json',{'model':'synthetic-no-network'})
        result,launch,output=self.supervise(9)
        self.assertFalse(result['complete']);self.assertTrue((launch/'INCOMPLETE.json').exists())
        account=c.read(launch/'costs.json')['deduplicated_total']
        self.assertEqual(account['forwarded_requests'],1)
        self.assertEqual(account['requests_without_reported_usage'],1)
        self.assertEqual(account['returned_responses'],0)
        self.assertTrue((output/folder.relative_to(self.output)/'api-01-forwarded.json').exists())

    def test_untrusted_cohort_never_starts_subprocess(self):
        with patch.object(c.subprocess,'Popen',side_effect=AssertionError('must not launch')):
            with self.assertRaises(ValueError):
                c.supervise([sys.executable,'-c','pass'],self.root/'fresh-launch',self.root/'fresh-run',b'{}',b'{}',stage='pilot')
        self.assertFalse((self.root/'fresh-launch').exists())


if __name__=='__main__':unittest.main()
