"""Execution-boundary regressions: rejected inputs never reach a subprocess."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmarks.ledger_optimization.execution_boundary_093 import LabBoundary, BoundaryRejected


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        (self.work/'lab.py').write_text("print('trusted lab')\n")
        (self.work/'history.csv').write_text('timestamp,value,promo\n2026-01-01,0,0\n2026-01-02,2,1\n2026-01-03,4,1\n')
        self.hashes = {'lab.py': hashlib.sha256((self.work/'lab.py').read_bytes()).hexdigest()}
        self.calls = []
        def runner(argv, **kwargs):
            self.calls.append((argv,kwargs));return subprocess.CompletedProcess(argv,0,'{}','')
        self.boundary = LabBoundary(self.work,sys.executable,self.hashes,runner=runner)

    def reject_without_execution(self, tool, args):
        before=len(self.calls);result=self.boundary.dispatch(tool,args)
        self.assertEqual(result['status'],'error');self.assertFalse(result['execution_started'])
        self.assertEqual(len(self.calls),before)

    def test_observed_direct_model_search_rejected(self):
        self.reject_without_execution('terminal',{'command':'python local_probe.py'})
        self.reject_without_execution('terminal',{'command':'python -c "from numerical import predict; predict(...)"'})
        self.reject_without_execution('write_file',{'path':'local_probe.py','content':'from numerical import predict'})

    def test_other_dispatch_paths_fail_closed(self):
        for tool in ['delegate_task','python','mcp','process','browser','skill_manage','shell']:
            with self.subTest(tool=tool):self.reject_without_execution(tool,{'command':'echo bypass'})

    def test_metered_operations_use_fixed_interpreter_no_shell(self):
        result=self.boundary.dispatch('lab',{'operation':'backtest','config':{'model':'ridge','window':365,'lags':14,'alpha':10}})
        self.assertEqual(result['status'],'ok');argv,kwargs=self.calls[0]
        self.assertEqual(argv[1:4],['-I','-B','-c']);self.assertNotIn('shell',kwargs)
        self.assertEqual(argv[6],'backtest');self.assertEqual(json.loads(argv[8])['model'],'ridge')
        self.assertNotIn('PYTHONPATH',kwargs['env']);self.assertNotIn('ENGY_API_KEY',kwargs['env'])

    def test_shell_injection_is_never_evaluated(self):
        marker=self.work/'owned'
        self.boundary.dispatch('lab',{'operation':'commit','execution_id':f'$(touch {marker})'})
        self.assertFalse(marker.exists());self.assertIn('$(touch ',self.calls[0][0][-1])
        self.reject_without_execution('lab',{'operation':'status; touch owned'})
        self.reject_without_execution('lab',{'operation':'status','env':{'PYTHONPATH':'.'}})

    def test_review_pair_preserves_both_provider_identifiers(self):
        result=self.boundary.dispatch('lab',{'operation':'review','pair':['ridge_a','seasonal_b']})
        self.assertEqual(result['status'],'ok')
        self.assertEqual(self.calls[-1][0][-3:],['--pair','ridge_a','seasonal_b'])
        self.reject_without_execution('lab',{'operation':'review','pair':'ridge_a seasonal_b'})

    def test_bad_types_duplicates_and_options_rejected(self):
        for args in [[],{'operation':'backtest'}, {'operation':'status','config':{}},
                     {'operation':'review','limit':True},{'operation':'review','limit':101},
                     {'operation':'review','offset':-1},{'operation':'commit','execution_id':'--help'}]:
            self.reject_without_execution('lab',args)

    def test_changed_source_and_import_shadowing_stop_execution(self):
        (self.work/'lab.py').write_text("raise RuntimeError('changed')")
        self.reject_without_execution('lab',{'operation':'start'})
        (self.work/'lab.py').write_text("print('trusted lab')\n")
        (self.work/'sklearn.py').write_text('print("shadow")')
        self.reject_without_execution('lab',{'operation':'start'})

    def test_model_running_failure_is_not_reported_as_zero_execution(self):
        def runner(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],60)
        boundary=LabBoundary(self.work,sys.executable,self.hashes,runner=runner)
        result=boundary.dispatch('lab',{'operation':'start'})
        self.assertEqual(result['status'],'error');self.assertTrue(result['execution_started'])

    def test_traversal_symlinks_fifo_and_hardlinked_writes(self):
        self.reject_without_execution('evidence_read',{'path':'../secret'})
        self.reject_without_execution('evidence_read',{'path':'/etc/passwd'})
        (self.work/'link').symlink_to(self.work/'history.csv')
        self.reject_without_execution('evidence_read',{'path':'link'})
        os.mkfifo(self.work/'fifo');self.reject_without_execution('evidence_read',{'path':'fifo'})
        (self.work/'notes').mkdir();os.link(self.work/'lab.py',self.work/'notes'/'protected.md')
        self.reject_without_execution('notes_write',{'path':'notes/protected.md','text':'overwrite'})
        self.assertEqual((self.work/'lab.py').read_text(),"print('trusted lab')\n")

    def test_notes_cannot_create_or_modify_executable_or_evidence(self):
        for name in ['lab.py','numerical.py','agent-budget.json','checkpoint.json','notes/startup.py','notes/../lab.py']:
            self.reject_without_execution('notes_write',{'path':name,'text':'override'})
        self.assertEqual(self.boundary.dispatch('notes_write',{'path':'notes/lesson.md','text':'dated lesson'})['status'],'ok')
        self.assertEqual(self.boundary.dispatch('notes_write',{'path':'decision.json','text':'{"rationale":"test"}'})['status'],'ok')

    def test_raw_evidence_pages_reassemble_and_hash(self):
        pieces=[];offset=0
        while True:
            r=self.boundary.dispatch('evidence_read',{'path':'history.csv','offset':offset,'max_chars':13})['result'];pieces.append(r['text'])
            if r['next_offset'] is None:break
            offset=r['next_offset']
        self.assertEqual(''.join(pieces),(self.work/'history.csv').read_text())
        self.assertEqual(r['sha256'],hashlib.sha256((self.work/'history.csv').read_bytes()).hexdigest())
        self.assertFalse(self.calls)

    def test_data_inspection_preserves_explicit_window_and_groups(self):
        r=self.boundary.dispatch('data_summary',{'column':'value','start_row':1,'end_row':3,'group_by':'promo'})['result']
        self.assertEqual(r['groups']['1'],{'n':2,'mean':3,'minimum':2,'maximum':4,'zero_count':0})
        self.assertEqual(r['model_fits'],0);self.assertFalse(self.calls)

    def test_real_isolated_bootstrap_executes_only_manifested_lab(self):
        boundary=LabBoundary(self.work,sys.executable,self.hashes)
        r=boundary.dispatch('lab',{'operation':'status'})['result']
        self.assertEqual(r['exit_code'],0);self.assertEqual(r['stdout'],'trusted lab\n')
        self.assertFalse((self.work/'__pycache__').exists())

    def test_symlinked_virtualenv_interpreter_retains_its_environment(self):
        import venv
        with tempfile.TemporaryDirectory() as environment:
            venv.EnvBuilder(with_pip=False,symlinks=True).create(environment)
            python=Path(environment)/'bin/python'
            (self.work/'lab.py').write_text('import sys,json;print(json.dumps({"prefix":sys.prefix}))\n')
            hashes={'lab.py':hashlib.sha256((self.work/'lab.py').read_bytes()).hexdigest()}
            boundary=LabBoundary(self.work,python,hashes)
            result=boundary.dispatch('lab',{'operation':'status'})['result']
            self.assertEqual(result['exit_code'],0,result)
            self.assertEqual(json.loads(result['stdout'])['prefix'],environment)



class RealLabBoundaryTests(unittest.TestCase):
    def test_real_model_search_cannot_exceed_sixty_metered_attempts(self):
        import csv
        from datetime import date, timedelta
        import shutil
        import time
        from benchmarks.hermes_ml_checkpoint_v5.run import PROJECT_FILES
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory);source=Path(__file__).resolve().parents[1]/'hermes_ml_checkpoint_v5'
            hashes={}
            for name in PROJECT_FILES:
                shutil.copyfile(source/name,work/name)
                hashes[name]=hashlib.sha256((work/name).read_bytes()).hexdigest()
            dates=[(date(2020,1,1)+timedelta(days=i)).isoformat()+'T00:00:00+00:00' for i in range(744)]
            with (work/'history.csv').open('w') as stream:
                writer=csv.writer(stream);writer.writerow(['timestamp','value','promo'])
                writer.writerows((dates[i],1+i%7,i%2) for i in range(730))
            with (work/'future.csv').open('w') as stream:
                writer=csv.writer(stream);writer.writerow(['timestamp','promo'])
                writer.writerows((dates[i],i%2) for i in range(730,744))
            for name,value in {'task.json':{'origin':dates[729],'series_id':'synthetic','unit':'widgets','future_timestamps':dates[730:]},
                               'backend.json':{'arm':'plain'},'previous_runs.json':[],
                               'agent-budget.json':{'forwarded_requests':1,'remaining_requests':15,'deadline_epoch':time.time()+600}}.items():
                (work/name).write_text(json.dumps(value))
            boundary=LabBoundary(work,sys.executable,hashes)
            def lab(**args):
                result=boundary.dispatch('lab',args);self.assertEqual(result['status'],'ok',result)
                self.assertEqual(result['result']['stderr'],'',result)
                return json.loads(result['result']['stdout'])
            self.assertEqual(lab(operation='start')['status'],'ok')
            config={'model':'ridge','window':90,'lags':7,'alpha':10}
            self.assertEqual(lab(operation='backtest',config=config)['status'],'ok')
            self.assertEqual(lab(operation='commit',config=config)['status'],'ok')
            original=(work/'checkpoint.json').read_bytes()
            self.assertEqual(boundary.dispatch('terminal',{'command':'python -c "from numerical import predict"'})['status'],'error')
            self.assertEqual(original,(work/'checkpoint.json').read_bytes())
            for i in range(17):
                config={**config,'alpha':100+i}
                self.assertEqual(lab(operation='backtest',config=config)['status'],'ok')
            blocked=lab(operation='backtest',config={**config,'alpha':500})
            self.assertEqual(blocked['status'],'error')
            self.assertEqual(lab(operation='commit',config=config)['status'],'ok')
            at_limit=lab(operation='status')['result']
            self.assertEqual(at_limit['budget']['numerical_attempts'],60)
            self.assertTrue(at_limit['checkpoint']['selection_after_comparison'])
            self.assertEqual(lab(operation='backtest',config={**config,'alpha':600})['status'],'error')
            self.assertEqual(lab(operation='status')['result']['budget']['numerical_attempts'],60)
            events=[json.loads(line) for line in (work/'experiments.jsonl').read_text().splitlines()]
            attempts=[x for x in events if x['event']=='attempt'];results=[x for x in events if x['event']=='result']
            self.assertEqual(len(attempts),60);self.assertEqual(len(results),60)
            self.assertEqual({x['attempt_id'] for x in attempts},{x['attempt_id'] for x in results})


if __name__=='__main__':unittest.main()
