"""Continuation gates, exact state transfer, arm order and maturation; no APIs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import continue_guarded_093 as c


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def jobs():
    return [{'series_id':'synthetic', 'round':i, 'origin':f'2020-01-{i+1:02d}T00:00:00+00:00',
        'request':{'unit':'widgets'}, 'actual':[i], 'future_timestamps':[f'target-{i}'],
        'outcome_recorded_at':f'2020-01-{i+2:02d}T00:00:00+00:00'} for i in range(5)]


def prefix(root, tasks):
    for arm in c.run.ARMS:
        for job in tasks[:3]:
            dump(root/arm/'synthetic'/f'round-{job["round"]}'/'grade.json',
                {**{k:job[k] for k in ('origin','series_id','round')},'arm':arm,
                 'point':[7], 'execution_id':arm+str(job['round']),
                 'config':{'model':'seasonal','season':7}, 'fallback_used':False,'rmsle':.2})


class ContinuationTests(unittest.TestCase):
    def test_process_identity_requires_actual_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity={'pid':3,'boot_id':'boot','start_ticks':'88'}
            p=root/'3/stat';p.parent.mkdir();(root/'sys/kernel/random').mkdir(parents=True)
            (root/'sys/kernel/random/boot_id').write_text('boot\n')
            fields=['S']+['0']*18+['88'];p.write_text('3 (a name) '+' '.join(fields))
            with self.assertRaisesRegex(ValueError,'still live'):c.require_terminal(identity,root)
            fields[0]='Z';p.write_text('3 (a name) '+' '.join(fields));c.require_terminal(identity,root)
            fields[0]='S';fields[19]='89';p.write_text('3 (a name) '+' '.join(fields));c.require_terminal(identity,root)
            p.unlink();c.require_terminal(identity,root)

    def test_state_copy_preserves_original_and_retires_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'pilot';source.mkdir()
            dump(source/'complete.json',{'completed':36});dump(source/'manifest.json',{'planned':36})
            dump(source/'plain/series/round-0/grade.json',{'valid':False})
            dump(source/'plain/series/home/memory.json',{'lesson':'keep exact'})
            db=source/'plain/series/work/ledger.db';db.parent.mkdir();db.write_bytes(b'opaque-state')
            before=c.inventory(source);copied=root/'continuation'
            c.copy_prefix(source,copied,before)
            self.assertEqual(c.inventory(source),before)
            self.assertFalse((copied/'complete.json').exists())
            self.assertEqual(c.read(copied/'pilot-metadata/complete.json'),{'completed':36})
            self.assertEqual((copied/'plain/series/work/ledger.db').read_bytes(),b'opaque-state')
            self.assertEqual(c.read(copied/'plain/series/round-0/grade.json'),{'valid':False})
            with self.assertRaisesRegex(ValueError,'fresh'):c.copy_prefix(source,copied,before)

    def test_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'link').symlink_to('/etc/passwd')
            with self.assertRaisesRegex(ValueError,'Non-regular'):c.inventory(root)

    def test_chains_retain_failed_prefix_and_absolute_rotating_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tasks=jobs();prefix(root,tasks);calls=[]
            p=root/'plain/synthetic/round-1/grade.json';g=c.read(p)
            g.update(fallback_used=True,execution_id=None,config=None);dump(p,g)
            def execute(out,arm,series,job,prior,runtime,key):
                calls.append((arm,job['round'],len(prior),[v['execution_id'] for v in prior]))
                self.assertEqual([x['round'] for x in tasks[:3]],[0,1,2])
                if arm=='plain':self.assertIsNone(prior[1]['execution_id'])
                prior.append({'execution_id':arm+str(job['round']),
                    'outcome_recorded_at':job['outcome_recorded_at']})
                return {'arm':arm,'round':job['round']}
            with patch.object(c.run,'execute',execute):
                c.run.STOP.clear();result=c.continue_chain(1,'synthetic',tasks,root,
                    {a:Path('/python') for a in c.run.ARMS},'synthetic')
            self.assertEqual([(a,n) for a,n,_,_ in calls],
                [('gnomon',3),('ledger',3),('plain',3),('ledger',4),('plain',4),('gnomon',4)])
            self.assertEqual([n for _,_,n,_ in calls],[3,3,3,4,4,4]);self.assertEqual(len(result),6)

    def test_immature_outcomes_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tasks=jobs();tasks[2]['outcome_recorded_at']='2099-01-01T00:00:00+00:00'
            prefix(root,tasks)
            with patch.object(c.run,'execute') as execute:
                c.run.STOP.clear()
                with self.assertRaisesRegex(ValueError,'immature'):
                    c.continue_chain(0,'synthetic',tasks,root,{a:Path('/python') for a in c.run.ARMS},'synthetic')
                execute.assert_not_called()

    def test_failed_gate_refuses_credentials_and_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('sys.argv',['continue','--pilot-root',str(root/'pilot'),
                    '--pilot-launch',str(root/'launch'),'--output',str(root/'out'),
                    '--continuation-preflight',str(root/'preflight.json')]),\
                 patch.object(c,'jobs_from_source',return_value={}),\
                 patch.object(c,'verify_gate',side_effect=ValueError('Pilot completion gate did not pass')),\
                 patch.object(c.run,'key') as key,patch.object(c,'copy_prefix') as copy:
                with self.assertRaisesRegex(ValueError,'did not pass'):c.main()
                key.assert_not_called();copy.assert_not_called();self.assertFalse((root/'out').exists())

    def test_gate_rechecks_threshold_and_terminal_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot=root/'pilot';launch=root/'launch';launch.mkdir()
            sources={p.name:c.digest(p) for p in c.run.HERE.iterdir() if p.is_file()}
            tasks={f's{i}':[] for i in range(4)}
            for series in tasks:
                for j in jobs()[:3]:
                    tasks[series].append({**j,'series_id':series})
                    for arm in c.run.ARMS:
                        dump(pilot/arm/series/f'round-{j["round"]}'/'grade.json',
                            {'arm':arm,'series_id':series,'round':j['round'],'workflow_complete':True})
            dump(pilot/'host-jobs.json',tasks)
            dump(pilot/'manifest.json',{'sources':sources,'source_jobs_sha256':c.run.SOURCE_SHA,'planned':36})
            dump(pilot/'report.json',{'complete':True,'audit_failures':[],
                'arms':{a:{'workflow_complete':12} for a in c.run.ARMS}})
            dump(launch/'GATE.json',{'passed':True});dump(launch/'pilot-exit.json',{'exit_status':0})
            dump(launch/'launch.json',{'pid':999999999,'start_ticks':'1','boot_id':'synthetic','source_hashes':sources})
            dump(launch/'pilot-process.json',{'pid':999999998,'start_ticks':'1'})
            (launch/'evidence.tar.gz').write_bytes(b'synthetic archive')
            dump(launch/'FINISHED.json',{'pilot_exit_status':0,'archive_sha256':c.digest(launch/'evidence.tar.gz')})
            dump(launch/'SHA256SUMS.json',c.inventory(pilot))
            files,_,_=c.verify_gate(pilot,launch,tasks);self.assertEqual(files,c.inventory(pilot))
            (pilot/'added-state').write_text('unexpected')
            with self.assertRaisesRegex(ValueError,'terminal inventory'):c.verify_gate(pilot,launch,tasks)
            (pilot/'added-state').unlink()
            for n in (0,1):
                p=pilot/'plain/s0'/f'round-{n}'/'grade.json';g=c.read(p);g['workflow_complete']=False;dump(p,g)
            with self.assertRaisesRegex(ValueError,'threshold failed'):c.verify_gate(pilot,launch,tasks)


if __name__=='__main__':unittest.main()
