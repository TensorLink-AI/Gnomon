from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import launch_m5_ml as launch


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value))


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.plan={'runtime_inventory':{'synthetic':True},'build':{'package_version':'1.2.0'}}
        self.proof={'status':'m5_integrated_host_preflight_passed','passed':True,'engy_calls':0,
                    'host_sources':launch.source_identity(),'plan_hashes':{str(k):v for k,v in launch.PLAN_HASHES.items()},
                    'runtime_inventory':self.plan['runtime_inventory'],'build':self.plan['build'],
                    'requested_seeds':[7,19],'series_per_seed':8,'pilot_sessions_per_seed':72,
                    'resumed_sessions_per_seed':552,'full_workflows_per_seed':624,'parallel_series':2,
                    'original_prefix_unchanged':True,'copied_prefix_audit_passed':True,
                    'cross_seed_state_isolated':True,'reservation_reentry_rejected':True,
                    'checks':[{'passed':True}]}

    def test_exact_host_preflight_required(self):
        path=self.root/'proof.json';dump(path,self.proof)
        self.assertEqual(launch.verify_host_preflight(path,self.plan),self.proof)
        for field,value in [('series_per_seed',2),('requested_seeds',[7]),('pilot_sessions_per_seed',18),
                            ('resumed_sessions_per_seed',6),('host_sources',{}),('plan_hashes',{}),
                            ('engy_calls',1),('checks',[]),('original_prefix_unchanged',False),
                            ('cross_seed_state_isolated',False),('reservation_reentry_rejected',False),
                            ('copied_prefix_audit_passed',False),('parallel_series',1)]:
            bad=deepcopy(self.proof);bad[field]=value;dump(path,bad)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'preflight required'):
                launch.verify_host_preflight(path,self.plan)

    def test_output_cannot_nest_in_pilot_or_runtime(self):
        args=SimpleNamespace(output=self.root/'output',launch=self.root/'launch',runtime=self.root/'runtime',
                             capsule=self.root/'capsule',pilot_root=self.root/'pilot')
        launch.validate_paths(args)
        for root in ('runtime','capsule','pilot_root'):
            bad=deepcopy(args);bad.output=getattr(args,root)/'child'
            with self.subTest(root=root),self.assertRaises(ValueError):launch.validate_paths(bad)
        args.launch=args.output/'child'
        with self.assertRaises(ValueError):launch.validate_paths(args)

    def test_bad_copied_audit_precedes_credential_access(self):
        class ForbiddenCredential:
            def read_text(self):raise AssertionError('credentials were accessed')
        run=SimpleNamespace(dump=dump,ARMS=['plain','gnomon','ledger'],STOP=SimpleNamespace(set=lambda:None))
        helper=SimpleNamespace(run=run)
        args=SimpleNamespace(output=self.root/'new',runtime=self.root/'runtime',stage='complete',
                             pilot_root=self.root/'pilot',credentials_file=ForbiddenCredential())
        dump(args.pilot_root/'manifest.json',{'planned':72})
        with patch.object(launch,'prepare_copy',side_effect=ValueError('copied audit failed')):
            with self.assertRaisesRegex(ValueError,'copied audit failed'):
                launch.execute_stage(args,helper,{},b'{}',b'{}',{'pilot_files':{}})
        self.assertFalse(args.output.exists())

    def test_pilot_manifest_stamped_and_gate_computed_after_worker(self):
        calls=[];package=self.root/'package';package.mkdir();(package/'run.py').write_text('# synthetic')
        class Credential:
            def read_text(self):calls.append('credential');return 'ENGY_API_KEY=synthetic-test-only\n'
        run=SimpleNamespace(dump=dump,ARMS=['plain','gnomon','ledger'],HERE=package,
                            STOP=SimpleNamespace(set=lambda:None))
        args=SimpleNamespace(output=self.root/'output',runtime=self.root/'runtime',stage='pilot',
                             launch=self.root/'launch',credentials_file=Credential(),
                             plan=self.root/'plan.json',host_preflight=self.root/'proof.json',reservation=self.root/'reserve.json')
        for p in (args.plan,args.host_preflight,args.reservation):dump(p,{})
        plan={'requested_seed':19,'runtime_inventory':{},'capsule':{'sources':{'run.py':launch.inputs.sha(package/'run.py')}}}
        def main():
            calls.append('worker');args.output.mkdir()
            run.dump(args.output/'manifest.json',{'planned':72})
            run.key()
        run.main=main
        helper=SimpleNamespace(run=run,analyze=lambda root:{'synthetic':True})
        with patch.object(launch.stages,'check_development_stage',return_value={'pilot_quality_passed':False}),patch.object(launch.sys,'argv',['original']):
            launch.execute_stage(args,helper,plan,b'{}',b'{}',None)
        self.assertEqual(calls,['worker','credential'])
        self.assertEqual(launch.read(args.output/'manifest.json')['requested_seed'],19)
        self.assertFalse(launch.read(args.output/'GATE.json')['passed'])
        self.assertEqual(launch.read(args.output/'runner-exit.json'),{'exit_status':0,'continuation_launched':False})

    def test_predecessor_live_process_is_not_treated_as_terminal(self):
        root=self.root/'predecessor';controller=self.root/'controller';parent=self.root/'parent.json'
        dump(parent,{})
        for name in ('launch.json','development-process.json'):dump(controller/name,{'pid':1})
        with patch.object(launch.inputs,'PARENT_PLAN_SHA',launch.inputs.sha(parent)),patch.object(launch.terminal,'terminal',side_effect=ValueError('still live')):
            with self.assertRaisesRegex(ValueError,'still live'):launch.verify_predecessor(root,controller,parent)

    def test_worker_reservation_cannot_be_consumed_twice(self):
        registry=self.root/'registry';registry.mkdir();plan_path=self.root/'plan.json';dump(plan_path,{})
        args=SimpleNamespace(reservation=registry/'reservation.json',plan=plan_path,stage='pilot',
                             output=self.root/'output',launch=self.root/'launch')
        dump(args.reservation,{'plan_sha256':launch.inputs.sha(plan_path),'stage':'pilot',
                               'output':str(args.output),'controller':str(args.launch)})
        plan={'dispatch_registry':str(registry)}
        launch.consume_reservation(args,plan)
        with self.assertRaises(FileExistsError):launch.consume_reservation(args,plan)


if __name__=='__main__':unittest.main()
