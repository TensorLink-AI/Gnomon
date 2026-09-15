import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import wait_contrast_100 as w


class WaitTests(unittest.TestCase):
    def fixture(self, root):
        config = {k: str(root/k) for k in w.FIELDS}
        for k in ('bundle', 'previous_root', 'previous_launch', 'runtime'):
            Path(config[k]).mkdir()
        Path(config['credentials_file']).write_text('MUST_NOT_BE_READ')
        payload = Path(config['bundle'])/'payload'; (payload/'code').mkdir(parents=True)
        for i, name in enumerate(('launch.json', 'development-process.json')):
            w.dump(Path(config['previous_launch'])/name, {'pid': i+10, 'boot_id': 'old', 'start_ticks': str(i+20)})
        return config, root/'state', payload

    def finish(self, config, complete=True):
        w.dump(Path(config['previous_launch'])/'FINISHED.json', {'complete': complete})

    def child(self, config, status=0):
        class Child:
            pid = 999999999
            def wait(self):
                launch = Path(config['launch_directory']); launch.mkdir()
                w.dump(launch/'FINISHED.json', {'complete': status == 0})
                return status
        return Child()

    def test_live_process_waits_then_delegates_once_without_credentials_or_inherited_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, state, payload = self.fixture(Path(tmp)); self.finish(config)
            seen = []
            def sleep(seconds): seen.append(seconds)
            def popen(command, **kwargs):
                self.assertEqual(command[3], 'benchmarks.ledger_optimization.control_contrast_100')
                self.assertEqual(command[command.index('--predecessor-audit')+1], str(Path(config['launch_directory'])/'predecessor-audit'))
                self.assertNotIn('ENGY_API_KEY', kwargs['env']); self.assertNotIn('PYTHONPATH', kwargs['env'])
                self.assertTrue((Path(config['bundle'])/'dispatch-reservation.json').exists())
                return self.child(config)
            original = Path.read_text
            def guarded(path, *args, **kwargs):
                self.assertNotEqual(path, Path(config['credentials_file']))
                return original(path, *args, **kwargs)
            with patch.object(w, 'validate_bundle', return_value=payload), patch.object(w, 'process_live', side_effect=[True,False,False,False]), patch.object(Path, 'read_text', guarded), patch.dict(os.environ, {'ENGY_API_KEY':'sentinel','PYTHONPATH':'untrusted'}):
                result=w.run(config,state,sleep=sleep,popen=popen)
            self.assertEqual(seen,[30]); self.assertEqual(result['status'],'pilot_controller_finished')
            self.assertFalse(result['continuation_launched'])

    def test_disappeared_process_without_finished_receipt_never_dispatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,state,payload=self.fixture(Path(tmp))
            with patch.object(w,'validate_bundle',return_value=payload),patch.object(w,'process_live',return_value=False),patch.object(w.subprocess,'Popen') as child:
                with self.assertRaises(FileNotFoundError):w.run(config,state,popen=child)
                child.assert_not_called()
            self.assertTrue((state/'STOPPED.json').exists())

    def test_incomplete_terminal_record_never_dispatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,state,payload=self.fixture(Path(tmp));self.finish(config,False)
            with patch.object(w,'validate_bundle',return_value=payload),patch.object(w,'process_live',return_value=False),patch.object(w.subprocess,'Popen') as child:
                with self.assertRaisesRegex(ValueError,'incomplete'):w.run(config,state,popen=child)
                child.assert_not_called()

    def test_changed_process_identity_receipt_stops_after_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,state,payload=self.fixture(Path(tmp))
            def sleep(_): (Path(config['previous_launch'])/'launch.json').write_text('{}')
            with patch.object(w,'validate_bundle',return_value=payload),patch.object(w,'process_live',return_value=True),patch.object(w.subprocess,'Popen') as child:
                with self.assertRaisesRegex(ValueError,'receipt changed'):w.run(config,state,sleep=sleep,popen=child)
                child.assert_not_called()

    def test_bundle_is_revalidated_after_wait(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,state,payload=self.fixture(Path(tmp));self.finish(config)
            with patch.object(w,'validate_bundle',side_effect=[payload,ValueError('Bundle retired')]),patch.object(w,'process_live',return_value=False),patch.object(w.subprocess,'Popen') as child:
                with self.assertRaisesRegex(ValueError,'retired'):w.run(config,state,popen=child)
                child.assert_not_called()

    def test_spawn_failure_reserves_once_and_cannot_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,state,payload=self.fixture(Path(tmp));self.finish(config)
            with patch.object(w,'validate_bundle',return_value=payload),patch.object(w,'process_live',return_value=False),patch.object(w.subprocess,'Popen',side_effect=OSError('unavailable')) as child:
                with self.assertRaises(OSError):w.run(config,state,popen=child)
                self.assertTrue((Path(config['bundle'])/'dispatch-reservation.json').exists())
                with self.assertRaises(FileExistsError):w.run(config,Path(tmp)/'second-state',popen=child)
                self.assertEqual(child.call_count,1)

    def test_controller_rejection_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,state,payload=self.fixture(Path(tmp));self.finish(config)
            with patch.object(w,'validate_bundle',return_value=payload),patch.object(w,'process_live',return_value=False),patch.object(w.subprocess,'Popen',return_value=self.child(config,1)) as child:
                with self.assertRaisesRegex(ValueError,'no retry'):w.run(config,state,popen=child)
                self.assertEqual(child.call_count,1)
            self.assertEqual(w.read(state/'controller-exit.json')['exit_status'],1)

    def test_output_overlap_and_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config,state,_=self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'overlaps'):w.validate_paths(config,Path(config['bundle'])/'state')
            state.mkdir()
            with self.assertRaisesRegex(ValueError,'fresh'):w.validate_paths(config,state)

    def test_pid_reuse_boot_change_zombie_and_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc=Path(tmp);(proc/'10').mkdir();(proc/'sys/kernel/random').mkdir(parents=True)
            (proc/'sys/kernel/random/boot_id').write_text('boot')
            def stat(state,ticks): (proc/'10/stat').write_text('10 (process name) '+' '.join([state]+['0']*18+[str(ticks)]))
            record={'pid':10,'boot_id':'boot','start_ticks':'20'}
            stat('S',20);self.assertTrue(w.process_live(record,proc))
            stat('S',21);self.assertFalse(w.process_live(record,proc))
            stat('Z',20);self.assertFalse(w.process_live(record,proc))
            stat('S',20);self.assertFalse(w.process_live({**record,'boot_id':'old'},proc))
            (proc/'10/stat').unlink();self.assertFalse(w.process_live(record,proc))

    @unittest.skipUnless(Path('results/contrast-100-dispatch-bundle-002/payload').exists(),'requires staged immutable bundle')
    def test_real_frozen_bundle_validates(self):
        self.assertEqual(w.validate_bundle(Path('results/contrast-100-dispatch-bundle-002')).name,'payload')

    @unittest.skipUnless(Path('results/contrast-100-dispatch-bundle-002/payload').exists(),'requires staged immutable bundle')
    def test_manifest_and_payload_cannot_be_changed_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=Path(tmp)/'bundle';bundle.mkdir()
            source=Path('results/contrast-100-dispatch-bundle-002')
            shutil.copyfile(source/'dispatch.tar.gz',bundle/'dispatch.tar.gz')
            shutil.copytree(source/'payload',bundle/'payload')
            target=bundle/'payload/code/benchmarks/ledger_optimization/control_contrast_100.py'
            target.write_text('raise SystemExit(0)\n')
            inventory=bundle/'payload/SHA256SUMS.json';values=w.read(inventory)
            values[str(target.relative_to(bundle/'payload'))]=w.sha(target)
            inventory.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError,'pinned archive'):w.validate_bundle(bundle)

    def test_retired_bundle_never_loads_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'RETIRED.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'retired'):w.validate_bundle(root)


if __name__=='__main__':unittest.main()
