from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import m5_ml_terminal_prefix as t
from benchmarks.ledger_optimization.control_collection_096 import archive_evidence
from benchmarks.ledger_optimization import m5_ml_stage_checks as stages
from benchmarks.tests import test_m5_ml_stage_checks as stage_fixtures


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, sort_keys=True))


class TerminalPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stage_fixtures.StageCoverageTests.setUpClass()
        cls.fixture = stage_fixtures.StageCoverageTests()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pilot, self.launch, self.capsule = (self.root/n for n in ('pilot', 'launch', 'capsule'))
        self.proc = self.root/'proc'; self.proc.mkdir()
        self.plan = self.root/'plan.json'
        self.runtime = {'plain': {'python': '3.12'}, 'gnomon': {'gnomon': '1.2.0'}}
        source = self.capsule/'benchmarks/hermes_ml_checkpoint_v6/run.py'
        source.parent.mkdir(parents=True); source.write_text('# synthetic capsule, never executed\n')
        self.cap = {'requested_seed': 19, 'sources': {'run.py': t.digest(source)}}
        dump(self.capsule/'capsule.json', self.cap)
        plan = {'capsule_sha256': t.digest(self.capsule/'capsule.json'), 'capsule': self.cap,
                'task_source_sha256': stages.DEVELOPMENT_JOBS_SHA,
                'arms': ['plain', 'gnomon', 'ledger'], 'requested_seed': 19,
                'runtime_inventory': self.runtime,
                'planned': {'pilot_sessions': 72, 'continuation_sessions': 552, 'total_sessions': 624},
                'final_gate_opened': False}
        dump(self.plan, plan)
        dump(self.pilot/'pilot-plan.json', plan)
        dump(self.pilot/'accepted-launch.json', {'plan_sha256': t.digest(self.plan)})
        dump(self.pilot/'manifest.json', {'requested_seed': 19, 'sources': self.cap['sources'],
             'inventory': self.runtime, 'planned': 72, 'source_jobs_sha256': stages.DEVELOPMENT_JOBS_SHA})
        dump(self.pilot/'final-runtime-inventory.json', self.runtime)
        dump(self.pilot/'runner-exit.json', {'exit_status': 0, 'continuation_launched': False})
        dump(self.pilot/'GATE.json', {'passed': True, 'accuracy_used_for_gate': False})
        self.rows, report = self.fixture.evidence('pilot')
        dump(self.pilot/'report.json', report)
        for row in self.rows:
            dump(self.pilot/row['arm']/row['series_id']/f"round-{row['round']}"/'grade.json', row)
        dump(self.pilot/'host-jobs.json', {s: v[:3] for s, v in self.fixture.jobs.items()})
        for name, pid in [('launch.json', 100), ('pilot-process.json', 101)]:
            dump(self.launch/name, {'pid': pid, 'start_ticks': '123', 'boot_id': 'boot'})
        dump(self.launch/'pilot-exit.json', {'exit_status': 0})
        dump(self.launch/'AUDITED.json', {'complete': True, 'continuation_gate_passed': True})
        self.archive()

    def archive(self):
        for name in ('SHA256SUMS.json', 'evidence.tar.gz', 'FINISHED.json'):
            (self.launch/name).unlink(missing_ok=True)
        result = archive_evidence(self.launch, self.pilot, [])
        dump(self.launch/'FINISHED.json', {**result, 'complete': True,
             'continuation_gate_passed': True, 'continuation_launched': False})

    def check(self):
        # Only the fixed production input-hash authenticator is mocked. The
        # real 72-row cohort, score, seed, source, archive and process checks run.
        with patch.object(stages, 'authenticated_contract', return_value=self.fixture.cohort):
            return t.verify_terminal_prefix(self.pilot, self.launch, self.plan, self.capsule,
                self.runtime, b'synthetic manifest', json.dumps(self.fixture.jobs).encode(), proc=self.proc)

    def test_full_72_prefix_accepted_without_dispatch(self):
        before = t.inventory(self.pilot)
        result = self.check()
        self.assertTrue(result['terminal_prefix_checks_passed'])
        self.assertEqual(result['prefix_checks']['sessions'], 72)
        self.assertFalse(result['execution_authorized'])
        self.assertEqual(t.inventory(self.pilot), before)

    def test_live_controller_or_child_rejected(self):
        for pid in (100, 101):
            p = self.proc/str(pid)/'stat'; p.parent.mkdir(exist_ok=True)
            fields = ['S'] + ['0']*18 + ['123']
            p.write_text(f'{pid} (process name) ' + ' '.join(fields))
            boot = self.proc/'sys/kernel/random/boot_id'; boot.parent.mkdir(parents=True, exist_ok=True); boot.write_text('boot')
            with self.subTest(pid=pid), self.assertRaisesRegex(ValueError, 'still live'):
                self.check()
            p.unlink()

    def test_reused_pid_is_not_old_live_process(self):
        p = self.proc/'100/stat'; p.parent.mkdir()
        p.write_text('100 (new process) ' + ' '.join(['S']+['0']*18+['999']))
        boot = self.proc/'sys/kernel/random/boot_id'; boot.parent.mkdir(parents=True); boot.write_text('boot')
        self.assertTrue(self.check()['terminal_prefix_checks_passed'])

    def test_changed_or_missing_pilot_file_rejected(self):
        p = self.pilot/'agent-memory.txt'; p.write_text('unarchived memory')
        with self.assertRaisesRegex(ValueError, 'terminal inventory'): self.check()
        p.unlink()
        grade = next(self.pilot.glob('*/*/round-*/grade.json')); grade.unlink()
        self.archive()  # Even a self-consistent smaller archive cannot pass.
        with self.assertRaisesRegex(ValueError, 'Missing stage tasks'): self.check()

    def test_wrong_seed_and_plan_rejected_even_when_rearchived(self):
        p = self.pilot/'manifest.json'; m = t.read(p); m['requested_seed'] = 7; dump(p,m); self.archive()
        with self.assertRaisesRegex(ValueError, 'seed'): self.check()
        m['requested_seed'] = 19; dump(p,m); self.archive()
        plan=t.read(self.plan); plan['requested_seed']=7; dump(self.plan,plan)
        with self.assertRaisesRegex(ValueError, 'plan'): self.check()

    def test_extra_controller_evidence_and_incomplete_marker_rejected(self):
        p=self.launch/'late.log'; p.write_text('late unarchived output')
        with self.assertRaisesRegex(ValueError, 'unarchived'): self.check()
        p.unlink(); dump(self.launch/'INCOMPLETE.json', {'complete':False}); self.archive()
        with self.assertRaisesRegex(ValueError, 'Incomplete'): self.check()

    def test_corrupt_or_omitted_archive_member_rejected_even_with_new_archive_hash(self):
        archive=self.launch/'evidence.tar.gz'
        with tarfile.open(archive,'r:gz') as f:
            data=[(m.name,f.extractfile(m).read()) for m in f]
        for mode in ('missing','duplicate','wrong_bytes','link'):
            with tarfile.open(archive,'w:gz') as f:
                rows=data[1:] if mode=='missing' else data+data[:1] if mode=='duplicate' else data
                for i,(name,raw) in enumerate(rows):
                    if mode=='wrong_bytes' and i==0: raw=b'changed'
                    m=tarfile.TarInfo(name);m.size=len(raw)
                    if mode=='link' and i==0:m.type=tarfile.SYMTYPE;m.linkname='/tmp/unrelated';m.size=0
                    f.addfile(m,io.BytesIO(raw))
            finished=t.read(self.launch/'FINISHED.json');finished['archive_sha256']=t.digest(archive);dump(self.launch/'FINISHED.json',finished)
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.check()

    def test_symlinks_and_wrong_grade_location_rejected(self):
        p=self.pilot/'link';p.symlink_to(self.plan)
        with self.assertRaisesRegex(ValueError, 'Links'):self.check()
        p.unlink();grade=next(self.pilot.glob('*/*/round-*/grade.json'));r=t.read(grade);r['series_id']='other';dump(grade,r);self.archive()
        with self.assertRaisesRegex(ValueError,'location'):self.check()

    def test_misplaced_extra_grade_cannot_hide_outside_session_glob(self):
        dump(self.pilot/'unexpected/nested/grade.json', self.rows[0]); self.archive()
        with self.assertRaisesRegex(ValueError, 'location'): self.check()

    def test_untrusted_real_inputs_still_fail_hash_authentication(self):
        with self.assertRaises(ValueError):
            t.verify_terminal_prefix(self.pilot,self.launch,self.plan,self.capsule,self.runtime,
                                     b'{}',json.dumps(self.fixture.jobs).encode(),proc=self.proc)


if __name__ == '__main__':
    unittest.main()
