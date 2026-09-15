"""Reject stale/failed collection pilot receipts before continuation or spending."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import continue_collection_096 as c
from benchmarks.ledger_optimization import continue_guarded_093 as helper


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ContinuationCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pilot, self.launch = self.root/'pilot', self.root/'launch'
        self.launch.mkdir()
        self.plan = self.root/'plan.json'
        dump(self.plan, {'frozen': True})
        self.sources = {'synthetic.py': 'synthetic'}
        self.jobs = {f's{i}': [{'series_id': f's{i}', 'round': j} for j in range(26)] for i in range(4)}
        for arm in helper.run.ARMS:
            for series in self.jobs:
                for number in range(3):
                    dump(self.pilot/arm/series/f'round-{number}/grade.json', {
                        'arm': arm, 'series_id': series, 'round': number,
                        'valid': True, 'workflow_complete': True})
        dump(self.pilot/'host-jobs.json', {s: j[:3] for s, j in self.jobs.items()})
        dump(self.pilot/'manifest.json', {'sources': self.sources, 'planned': 36,
            'source_jobs_sha256': helper.run.SOURCE_SHA})
        dump(self.pilot/'report.json', {'complete': True, 'audit_failures': [], 'shutdown_record_gaps': [],
            'arms': {a: {'workflow_complete': 12} for a in helper.run.ARMS}})
        dump(self.pilot/'GATE.json', {'passed': True, 'accuracy_used_for_gate': False})
        dump(self.pilot/'runner-exit.json', {'exit_status': 0, 'continuation_launched': False})
        dump(self.pilot/'accepted-launch.json', {'plan_sha256': c.sha(self.plan)})
        (self.pilot/'pilot-plan.json').write_bytes(self.plan.read_bytes())
        for name in ('launch.json', 'pilot-process.json'):
            dump(self.launch/name, {'pid': 999999998, 'boot_id': 'synthetic', 'start_ticks': '1'})
        dump(self.launch/'pilot-exit.json', {'exit_status': 0})
        dump(self.launch/'AUDITED.json', {'complete': True})
        (self.launch/'evidence.tar.gz').write_bytes(b'synthetic archive')
        self.seal()

    def seal(self):
        inventory = {'pilot/'+k: v for k, v in helper.inventory(self.pilot).items()}
        inventory.update({'launch/'+p.name: c.sha(p) for p in self.launch.iterdir()
            if p.name not in ('evidence.tar.gz', 'SHA256SUMS.json', 'FINISHED.json')})
        dump(self.launch/'SHA256SUMS.json', inventory)
        dump(self.launch/'FINISHED.json', {'complete': True, 'continuation_gate_passed': True,
            'archive_sha256': c.sha(self.launch/'evidence.tar.gz'),
            'inventory_sha256': c.sha(self.launch/'SHA256SUMS.json')})

    def verify(self):
        return c.verify_gate(helper, self.pilot, self.launch, self.jobs, self.sources, self.plan)

    def test_accepts_exact_terminal_prefix(self):
        files, _ = self.verify()
        self.assertEqual(files, helper.inventory(self.pilot))

    def test_requires_actual_controller_and_worker_termination(self):
        with patch.object(c, 'require_terminal', side_effect=ValueError('still live')):
            with self.assertRaisesRegex(ValueError, 'still live'):
                self.verify()

    def test_threshold_is_recalculated_from_rows(self):
        for n in (0, 1):
            p = self.pilot/'plain/s0'/f'round-{n}/grade.json'
            value = c.read(p); value['workflow_complete'] = False; dump(p, value)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'threshold'):
            self.verify()

    def test_invalid_forecast_fails_even_with_full_workflow(self):
        p = self.pilot/'ledger/s0/round-0/grade.json'
        value = c.read(p); value['valid'] = False; dump(p, value)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'threshold'):
            self.verify()

    def test_failed_gate_not_overridden_by_scores_or_complete_archive(self):
        dump(self.pilot/'GATE.json', {'passed': False, 'accuracy_used_for_gate': False})
        self.seal()
        with self.assertRaisesRegex(ValueError, 'gate did not pass'):
            self.verify()

    def test_accuracy_based_gate_rejected(self):
        dump(self.pilot/'GATE.json', {'passed': True, 'accuracy_used_for_gate': True})
        self.seal()
        with self.assertRaisesRegex(ValueError, 'gate did not pass'):
            self.verify()

    def test_added_pilot_file_is_not_silently_inherited(self):
        (self.pilot/'unexpected').write_text('extra')
        with self.assertRaisesRegex(ValueError, 'terminal inventory'):
            self.verify()

    def test_changed_launch_receipt_rejected(self):
        dump(self.launch/'AUDITED.json', {'complete': True, 'changed': True})
        with self.assertRaisesRegex(ValueError, 'launch evidence changed'):
            self.verify()

    def test_wrong_plan_refused(self):
        dump(self.plan, {'different': True})
        with self.assertRaisesRegex(ValueError, 'frozen plan'):
            self.verify()

    def test_shutdown_gap_cannot_be_hidden_by_complete(self):
        p = self.pilot/'report.json'; value = c.read(p)
        value['shutdown_record_gaps'] = ['missing']; dump(p, value); self.seal()
        with self.assertRaisesRegex(ValueError, 'independent audit'):
            self.verify()

    def test_requires_exact_resume_proof(self):
        proof = self.root/'proof.json'
        value = {'passed': True, 'engy_calls': 0, 'continuation_sources': c.source_identity(),
            'frozen_sources': self.sources, 'tested_inventory': {'test': True},
            'resumed_arms': ['plain', 'gnomon', 'ledger'], 'numerical_attempts': 96,
            'independent_audit_checks': 10, 'checks': [{'passed': True}]}
        dump(proof, value)
        c.verify_preflight(proof, self.sources, {'test': True})
        for field, bad in (('engy_calls', 1), ('continuation_sources', {}), ('checks', []),
                           ('tested_inventory', {}), ('resumed_arms', ['ledger'])):
            with self.subTest(field=field):
                dump(proof, {**value, field: bad})
                with self.assertRaisesRegex(ValueError, 'preflight'):
                    c.verify_preflight(proof, self.sources, {'test': True})


if __name__ == '__main__':
    unittest.main()
