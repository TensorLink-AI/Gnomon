from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from benchmarks.ledger_optimization import reconcile_contrast_100 as r


def fixture():
    plan = {'arms': ['plain', 'gnomon', 'ledger'],
            'cases': [{'series_id': s, 'round': n, 'origin': f'origin-{n}'}
                      for s in ('a', 'b', 'c', 'd') for n in range(26)],
            'capsule': {'sources': {'worker.py': 'fixed'}}, 'task_source_sha256': 'jobs',
            'agent': {'model': 'deepseek-v4.1-flash'},
            'gnomon': {'version': '1.2.0', 'build_source_sha256': 'build'}}
    manifest = {'planned': 312, 'pilot_sessions_retained': 36, 'new_sessions_planned': 276,
                'accuracy_used_for_promotion': False, 'sources': plan['capsule']['sources'],
                'source_jobs_sha256': 'jobs', 'model': 'deepseek-v4.1-flash',
                'build': {'package_version': '1.2.0', 'source_sha256': 'build'},
                'inventory': {'plain': 'plain-inventory', 'gnomon': 'gnomon-inventory'}}
    done = {'completed': 312, 'planned': 312, 'new_sessions': 276,
            'retained_pilot_sessions': 36, 'source_unchanged': True}
    report = {'complete': True, 'audit_failures': [], 'shutdown_record_gaps': [],
              'audit_checks': 100, 'rows': [dict(c, arm=a, valid=False, rmsle=100.)
                                          for a in plan['arms'] for c in plan['cases']]}
    costs = {'deduplicated_total': {'sessions': 312},
             'stages': {'retained_pilot': {'sessions': 36}, 'continuation': {'sessions': 276}}}
    return [plan, manifest, done, {'exit_status': 1, 'final_gate_opened': False},
            r.KNOWN_CAUSE, report, costs, manifest['inventory']]


class ReconciliationTests(unittest.TestCase):
    def test_full_failed_agent_cohort_is_not_accuracy_filtered(self):
        result = r.validate_reports(*fixture())
        self.assertEqual(result['sessions'], 312)
        self.assertFalse(result['accuracy_used_for_admission'])

    def test_missing_or_duplicate_cases_rejected(self):
        for duplicate in (False, True):
            args = fixture()
            args[5]['rows'].pop()
            if duplicate:
                args[5]['rows'].append(deepcopy(args[5]['rows'][0]))
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                r.validate_reports(*args)

    def test_unrelated_failure_or_unexecuted_original_rejected(self):
        for index, value in [(4, 'Request identity mismatch'),
                             (3, {'exit_status': 0, 'final_gate_opened': False})]:
            args = fixture(); args[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                r.validate_reports(*args)

    def test_remaining_audit_or_shutdown_failure_rejected(self):
        for key in ('audit_failures', 'shutdown_record_gaps'):
            args = fixture(); args[5][key] = ['remaining failure']
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_reports(*args)

    def test_source_runtime_version_or_cutoff_change_rejected(self):
        mutations = [lambda a: a[1].update(source_jobs_sha256='different'),
                     lambda a: a[1]['build'].update(package_version='1.1.9'),
                     lambda a: a.__setitem__(7, {'plain': 'changed'}),
                     lambda a: a[5]['rows'][0].update(origin='different'),
                     lambda a: a[2].update(source_unchanged=False)]
        for mutate in mutations:
            args = fixture(); mutate(args)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                r.validate_reports(*args)

    def test_partial_costs_rejected(self):
        args = fixture(); args[6]['stages']['continuation']['sessions'] = 275
        with self.assertRaises(ValueError):
            r.validate_reports(*args)

    def test_cost_replay_only_ignores_clock_and_local_root(self):
        a = {'at': 'earlier', 'root': '/old', 'tokens': 100, 'unknown_usage': 1}
        b = dict(a, at='later', root='/new')
        self.assertEqual(r.cost_facts(a), r.cost_facts(b))
        for key in ('tokens', 'unknown_usage'):
            c = dict(b); c[key] += 1
            self.assertNotEqual(r.cost_facts(a), r.cost_facts(c))

    def test_live_predecessor_blocks_before_output_or_audit(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); launch = root/'launch'; launch.mkdir()
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            (launch/'launch.json').write_text(json.dumps({'pid': 123, 'boot_id': boot}))
            with patch.object(r.terminal, 'digest', return_value=r.PLAN_SHA), \
                    patch.object(r.terminal, 'terminal', side_effect=ValueError('live')), \
                    patch.object(r.terminal, 'archived_prefix') as archive, \
                    patch.object(r.continuation, 'configure') as configure:
                with self.assertRaisesRegex(ValueError, 'live'):
                    r.reconcile(root/'output', launch, root/'capsule', root/'runtime',
                                root/'plan.json', root/'reconciliation')
                archive.assert_not_called(); configure.assert_not_called()
                self.assertFalse((root/'reconciliation').exists())

    def test_foreign_host_process_receipt_cannot_prove_terminal_state(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); launch = root/'launch'; launch.mkdir()
            (launch/'launch.json').write_text(json.dumps({'pid': 123, 'boot_id': 'another-host'}))
            with patch.object(r.terminal, 'digest', return_value=r.PLAN_SHA), \
                    patch.object(r.terminal, 'terminal') as terminal:
                with self.assertRaisesRegex(ValueError, 'original host'):
                    r.reconcile(root/'output', launch, root/'capsule', root/'runtime',
                                root/'plan.json', root/'reconciliation')
                terminal.assert_not_called()
                self.assertFalse((root/'reconciliation').exists())

    def test_existing_destination_blocks_before_source_reads(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(r.terminal, 'digest') as digest:
                with self.assertRaisesRegex(ValueError, 'Fresh'):
                    r.reconcile('/o', '/l', '/c', '/r', '/p', root)
                digest.assert_not_called()

    def test_corrected_audit_failure_restores_original_callable(self):
        sentinel = object()
        namespace = {'audit_annotations': sentinel, 'frozen': sentinel, 'cause': r.KNOWN_CAUSE}
        exec('def analyze(source, destination):\n'
             '    if audit_annotations is frozen: raise ValueError(cause)\n'
             '    raise ValueError("different evidence failure")\n', namespace)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'different evidence failure'):
                r.replay_audits(SimpleNamespace(analyze=namespace['analyze']), Path('/unused'), Path(root))
            self.assertIs(namespace['audit_annotations'], sentinel)
            failure = json.loads((Path(root)/'original-audit-failure.json').read_text())
            self.assertEqual(failure['cause'], r.KNOWN_CAUSE)
            self.assertFalse((Path(root)/'receipt.json').exists())

    def test_passing_original_is_not_relabelled_known_failure(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'did not reproduce'):
                r.replay_audits(SimpleNamespace(analyze=lambda *args: {}), Path('/unused'), Path(root))


if __name__ == '__main__':
    unittest.main()
