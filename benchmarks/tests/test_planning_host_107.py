import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.ledger_optimization import planning_host_107 as host
from benchmarks.ledger_optimization.probe_planning_host_107 import fixture


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.jobs = fixture()

    def evidence(self, stage='pilot'):
        jobs = {s: rows[:6] if stage == 'pilot' else rows for s, rows in self.jobs.items()}
        host.dump(self.root/'host-jobs.json', jobs)
        rows = []
        for arm in host.ARMS:
            for series, tasks in jobs.items():
                for task in tasks:
                    row = {'arm': arm, 'series_id': series, 'round': task['round'], 'origin': task['origin'],
                           'point': task['actual'], 'rmsle': 0., 'valid': True, 'workflow_complete': True, 'fallback_used': False}
                    path = self.root/arm/series/f'round-{task["round"]}'/'grade.json'; path.parent.mkdir(parents=True)
                    host.dump(path, row); rows.append(row)
        per_arm = len(rows)//3
        return {'complete': True, 'audit_failures': [], 'shutdown_record_gaps': [], 'rows': rows,
                'arms': {a: {'tasks': per_arm, 'valid': per_arm, 'workflow_complete': per_arm} for a in host.ARMS}}

    def test_pilot_has_six_origins_and_72_sessions(self):
        checked = host.stage_checks(self.root, self.jobs, 'pilot', self.evidence())
        self.assertEqual(checked['sessions'], 72); self.assertTrue(checked['pilot_quality_passed'])

    def test_complete_has_all_312_sessions(self):
        checked = host.stage_checks(self.root, self.jobs, 'complete', self.evidence('complete'))
        self.assertEqual(checked['sessions'], 312); self.assertIsNone(checked['pilot_quality_passed'])

    def test_missing_failed_session_cannot_be_dropped(self):
        report = self.evidence(); row = report['rows'].pop()
        (self.root/row['arm']/row['series_id']/f'round-{row["round"]}'/'grade.json').unlink()
        with self.assertRaisesRegex(ValueError, 'denominator'): host.stage_checks(self.root, self.jobs, 'pilot', report)

    def test_self_consistent_wrong_score_rejected(self):
        report = self.evidence(); row = report['rows'][0]; row['rmsle'] = 1.
        path = self.root/row['arm']/row['series_id']/f'round-{row["round"]}'/'grade.json'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'score disagrees'): host.stage_checks(self.root, self.jobs, 'pilot', report)

    def test_quality_failure_is_complete_evidence_not_missing_evidence(self):
        report = self.evidence()
        for row in report['rows'][:3]:
            row['workflow_complete'] = False
            (self.root/row['arm']/row['series_id']/f'round-{row["round"]}'/'grade.json').write_text(json.dumps(row))
        report['arms']['plain']['workflow_complete'] = 21
        checked = host.stage_checks(self.root, self.jobs, 'pilot', report)
        self.assertTrue(checked['passed']); self.assertFalse(checked['pilot_quality_passed'])

    def test_pilot_costs_include_round_five_only_once(self):
        for n in (0, 3, 5, 6): (self.root/'plain'/'example'/f'round-{n}').mkdir(parents=True)
        result = host.accounting(self.root)
        self.assertEqual(result['deduplicated_total']['sessions'], 4)
        self.assertEqual(result['stages']['retained_pilot']['sessions'], 3)
        self.assertEqual(result['stages']['continuation']['sessions'], 1)
        self.assertIsNone(result['billing_dollars'])

    def test_immature_prior_outcomes_rejected(self):
        series = next(iter(self.jobs)); self.jobs[series][0]['outcome_recorded_at'] = '2099-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError, 'mature'): host.validate_jobs(self.jobs)

    def test_foreign_boot_does_not_prove_terminal(self):
        proc = self.root/'proc'; (proc/'sys/kernel/random').mkdir(parents=True)
        (proc/'sys/kernel/random/boot_id').write_text('current')
        launch = self.root/'launch'; launch.mkdir()
        host.dump(launch/'launch.json', {'pid': 123, 'start_ticks': '10', 'boot_id': 'foreign'})
        with self.assertRaisesRegex(ValueError, 'original host boot'):
            host.verify_pilot(self.root/'pilot', launch, self.jobs, {}, '', proc=proc)


if __name__ == '__main__': unittest.main()
