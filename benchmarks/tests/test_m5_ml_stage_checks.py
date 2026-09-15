from copy import deepcopy
import json
import math
import unittest
from unittest.mock import patch

from benchmarks.ledger_optimization import m5_ml_stage_checks as s
from benchmarks.ledger_optimization.m5_ml_development_contract import describe_cohort
from benchmarks.tests.test_m5_ml_development_contract import fixture


class StageCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.jobs = fixture()
        cls.cohort = describe_cohort(cls.manifest, cls.jobs)

    def evidence(self, stage):
        rows = []
        for case in self.cohort['cases']:
            if stage == 'pilot' and case['stage'] != 'pilot':
                continue
            for arm in self.cohort['arms']:
                rows.append({'arm': arm, 'series_id': case['series_id'], 'round': case['round'],
                             'origin': case['origin'], 'valid': True, 'workflow_complete': True,
                             'fallback_used': False, 'rmsle': 0.,
                             'point': list(self.jobs[case['series_id']][case['round']]['actual'])})
        return rows, self.report(rows)

    def report(self, rows):
        return {'complete': True, 'audit_failures': [], 'shutdown_record_gaps': [],
                'rows': deepcopy(rows), 'arms': {arm: {
                    'tasks': sum(r['arm'] == arm for r in rows),
                    'valid': sum(r['arm'] == arm and r['valid'] for r in rows),
                    'workflow_complete': sum(r['arm'] == arm and r['workflow_complete'] for r in rows)}
                    for arm in self.cohort['arms']}}

    def check(self, rows, report, stage='pilot'):
        return s._check_stage(self.cohort, self.jobs, rows, report, stage)

    def test_full_pilot_and_continuation_counts_without_dispatch_permission(self):
        for stage, count in (('pilot', 72), ('complete', 624)):
            rows, report = self.evidence(stage)
            result = self.check(rows, report, stage)
            self.assertEqual(result['sessions'], count)
            self.assertEqual(result['scores_recomputed'], count)
            self.assertEqual(result['retained_pilot_sessions'], 72)
            self.assertEqual(result['continuation_sessions'], 552)
            self.assertFalse(result['accuracy_used_for_completion'])
            self.assertFalse(result['operational_gate_passed'])
            self.assertFalse(result['execution_authorized'])
            self.assertFalse(result['final_gate_opened'])

    def test_half_pilot_or_missing_losing_rows_cannot_pass_even_if_report_agrees(self):
        rows, _ = self.evidence('pilot')
        for subset in (rows[:36], rows[:-1], rows+rows[:1]):
            with self.assertRaises(ValueError): self.check(subset, self.report(subset))
        with self.assertRaises(ValueError): self.check(rows, self.report(rows), 'complete')

    def test_bad_identity_or_score_cannot_be_hidden_in_agreeing_report(self):
        for field, value in (('round', True), ('origin', 'wrong'), ('series_id', 'reserved-0'),
                             ('rmsle', .5), ('rmsle', float('nan')), ('valid', 1),
                             ('point', [1.]), ('point', [True]*14)):
            rows, _ = self.evidence('pilot'); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(rows, self.report(rows))

    def test_pilot_threshold_keeps_same_proportion_and_rejects_invalid_forecasts(self):
        rows, _ = self.evidence('pilot')
        selected = [r for r in rows if r['arm'] == 'ledger']
        for row in selected[:2]: row['workflow_complete'] = False
        self.assertTrue(self.check(rows, self.report(rows))['evidence_checks_passed'])
        selected[2]['workflow_complete'] = False
        with self.assertRaisesRegex(ValueError, 'at least 22'): self.check(rows, self.report(rows))
        selected[2]['workflow_complete'] = True
        selected[0].update(valid=False, fallback_used=True)
        with self.assertRaisesRegex(ValueError, '24 valid'): self.check(rows, self.report(rows))

    def test_complete_development_keeps_failed_fallbacks_in_denominator(self):
        rows, _ = self.evidence('complete')
        rows[0].update(valid=False, workflow_complete=False, fallback_used=True)
        result = self.check(rows, self.report(rows), 'complete')
        self.assertEqual(result['sessions'], 624)
        self.assertEqual(result['arms']['plain']['fallbacks'], 1)
        self.assertEqual(result['arms']['plain']['tasks'], 208)

    def test_poor_but_correctly_scored_forecasts_do_not_block_continuation(self):
        rows, _ = self.evidence('pilot')
        for row in rows:
            actual = self.jobs[row['series_id']][row['round']]['actual']
            row['point'] = [0.]*14
            row['rmsle'] = math.sqrt(sum(math.log1p(v)**2 for v in actual)/14)
        result = self.check(rows, self.report(rows))
        self.assertTrue(all(r['rmsle'] > 1 for r in rows))
        self.assertTrue(result['evidence_checks_passed'])
        self.assertFalse(result['accuracy_used_for_completion'])

    def test_independent_audit_failures_and_changed_report_are_not_overridden(self):
        for mutate in (lambda r: r.update(complete=False),
                       lambda r: r.update(audit_failures=['failed']),
                       lambda r: r.update(shutdown_record_gaps=['missing']),
                       lambda r: r['rows'][0].update(rmsle=1.),
                       lambda r: r['arms']['ledger'].update(tasks=1)):
            rows, report = self.evidence('pilot'); mutate(report)
            with self.assertRaises(ValueError): self.check(rows, report)

    def test_operational_entry_authenticates_before_inspecting_grades(self):
        with patch.object(s, '_check_stage', side_effect=AssertionError('Inspected untrusted rows')):
            with self.assertRaises(ValueError):
                s.check_development_stage(b'{}', b'{}', [], {}, stage='pilot')


if __name__ == '__main__':
    unittest.main()
