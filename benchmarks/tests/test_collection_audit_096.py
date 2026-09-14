from copy import deepcopy
import hashlib
import json
import unittest

from benchmarks.ledger_optimization.collection_audit_096 import audit_collection_events


JOB = {'origin': 't729', 'series_id': 's', 'request': {'unit': 'widgets', 'timestamps': [f't{i}' for i in range(730)]}}
ENDS = [688, 702, 716, 730]


def batch(config, before=60, missing=ENDS, baseline=False, tag='a'):
    cid = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    events = [{'event': 'collection_batch_admitted', 'task_origin': 't729', 'config': config,
               'config_id': cid, 'fits': len(missing), 'missing_ends': list(missing),
               'initial_baseline': baseline, 'reserve': 0 if baseline else 1,
               'remaining_before': before, 'phase': 'exploration', 'selection_changed': False}]
    for end in missing:
        a = {'event': 'attempt', 'task_origin': 't729', 'attempt_id': tag+str(end), 'config': config,
             'config_id': cid, 'kind': 'forecast' if end == 730 else 'backtest',
             'request': {'cutoff': f't{end-1}', 'series_id': 's', 'unit': 'widgets'}}
        events.extend([a, {**deepcopy(a), 'event': 'result'}])
    return events


class CollectionAuditTests(unittest.TestCase):
    def test_complete_batches_and_no_free_repeated_fits(self):
        events = batch({'model': 'seasonal', 'season': 7}, baseline=True)
        events += batch({'model': 'ridge', 'alpha': 10.}, before=56, tag='b')
        report = audit_collection_events(events, JOB)
        self.assertEqual(report, {'admitted_batches': 2, 'numerical_attempts': 8,
                                  'successful_results': 8, 'failed_or_unfinished_attempts': 0,
                                  'production_results': 2})
        with self.assertRaisesRegex(ValueError, 'refitted'):
            audit_collection_events(events + batch({'model': 'ridge', 'alpha': 10.}, tag='c')[7:], JOB)

    def test_failed_fit_counted_and_completed_folds_reused(self):
        config = {'model': 'ridge', 'alpha': 10.}
        events = batch(config)[:-1]  # Production attempt failed, no result.
        events += batch(config, before=56, missing=[730], tag='retry')
        report = audit_collection_events(events, JOB)
        self.assertEqual(report['numerical_attempts'], 5)
        self.assertEqual(report['failed_or_unfinished_attempts'], 1)
        self.assertEqual(report['production_results'], 1)

    def test_explicit_commit_recovery_can_produce_missing_forecast(self):
        config = {'model': 'ridge', 'alpha': 10.}
        events = batch(config)[:-1]
        events += batch(config, missing=[730], tag='commit')[1:]
        self.assertEqual(audit_collection_events(events, JOB)['production_results'], 1)

    def test_budget_phase_and_admission_tampering_rejected(self):
        edits = [('fits', 3), ('remaining_before', 61), ('reserve', 0),
                 ('phase', 'selection'), ('selection_changed', True), ('missing_ends', [730])]
        for field, bad in edits:
            with self.subTest(field=field):
                events = batch({'model': 'ridge', 'alpha': 10.})
                events[0][field] = bad
                with self.assertRaises(ValueError):
                    audit_collection_events(events, JOB)

    def test_unadmitted_backtest_or_orphan_result_rejected(self):
        events = batch({'model': 'ridge', 'alpha': 10.})
        for invalid in (events[1:], [events[2]], events+[deepcopy(events[2])]):
            with self.assertRaises(ValueError):
                audit_collection_events(invalid, JOB)

    def test_partial_admission_does_not_invent_success(self):
        events = batch({'model': 'ridge', 'alpha': 10.})[:1]
        report = audit_collection_events(events, JOB)
        self.assertEqual(report['numerical_attempts'], 0)
        self.assertEqual(report['production_results'], 0)


if __name__ == '__main__':
    unittest.main()
