"""Check M5 development stage coverage against authenticated host-only jobs.

These checks do not replace terminal-process, source/runtime, archive, copied
state or one-shot dispatch checks. They grant no execution or final-data access.
They never drop failed sessions from a complete development comparison.
"""
import json
import math

from .m5_ml_development_contract import authenticated_contract, DEVELOPMENT_JOBS_SHA
from .m5_ml_prefix_identity import check_prefix_identity


def _check_stage(cohort, jobs, grades, report, stage):
    """Pure structural/numerical check; public entry authenticates its inputs."""
    if stage not in ('pilot', 'complete'):
        raise ValueError('Development stage must be pilot or complete')
    selected = [c for c in cohort['cases'] if stage == 'complete' or c['stage'] == 'pilot']
    expected = {(a, c['series_id'], c['round']): c for a in cohort['arms'] for c in selected}
    counts = cohort['decisions_per_seed']
    required = counts['pilot'] if stage == 'pilot' else counts['total']
    if len(expected) != required or required != (72 if stage == 'pilot' else 624):
        raise ValueError('Require the exact fixed eight-series development stage')
    if (report.get('complete') is not True or report.get('audit_failures') != []
            or report.get('shutdown_record_gaps') != []):
        raise ValueError('Complete independent stage audit required')

    def indexed(rows):
        if type(rows) is not list:
            raise ValueError('Every stage row must be retained in a list')
        result = {}
        for row in rows:
            if type(row) is not dict or type(row.get('round')) is not int:
                raise ValueError('Invalid stage row identity')
            key = row.get('arm'), row.get('series_id'), row['round']
            if key not in expected or key in result:
                raise ValueError('Wrong, duplicate or additional stage task')
            result[key] = row
        if set(result) != set(expected):
            raise ValueError('Missing stage tasks; no success or accuracy filtering')
        return result

    raw, audited = indexed(grades), indexed(report.get('rows'))
    for key, row in raw.items():
        case = expected[key]
        if row.get('origin') != case['origin']:
            raise ValueError('Grade origin differs from fixed task')
        for field in ('valid', 'workflow_complete', 'fallback_used'):
            if type(row.get(field)) is not bool:
                raise ValueError('Explicit boolean completion/fallback fields required')
        if row['workflow_complete'] and not row['valid']:
            raise ValueError('Full workflow cannot have an invalid forecast')
        point = row.get('point')
        if (type(point) is not list or len(point) != case['horizon']
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in point)):
            raise ValueError('Every case, including fallback, requires a complete valid point forecast')
        actual = jobs[case['series_id']][case['round']]['actual']
        score = math.sqrt(sum((math.log1p(p)-math.log1p(a))**2
                              for p, a in zip(point, actual, strict=True))/len(actual))
        if (type(row.get('rmsle')) not in (int, float) or not math.isfinite(row['rmsle'])
                or abs(score-row['rmsle']) > 1e-12):
            raise ValueError('Case score does not match the fixed actuals and retained prediction')
        if any(audited[key].get(field) != value for field, value in row.items()):
            raise ValueError('Retained grade and independently audited row disagree')

    arms = {}
    for arm in cohort['arms']:
        rows = [r for key, r in raw.items() if key[0] == arm]
        summary = {'tasks': len(rows), 'valid': sum(r['valid'] for r in rows),
                   'workflow_complete': sum(r['workflow_complete'] for r in rows),
                   'fallbacks': sum(r['fallback_used'] for r in rows)}
        if any(report.get('arms', {}).get(arm, {}).get(k) != summary[k]
               for k in ('tasks', 'valid', 'workflow_complete')):
            raise ValueError('Per-arm report counts disagree with retained rows')
        # Same 11/12 full-workflow proportion as the existing four-series pilot;
        # twice as many series means 22/24, with every forecast valid.
        if stage == 'pilot' and (summary['valid'] != 24 or summary['workflow_complete'] < 22):
            raise ValueError('Pilot needs 24 valid forecasts and at least 22 full workflows per arm')
        arms[arm] = summary
    return {'stage': stage, 'evidence_checks_passed': True,
            'sessions': len(raw), 'arms': arms,
            'retained_pilot_sessions': counts['pilot'],
            'continuation_sessions': counts['continuation'],
            'planned_total_sessions': counts['total'],
            'scores_recomputed': len(raw), 'accuracy_used_for_completion': False,
            'operational_gate_passed': False, 'execution_authorized': False,
            'final_gate_opened': False,
            'scope': 'Development stage identity, coverage, score and completion checks only. '
                     'Terminal processes, immutable archives, copied state, source/runtime '
                     'identity and prospective dispatch admission remain separate.'}


def check_development_stage(manifest_bytes, jobs_bytes, grades, report, *, stage):
    cohort = authenticated_contract(manifest_bytes, jobs_bytes)
    return _check_stage(cohort, json.loads(jobs_bytes), grades, report, stage)


def check_continuation_prefix(manifest_bytes, jobs_bytes, grades, report, *,
                              pilot_manifest, capsule, runtime_inventory):
    """Require seed/source/runtime binding and every fixed pilot task together.

    Pure checks only. The caller must authenticate capsule/plan identity and
    validate terminal processes, archives and independent audit provenance
    before copying or dispatching any continuation.
    """
    binding = check_prefix_identity(pilot_manifest, capsule, runtime_inventory)
    if type(pilot_manifest.get('planned')) is not int or pilot_manifest['planned'] != 72:
        raise ValueError('Continuation requires the complete 72-session M5 pilot')
    if pilot_manifest.get('source_jobs_sha256') != DEVELOPMENT_JOBS_SHA:
        raise ValueError('Pilot was not bound to the fixed M5 development job source')
    stage = check_development_stage(manifest_bytes, jobs_bytes, grades, report, stage='pilot')
    return {**stage, 'prefix_identity': binding,
            'scope': 'Joint fixed-cohort score/completion and seed/source/runtime checks. '
                     'This does not replace terminal-process, source-plan authentication, '
                     'immutable archive, copied-state or one-shot dispatch admission.'}
