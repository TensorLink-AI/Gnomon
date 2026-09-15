"""Authenticate already-loaded M5 development jobs and describe their cohort.

No file/network access, credentials, model calls or dispatch. This is an input
contract, not an execution or final-data access gate. Callers must separately
freeze the worker, seeds and budget and pass their operational admission checks.
"""
from datetime import datetime
import hashlib
import json
import re

from . import m5_ml_adapter as adapter
from .m5_ml_panel import validate_panel

DEVELOPMENT_JOBS_SHA = 'dd608a2a0188cbe7e0e684127c4e5abbc77019982fdd14aabbb6e635064c164b'
ARMS = ('plain', 'gnomon', 'ledger')
PILOT_ORIGINS = 3
JOB_FIELDS = {'series_id', 'round', 'origin', 'future_timestamps',
              'outcome_recorded_at', 'request', 'actual'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def describe_cohort(manifest, jobs):
    """Validate structure/temporal consistency; does not authenticate provenance.

    Used with synthetic fixtures as well as by the authenticated bytes entry
    point. Exact agreement with the shared constructor is required, including
    every overlapping history value, covariate, timestamp and visibility cutoff.
    No numerical cells from the reserved panel are accepted or requested.
    """
    validate_panel(manifest)  # Identity checks precede any numerical consumption.
    selected = {row['series_id']: row for row in manifest['splits']['development']}
    if type(jobs) is not dict or set(jobs) != set(selected):
        raise ValueError('Require exactly the eight selected development series; no replacements')
    cases = []
    common_dates = None
    for series, metadata in sorted(selected.items()):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', series):
            raise ValueError('Series identity is not a safe task-directory component')
        records = jobs[series]
        if type(records) is not list or len(records) != adapter.ROUNDS:
            raise ValueError('Require all 26 ordered origins: '+series)
        for number, job in enumerate(records):
            if (type(job) is not dict or set(job) != JOB_FIELDS
                    or type(job['round']) is not int or job['round'] != number
                    or job['series_id'] != series):
                raise ValueError('Wrong, duplicate or reordered task identity: '+series)
            if (type(job['request']) is not dict or type(job['actual']) is not list
                    or len(job['actual']) != adapter.HORIZON
                    or type(job['future_timestamps']) is not list
                    or len(job['future_timestamps']) != adapter.HORIZON):
                raise ValueError('Incomplete task request or target horizon: '+series)
        try:
            first = records[0]['request']
            if type(first['history']) is not list or type(first['timestamps']) is not list:
                raise ValueError('History values and timestamps must be arrays')
            values = first['history'] + [v for job in records for v in job['actual']]
            times = first['timestamps'] + [t for job in records for t in job['future_timestamps']]
            stamps = [datetime.fromisoformat(t) for t in times]
            rebuilt = adapter.build_series_jobs(metadata, values, stamps)
            if _canonical(records) != _canonical(rebuilt):
                raise ValueError('Jobs differ from the shared history/visibility constructor: '+series)
        except (KeyError, TypeError) as exc:
            raise ValueError('Malformed development job fields: '+series) from exc
        dates = [(job['origin'], job['outcome_recorded_at']) for job in rebuilt]
        if common_dates is not None and dates != common_dates:
            raise ValueError('All development series must share the same origin/target grid')
        common_dates = dates
        cases.extend({'series_id': series, 'store_id': metadata['store_id'],
                      'item_id': metadata['item_id'], 'round': job['round'],
                      'origin': job['origin'], 'outcome_recorded_at': job['outcome_recorded_at'],
                      'horizon': adapter.HORIZON,
                      'stage': 'pilot' if job['round'] < PILOT_ORIGINS else 'continuation'}
                     for job in rebuilt)
    pilot = sum(case['stage'] == 'pilot' for case in cases)
    return {'schema_version': 'm5-ml-development-cohort-v1',
            'provenance_authenticated': False, 'arms': list(ARMS), 'series': len(selected),
            'stores': len({row['store_id'] for row in selected.values()}),
            'origins_per_series': adapter.ROUNDS, 'history': adapter.HISTORY,
            'horizon': adapter.HORIZON, 'host_tasks': len(cases),
            'decisions_per_seed': {'pilot': pilot*len(ARMS),
                                   'continuation': (len(cases)-pilot)*len(ARMS),
                                   'total': len(cases)*len(ARMS)},
            'cases': cases, 'execution_authorized': False, 'final_gate_opened': False,
            'scope': 'Development input contract only. Worker, seed, budget, runtime, '
                     'completion and operational gates remain separate. No scores or forecasts.'}


def authenticated_contract(manifest_bytes, jobs_bytes):
    """Authenticate the original manifest and prepared development bytes first.

    Inputs must already be loaded from authorized development/metadata artifacts.
    This helper deliberately has no file reader or reserved-data export option.
    Matching bytes establish input identity, not authority to execute a trial.
    """
    if type(manifest_bytes) is not bytes or type(jobs_bytes) is not bytes:
        raise ValueError('Already-loaded bytes required; no paths or lazy readers')
    if hashlib.sha256(manifest_bytes).hexdigest() != adapter.MANIFEST_SHA:
        raise ValueError('Original frozen manifest bytes required')
    if hashlib.sha256(jobs_bytes).hexdigest() != DEVELOPMENT_JOBS_SHA:
        raise ValueError('Exact prepared development-job bytes required')
    result = describe_cohort(json.loads(manifest_bytes), json.loads(jobs_bytes))
    return {**result, 'provenance_authenticated': True,
            'input_identity': {'manifest_sha256': adapter.MANIFEST_SHA,
                               'development_jobs_sha256': DEVELOPMENT_JOBS_SHA}}
