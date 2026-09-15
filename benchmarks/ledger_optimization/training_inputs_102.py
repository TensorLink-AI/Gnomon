"""Undeployed effective-input descriptions for the frozen ML lab predictor.

This is deliberately adapter-specific. Callers authenticate execution payloads
and supply the SHA-256 of the actual numerical.py. Equal effective-input hashes
are not equal Gnomon requests, recording evidence, or independent observations.
No forecasting, ledger query, or selection is performed here.
"""
from datetime import datetime, timezone
import hashlib
import json
import math

ALGORITHM_SHA256 = 'a71b75f06ad69fcca06275046c5e74565b15d7d6dd2538df0fe827c448fc0413'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _time(value):
    if not isinstance(value, str):
        raise ValueError('Explicit timestamp strings required')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.utcoffset() is None:
        raise ValueError('Timezone-aware timestamps required')
    return result.astimezone(timezone.utc).isoformat()


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite numerical inputs required')
    # Canonicalize integral floats, integers and signed zero alike.
    value = float(value)
    return 0. if value == 0 else value


def describe_inputs(request, config, *, algorithm_sha256):
    """Describe the input subset consumed by the pinned numerical predictor.

    Input requests/configurations must already belong to successful audited
    executions. This checks the dimensions needed by the description; it does
    not replace Gnomon's request contract or the execution/visibility auditor.
    """
    if algorithm_sha256 != ALGORITHM_SHA256:
        raise ValueError('Unknown numerical implementation; effective-input semantics unavailable')
    model = config.get('model')
    required = {'seasonal': {'model', 'season'},
                'ridge': {'model', 'window', 'lags', 'alpha'},
                'random_forest': {'model', 'window', 'lags', 'depth'}}
    if model not in required or set(config) != required[model]:
        raise ValueError('Complete canonical ML-lab configuration required')
    canonical = dict(config)
    for field in set(config)-{'model'}:
        value = _number(config[field])
        if value <= 0 or (field != 'alpha' and not value.is_integer()):
            raise ValueError('Positive integer sizes and positive alpha required')
        canonical[field] = value if field == 'alpha' else int(value)
    history = [_number(x) for x in request['history']]
    if not history or any(x < 0 for x in history):
        raise ValueError('Nonnegative observed sales history required')
    horizon = request['horizon']
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Positive integer horizon required')
    times = [_time(x) for x in request['timestamps']]
    future_times = [_time(x) for x in request['future_timestamps']]
    origin = _time(request['cutoff'])
    if (len(times) != len(history) or len(future_times) != horizon
            or times != sorted(set(times)) or future_times != sorted(set(future_times))
            or times[-1] != origin or future_times[0] <= origin):
        raise ValueError('Aligned increasing timestamps and exact history endpoint required')
    series, unit = request['series_id'], request['unit']
    if not isinstance(series, str) or not series or (unit is not None and
            (not isinstance(unit, str) or not unit)):
        raise ValueError('Explicit series and unit identity required')
    used = canonical['season'] if model == 'seasonal' else min(canonical['window'], len(history))
    if used > len(history) or (model != 'seasonal' and used < canonical['lags']+20):
        raise ValueError('Insufficient effective history for the declared predictor')
    past, future = [], []
    if model != 'seasonal':
        names = ['onpromotion', 'dow_sin', 'dow_cos']
        if request['past_covariate_names'] != names or request['future_covariate_names'] != names:
            raise ValueError('Pinned ML-lab covariate order required')
        past = [[_number(x) for x in row] for row in request['past_covariates']]
        future = [[_number(x) for x in row] for row in request['future_covariates']]
        if len(past) != len(history) or len(future) != horizon or any(len(r) != 3 for r in past+future):
            raise ValueError('Aligned three-column covariates required')
        past = past[-used:]
    training = {'algorithm_sha256': algorithm_sha256, 'config': canonical,
                'history': history[-used:], 'timestamps': times[-used:],
                'past_covariates': past}
    task = {'series_id': series, 'unit': unit, 'origin': origin,
            'horizon': horizon, 'future_timestamps': future_times}
    effective = {'training': training, 'task': task, 'future_covariates': future}
    return {
        'schema_version': 'effective-ml-inputs-102', 'algorithm_sha256': algorithm_sha256,
        'config': canonical, 'task': task, 'task_sha256': _digest(task),
        'available_history_rows': len(history), 'effective_history_rows': used,
        'configured_max_history_rows': None if model == 'seasonal' else canonical['window'],
        'window_truncated_by_available_history': model != 'seasonal' and used < canonical['window'],
        'history_start': times[0], 'effective_history_start': times[-used], 'history_end': times[-1],
        'supervised_training_rows': 0 if model == 'seasonal' else used-canonical['lags'],
        'effective_training_sha256': _digest(training), 'effective_model_inputs_sha256': _digest(effective),
        'temporal_provenance': {k: request.get(k) for k in
                                ('known_time_cutoff', 'recorded_time_cutoff', 'snapshot_id')},
        'scope': 'Pinned ML-lab numerical inputs only. Does not assert equal recording '
                 'provenance, Gnomon request fingerprints, independent evidence or forecast skill.',
    }


def compare_inputs(left_request, right_request, config, *, algorithm_sha256):
    """Compare two authenticated executions of the same versioned configuration."""
    left = describe_inputs(left_request, config, algorithm_sha256=algorithm_sha256)
    right = describe_inputs(right_request, config, algorithm_sha256=algorithm_sha256)
    same_task = left['task_sha256'] == right['task_sha256']
    same_training = left['effective_training_sha256'] == right['effective_training_sha256']
    same_inputs = same_task and left['effective_model_inputs_sha256'] == right['effective_model_inputs_sha256']
    return {'left': left, 'right': right, 'same_task': same_task,
            'same_effective_training_inputs': same_training, 'same_effective_model_inputs': same_inputs,
            'status': 'different_task' if not same_task else
                      ('matching_effective_model_inputs' if same_inputs else 'different_effective_model_inputs'),
            'provider_calls': 0, 'ledger_queries': 0, 'forecast_selection_made': False,
            'recording_evidence_equivalence_asserted': False}
