"""Validated client submissions; claimed computation is never server attestation."""
from dataclasses import asdict
import hashlib
import math
from uuid import uuid4

from gnomon.forecast_adapter import ForecastRequest, ForecastResult
from gnomon.inference import ForecastExecution
from .storage import ServiceError, encode


def text(value, name, maximum=512, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ServiceError('INVALID_ARGUMENTS', f'{name} must be a nonempty string up to {maximum} characters.')


def execution(args, principal):
    request = ForecastRequest.from_dict(args['request'])
    text(args['provider'], 'provider')
    text(args.get('revision'), 'revision', optional=True)
    raw = args['result']
    allowed = {'point', 'quantiles', 'timestamps', 'series_id', 'unit', 'sample_paths', 'metadata'}
    if not isinstance(raw, dict) or set(raw) - allowed or 'point' not in raw:
        raise ServiceError('INVALID_ARGUMENTS', 'Supply a ForecastResult object.')
    def numbers(values):
        if not isinstance(values, (list, tuple)) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ServiceError('INVALID_ARGUMENTS', 'Forecast arrays require finite numbers.')
        return tuple(values)
    quantiles = raw.get('quantiles')
    if quantiles is not None:
        rows = []
        if not isinstance(quantiles, list):
            raise ServiceError('INVALID_ARGUMENTS', 'Quantiles require an array of probability/value objects.')
        for row in quantiles:
            if not isinstance(row, dict):
                raise ServiceError('INVALID_ARGUMENTS', 'Quantile rows require objects.')
            numbers(list(row.values()))
            converted = {float(k): v for k, v in row.items()}
            if len(converted) != len(row):
                raise ServiceError('INVALID_ARGUMENTS', 'Duplicate quantile probabilities.')
            rows.append(converted)
        quantiles = tuple(rows)
    metadata = raw.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ServiceError('INVALID_ARGUMENTS', 'Metadata must be an object.')
    result = ForecastResult(point=numbers(raw['point']), quantiles=quantiles,
        timestamps=tuple(raw.get('timestamps', request.future_timestamps)),
        series_id=raw.get('series_id', request.series_id), unit=raw.get('unit', request.unit),
        sample_paths=None if raw.get('sample_paths') is None else tuple(numbers(p) for p in raw['sample_paths']),
        metadata={'client_metadata': metadata, 'submission': {
            'origin': 'client_submitted', 'principal': principal, 'provider_execution_verified': False,
            'claimed_computed_at': args.get('computed_at')}}).validate(request)
    fingerprint = hashlib.sha256(encode({'request': asdict(request), 'result': asdict(result),
        'provider': args['provider'], 'revision': args.get('revision')}).encode()).hexdigest()
    return ForecastExecution(str(uuid4()), fingerprint, args['provider'], args.get('revision'), request, result,
        evidence='client_submitted', provider_identity={'origin': 'client_submitted',
            'assertion_basis': 'caller_declared', 'submitted_by': principal, 'execution_verified': False})
