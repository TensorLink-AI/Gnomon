"""Local analysis of immutable evidence snapshots supplied by a trusted ledger.

Transport authentication and snapshot reference resolution belong to the caller.
These computations verify arithmetic, not the truth of observations or provenance.
"""
from datetime import datetime
from .forecast_adapter import ForecastAdapterError, ForecastRequest, point_error_metrics


def score_snapshot(snapshot):
    """Compute point metrics locally without providers, writes or network access."""
    if snapshot.get('kind') != 'evidence_snapshot/1':
        raise ForecastAdapterError('Unsupported evidence snapshot')
    execution = snapshot['execution']
    request = ForecastRequest.from_dict(execution['request'])
    targets = {datetime.fromisoformat(t): i for i, t in enumerate(request.future_timestamps)}
    points = execution['result']['point']
    if len(points) != request.horizon or len(targets) != request.horizon:
        raise ForecastAdapterError('Snapshot requires aligned targets and forecast points')
    pairs, seen = [], set()
    for actual in snapshot['actuals']:
        time = datetime.fromisoformat(actual['valid_time'])
        if time not in targets or time in seen or actual['series_id'] != request.series_id or actual['unit'] != request.unit:
            raise ForecastAdapterError('Snapshot actuals must be unique and match the forecast series, unit and targets')
        if (datetime.fromisoformat(actual['source_available_at']) > datetime.fromisoformat(snapshot['source_as_of'])
                or datetime.fromisoformat(actual['recorded_at']) > datetime.fromisoformat(snapshot['recorded_as_of'])):
            raise ForecastAdapterError('Snapshot actual exceeds evidence cutoff')
        seen.add(time)
        pairs.append((points[targets[time]], actual['value']))
    return {'method': 'gnomon.point_error_metrics', 'method_version': '1',
            'metrics': point_error_metrics(pairs),
            'scoring_status': 'complete' if len(pairs) == request.horizon else 'partial' if pairs else 'pending',
            'actual_ids': [a['actual_id'] for a in snapshot['actuals']],
            'numerically_verified_by': 'local_computation', 'source_truth_verified': False}
