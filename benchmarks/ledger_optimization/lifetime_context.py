"""Same contextual retrieval with all temporally visible history eligible."""
from datetime import datetime
import numpy as np


def retrieve(current, episodes):
    """No actuals or realized scores are needed to choose the neighborhood."""
    now = datetime.fromisoformat(current['origin'])
    if now.tzinfo is None:
        raise ValueError('Explicit current origin required')
    candidates = []; seen = set()
    for row in episodes:
        if row['domain'] != current['domain']:
            continue
        at, close, recorded = (datetime.fromisoformat(row[k]) for k in
                               ('origin', 'last_target', 'outcome_recorded_at'))
        if any(t.tzinfo is None for t in (at, close, recorded)):
            raise ValueError('Explicit evidence timestamps required')
        if at >= now or close > now or recorded > now:
            continue
        key = row['series_id'], at
        if key in seen or close <= at or recorded < close:
            raise ValueError('Duplicate or inconsistent visible evidence')
        seen.add(key); candidates.append(row)
    origins = sorted({datetime.fromisoformat(r['origin']) for r in candidates})
    candidates = sorted([r for r in candidates if datetime.fromisoformat(r['origin']) in origins],
                        key=lambda r: (r['origin'], r['series_id']))
    current_x = np.asarray(current['features'], dtype=float)
    if current_x.shape != (12,) or not np.isfinite(current_x).all():
        raise ValueError('Twelve finite current features required')
    diagnostic = {'candidates': [], 'selected': [], 'distinct_origins': len(origins),
                  'ready': len(candidates) >= 16 and len(origins) >= 3,
                  'location': None, 'scale': None}
    diagnostic.update(origin_window='all_visible', candidates_outside_recent_eight=0, selected_outside_recent_eight=0)
    if not candidates:
        return diagnostic
    x = np.asarray([r['features'] for r in candidates], dtype=float)
    if x.shape != (len(candidates), 12) or not np.isfinite(x).all():
        raise ValueError('Twelve finite historical features required')
    location = x.mean(axis=0); scale = np.maximum(x.std(axis=0), .1)
    distances = np.square((x-current_x)/scale).sum(axis=1)
    diagnostic.update(location=location.tolist(), scale=scale.tolist())
    diagnostic['candidates'] = [{'series_id': r['series_id'], 'origin': r['origin'],
                                 'distance': float(distance)}
                                for r, distance in zip(candidates, distances, strict=True)]
    if diagnostic['ready']:
        diagnostic['selected'] = sorted(diagnostic['candidates'],
            key=lambda r: (r['distance'], r['origin'], r['series_id']))[:16]
    recent = set(origins[-8:])
    diagnostic['candidates_outside_recent_eight'] = sum(datetime.fromisoformat(r['origin']) not in recent for r in diagnostic['candidates'])
    diagnostic['selected_outside_recent_eight'] = sum(datetime.fromisoformat(r['origin']) not in recent for r in diagnostic['selected'])
    return diagnostic

