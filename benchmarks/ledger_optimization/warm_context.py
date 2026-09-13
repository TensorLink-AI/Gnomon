"""Contextual CV residual estimates from temporally visible same-domain episodes."""
from datetime import datetime
import hashlib
import json
import numpy as np

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')
FEATURES = tuple('log1p_cv_'+m for m in MODELS)+('log_history_mean', 'log_history_std',
    'zero_fraction', 'recent_week_log_level_change', 'daily_log_difference', 'weekly_log_difference')


def features(cv, history):
    if len(history) != 730 or set(cv) != set(MODELS):
        raise ValueError('Full current history and CV cohort required')
    a = np.asarray(history, dtype=float)
    scores = np.asarray([cv[m] for m in MODELS])
    if not np.isfinite(a).all() or (a < 0).any() or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Nonnegative finite observations and errors required')
    z = np.log1p(a)
    return np.r_[np.log1p(scores), z.mean(), z.std(), (a == 0).mean(),
                 z[-168:].mean()-z[-336:-168].mean(), np.abs(z[24:]-z[:-24]).mean(),
                 np.abs(z[168:]-z[:-168]).mean()].tolist()


def select(cv, current_features, domain, origin, history):
    now = datetime.fromisoformat(origin)
    if now.tzinfo is None or set(cv) != set(MODELS):
        raise ValueError('Explicit origin and complete CV cohort required')
    control = min(MODELS, key=lambda m: (cv[m], MODELS.index(m)))
    available = []; seen = set()
    for row in history:
        if row['domain'] != domain:
            continue
        at, close, recorded = (datetime.fromisoformat(row[k]) for k in ('origin', 'last_target', 'outcome_recorded_at'))
        if any(t.tzinfo is None for t in (at, close, recorded)):
            raise ValueError('Explicit evidence timestamps required')
        if at >= now or close > now or recorded > now:
            continue
        key = row['series_id'], at
        if key in seen or close <= at or recorded < close:
            raise ValueError('Duplicate or inconsistent evidence')
        seen.add(key)
        if set(row['scores']) != set(MODELS) or set(row['cv']) != set(MODELS):
            raise ValueError('Incomplete matched cohort')
        available.append(row)
    origins = sorted({r['origin'] for r in available}, key=datetime.fromisoformat)[-8:]
    available = sorted([r for r in available if r['origin'] in origins], key=lambda r: (r['origin'], r['series_id']))
    estimates = dict(cv)
    fit = None
    if len(available) >= 16 and len(origins) >= 3:
        x = np.asarray([r['features'] for r in available], dtype=float)
        y = np.asarray([[r['scores'][m]-r['cv'][m] for m in MODELS] for r in available])
        current = np.asarray(current_features, dtype=float)
        if x.shape[1] != len(FEATURES) or current.shape != (len(FEATURES),) or not all(np.isfinite(v).all() for v in (x, y, current)):
            raise ValueError('Invalid contextual features or residuals')
        location = x.mean(axis=0); scale = np.maximum(x.std(axis=0), .1)
        centered_x = (x-location)/scale
        center_y = y.mean(axis=0)
        coefficients = np.linalg.solve(centered_x.T@centered_x+10*np.eye(len(FEATURES)),
                                        centered_x.T@(y-center_y))
        residual = center_y+((current-location)/scale)@coefficients
        estimates = {m: max(0., cv[m]+.5*float(residual[i])) for i, m in enumerate(MODELS)}
        fit = {'location': location.tolist(), 'scale': scale.tolist(), 'target_mean': center_y.tolist(),
               'coefficients': coefficients.tolist(), 'predicted_residual': residual.tolist(),
               'training_sha256': hashlib.sha256(json.dumps({'x': x.tolist(), 'y': y.tolist()}, separators=(',', ':')).encode()).hexdigest()}
    chosen = min(MODELS, key=lambda m: (estimates[m], cv[m], MODELS.index(m)))
    return {'provider': chosen, 'control_provider': control, 'estimated_rmsle': estimates,
        'records': len(available), 'distinct_origins': len(origins),
        'retrieved': [{'series_id': r['series_id'], 'origin': r['origin']} for r in available],
        'effective_source_as_of': origin, 'effective_recorded_as_of': origin,
        'feature_schema': FEATURES, 'current_features': current_features,
        'contextual_fits': int(fit is not None), 'fit': fit}
