"""Predecision error-shape evidence indexing over frozen backtest folds."""
import numpy as np
from .context_ensemble import retrieve
from .evidence_ensemble import MODELS


def profile(folds):
    result=[]
    if set(folds)!=set(MODELS):raise ValueError('Complete model cohort required')
    common=None
    for model in MODELS:
        rows=folds[model]
        if len(rows)!=3 or [f['end'] for f in rows]!=[658,682,706]:raise ValueError('Three frozen CV folds required')
        actual=np.asarray([f['actual'] for f in rows],dtype=float)
        point=np.asarray([f['point'] for f in rows],dtype=float)
        if actual.shape!=(3,24) or point.shape!=(3,24) or not np.isfinite(actual).all() or not np.isfinite(point).all() or (actual<0).any() or (point<0).any():raise ValueError('Valid CV pairs required')
        if common is not None and not np.array_equal(common,actual):raise ValueError('Matched CV actuals required')
        common=actual
        error=np.log1p(point)-np.log1p(actual)
        for b in range(4):
            values=error[:,6*b:6*(b+1)]
            result.extend((float(values.mean()),float(np.sqrt(np.square(values).mean()))))
    return result


def retrieve_profile(current,episodes):
    base=retrieve(current,episodes)
    x=np.asarray(current['error_profile'],dtype=float)
    if x.shape!=(48,) or not np.isfinite(x).all():raise ValueError('Forty-eight finite predecision profile values required')
    base.update(error_location=None,error_scale=None,distance_rule='legacy_mean_square_plus_error_profile_mean_square')
    if not base['candidates']:return base
    index={(r['series_id'],r['origin']):r for r in episodes}
    past=np.asarray([index[r['series_id'],r['origin']]['error_profile'] for r in base['candidates']],dtype=float)
    if past.shape!=(len(base['candidates']),48) or not np.isfinite(past).all():raise ValueError('Complete historical error profiles required')
    location=past.mean(axis=0);scale=np.maximum(past.std(axis=0),.1)
    extra=np.square((past-x)/scale).mean(axis=1)
    candidates=[{**r,'legacy_distance':r['distance'],'legacy_component':r['distance']/12,
        'profile_component':float(d),'distance':r['distance']/12+float(d)} for r,d in zip(base['candidates'],extra,strict=True)]
    base.update(candidates=candidates,error_location=location.tolist(),error_scale=scale.tolist(),
        selected=sorted(candidates,key=lambda r:(r['distance'],r['origin'],r['series_id']))[:16] if base['ready'] else [])
    return base
