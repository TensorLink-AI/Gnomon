"""Past-only evidence and validation-gated nonlinear error correction."""
from datetime import datetime
import hashlib,json,math
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

MODELS=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
RECIPE={'n_estimators':32,'max_depth':4,'min_samples_leaf':24,'max_features':1.0,'bootstrap':False,'random_state':17,'n_jobs':1,'criterion':'squared_error'}


def instant(value):
    at=datetime.fromisoformat(value)
    if at.tzinfo is None:raise ValueError('Timezone-aware instant required')
    return at


def visible(records,origin,domain):
    now=instant(origin);result=[];identities=set()
    for record in records:
        if record['domain']!=domain:continue
        at,end,source,recorded=(instant(record[k]) for k in ('origin','last_target','source_available_at','recorded_at'))
        if at>=now or end>now or source>now or recorded>now:continue
        if end<=at or source<end or recorded<end:raise ValueError('Inconsistent historical chronology')
        key=record['series_id'],at
        if key in identities:raise ValueError('Duplicate visible historical execution')
        identities.add(key);result.append(record)
    return sorted(result,key=lambda r:(instant(r['origin']),r['series_id']))


def training(current,past):
    if len(current) not in (2,3):raise ValueError('Two validation-training or three production-training CV folds required')
    pairs=list(current)+list(past);masses=([.5/len(current)]*len(current)+[.5/len(past)]*len(past)) if past else [1/len(current)]*len(current)
    return pairs,masses


def features(point,weights):
    if set(point)!=set(MODELS):raise ValueError('Exactly six fixed model forecasts required')
    p=np.asarray([point[m] for m in MODELS],dtype=float).T;w=np.asarray(weights,dtype=float)
    if p.shape!=(24,6) or w.shape!=(6,) or not np.isfinite(p).all() or not np.isfinite(w).all() or (p<0).any() or (w<0).any() or abs(w.sum()-1)>1e-8:raise ValueError('Complete nonnegative forecasts and simplex weights required')
    logs=np.log1p(p);base=logs@w;location=float(base.mean());scale=max(float(base.std()),.1);h=np.arange(24)*2*np.pi/24
    return np.column_stack(((logs-location)/scale,base,np.full(24,scale),np.sin(h),np.cos(h))),base


def fit(pairs,masses,weights):
    q=np.asarray(masses,dtype=float)
    if not pairs or q.shape!=(len(pairs),) or not np.isfinite(q).all() or (q<0).any() or abs(q.sum()-1)>1e-10:raise ValueError('Normalized nonnegative case masses required')
    x=[];y=[]
    for pair in pairs:
        design,base=features(pair['point'],weights);actual=np.asarray(pair['actual'],dtype=float)
        if actual.shape!=(24,) or not np.isfinite(actual).all() or (actual<0).any():raise ValueError('Complete nonnegative actuals required')
        x.append(design);y.append(np.log1p(actual)-base)
    x=np.concatenate(x);y=np.concatenate(y);sample_weights=np.repeat(q/24,24);model=ExtraTreesRegressor(**RECIPE);model.fit(x,y,sample_weight=sample_weights)
    trees=[]
    for estimator in model.estimators_:
        t=estimator.tree_;trees.append({k:getattr(t,k).tolist() for k in ('children_left','children_right','feature','threshold','value','n_node_samples','weighted_n_node_samples')})
    info={'weights':weights,'recipe':model.get_params(),'case_count':len(pairs),'row_count':len(x),'tree_count':len(trees),'trees':trees,'training_sha256':hashlib.sha256(json.dumps({'pairs':pairs,'masses':masses,'weights':weights},separators=(',',':')).encode()).hexdigest()}
    return model,info


def apply(point,weights,model):
    x,base=features(point,weights);correction=model.predict(x);raw=base+correction;point=np.expm1(np.maximum(raw,0.))
    if not np.isfinite(point).all():raise ValueError('Nonfinite corrected forecast')
    return {'point':point.tolist(),'base_point':np.expm1(base).tolist(),'correction_log':correction.tolist(),'raw_predicted_log':raw.tolist(),'clipped_leads':np.where(raw<0)[0].tolist()}


def risk(point,actual):
    if len(point)!=24 or len(actual)!=24 or any(not math.isfinite(v) or v<0 for v in list(point)+list(actual)):raise ValueError('Complete nonnegative24-step scores required')
    return math.sqrt(math.fsum((math.log1p(p)-math.log1p(y))**2 for p,y in zip(point,actual,strict=True))/24)


def decide(base,corrected,actual):
    before=risk(base,actual);after=risk(corrected,actual);enabled=before-after>1e-12
    return {'correction_enabled':enabled,'baseline_rmsle':before,'corrected_rmsle':after,'improvement':before-after,'cause':'lower_held_back_fold_rmsle' if enabled else 'no_validated_improvement','threshold':1e-12}
