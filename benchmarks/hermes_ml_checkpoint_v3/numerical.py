"""Common actual model fitting for all arms; no Gnomon imports."""
import hashlib
import json
import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def configuration(raw):
    if not isinstance(raw, dict):
        raise ValueError('config must be an object')
    model = raw.get('model', 'ridge')
    keys = {'ridge': {'model','window','lags','alpha'},
            'random_forest': {'model','window','lags','depth'},
            'seasonal': {'model','season'}}
    if model not in keys or set(raw)-keys[model]:
        raise ValueError('Unknown model or irrelevant config field')
    config = {'model':model}
    bounds = {'season':(1,28,7)} if model=='seasonal' else {
        'window':(90,730,365),'lags':(7,56,14),
        **({'alpha':(.01,10000,10.)} if model=='ridge' else {'depth':(2,16,6)})}
    for k,(lo,hi,default) in bounds.items():
        v=raw.get(k,default)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError(f'{k} must be between {lo} and {hi}')
        if k!='alpha' and int(v)!=v:
            raise ValueError(k+' must be an integer')
        config[k]=float(v) if k=='alpha' else int(v)
    return config


def config_id(config):
    return hashlib.sha256(json.dumps(configuration(config),sort_keys=True).encode()).hexdigest()[:16]


def predict(request, config):
    config=configuration(config)
    history=np.asarray(request['history'],dtype=float)
    h=request['horizon']
    if config['model']=='seasonal':
        return [float(history[-config['season']+i%config['season']]) for i in range(h)]
    history=history[-config['window']:]
    cov=np.asarray(request['past_covariates'],dtype=float)[-len(history):]
    future=np.asarray(request['future_covariates'],dtype=float)
    lag=config['lags']
    z=np.log1p(history)
    if len(z)<lag+20:
        raise ValueError('Need at least lags+20 observed training rows')
    def features(prior, c):
        return np.r_[prior[-lag:][::-1],np.mean(prior[-7:]),np.mean(prior[-28:]),c]
    x=np.asarray([features(z[:i],cov[i]) for i in range(lag,len(z))])
    y=z[lag:]
    if config['model']=='ridge':
        estimator=make_pipeline(StandardScaler(),Ridge(alpha=config['alpha']))
    else:
        estimator=RandomForestRegressor(n_estimators=80,max_depth=config['depth'],
                                       min_samples_leaf=3,random_state=17,n_jobs=1)
    estimator.fit(x,y)
    prior=z.tolist();points=[]
    for i in range(h):
        value=float(estimator.predict(features(prior,future[i]).reshape(1,-1))[0])
        # Common, declared overflow guard, not fitted using future outcomes.
        value=max(0.,min(value,math.log1p(max(history.max()*100,1000))))
        prior.append(value);points.append(float(math.expm1(value)))
    return points


def metrics(point,actual):
    errors=[p-a for p,a in zip(point,actual,strict=True)]
    return {'n':len(errors),'mae':sum(map(abs,errors))/len(errors),
            'rmse':math.sqrt(sum(e*e for e in errors)/len(errors)),
            'bias':sum(errors)/len(errors),
            'rmsle':math.sqrt(sum((math.log1p(p)-math.log1p(a))**2
                                 for p,a in zip(point,actual,strict=True))/len(errors))}
