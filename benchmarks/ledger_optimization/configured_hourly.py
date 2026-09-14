"""Common064 configurable hourly recipes; exact038 implementations and features."""
import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .hourly_numerical import RECIPES

WINDOWS=(336,504,730)
LAGS=(24,48,168)
ALPHAS=(.01,.1,1.,10.,100.,1000.,10000.)
FOREST_WINDOWS=(336,730)
FOREST_LAGS=(24,48)
DEPTHS=(3,6,12)


def configuration(raw):
    if not isinstance(raw,dict):raise ValueError('Configuration must be an object')
    kind=raw.get('kind')
    choices={'seasonal':{'season':(24,168)},'weekly_mean':{},
        'ridge':{'window':WINDOWS,'lags':LAGS,'alpha':ALPHAS},
        'forest':{'window':FOREST_WINDOWS,'lags':FOREST_LAGS,'depth':DEPTHS}}
    if not isinstance(kind,str) or kind not in choices:raise ValueError('Unknown configuration kind')
    fields=choices[kind]
    if set(raw)!={'kind',*fields}:raise ValueError('Supply exactly the fields for this configuration kind')
    result={'kind':kind}
    for field,allowed in fields.items():
        value=raw[field]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value not in allowed:
            raise ValueError('Value outside common catalogue: '+field)
        result[field]=float(value) if field=='alpha' else int(value)
    return result


def config_id(raw):
    canonical=configuration(raw)
    return hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def catalog():
    ordered=[configuration(r) for r in RECIPES.values()]
    rest=[{'kind':'ridge','window':w,'lags':l,'alpha':a} for w in WINDOWS for l in LAGS for a in ALPHAS]
    rest += [{'kind':'forest','window':w,'lags':l,'depth':d} for w in FOREST_WINDOWS for l in FOREST_LAGS for d in DEPTHS]
    seen={config_id(r) for r in ordered}
    for r in sorted(rest,key=config_id):
        if config_id(r) not in seen:ordered.append(configuration(r));seen.add(config_id(r))
    return [{'config_id':config_id(r),'config':r} for r in ordered]


def revision():
    return 'configured_hourly064/sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_labels(history_labels,future_labels):
    labels=list(history_labels)+list(future_labels)
    if not history_labels or len(future_labels)!=24:raise ValueError('Observed history and24 future labels required')
    if any(t.tzinfo is None or t.utcoffset()!=timedelta(0) for t in labels):raise ValueError('Explicit nominal UTC coordinates required')
    if any(b-a!=timedelta(hours=1) for a,b in zip(labels,labels[1:])):raise ValueError('Contiguous hourly source phase required')


def calendar_features(labels):
    return np.asarray([[math.sin(2*math.pi*t.hour/24), math.cos(2*math.pi*t.hour/24),
                        math.sin(2*math.pi*t.weekday()/7), math.cos(2*math.pi*t.weekday()/7)]
                       for t in labels])


def predict(history, history_labels, future_labels, raw_config):
    config = configuration(raw_config)
    validate_labels(history_labels, future_labels)
    history = np.asarray(history, dtype=float)
    if len(history) != len(history_labels) or not np.isfinite(history).all() or (history < 0).any():
        raise ValueError('Invalid aligned observed history')
    horizon = len(future_labels)
    if horizon != 24 or len(history) < 658:
        raise ValueError('Expected 24 targets and at least 658 observations')
    if config['kind'] == 'seasonal':
        season = config['season']
        return [float(history[-season+i % season]) for i in range(horizon)]
    if config['kind'] == 'weekly_mean':
        return [float(np.mean([history[-168*k+i] for k in range(1, 4)])) for i in range(horizon)]
    history = history[-config['window']:]
    labels = history_labels[-len(history):]
    cov = calendar_features(labels)
    future = calendar_features(future_labels)
    z = np.log1p(history)
    lag = config['lags']

    def features(prior, c):
        return np.r_[prior[-lag:][::-1], np.mean(prior[-24:]), np.mean(prior[-168:]), c]

    x = np.asarray([features(z[:i], cov[i]) for i in range(lag, len(z))])
    y = z[lag:]
    if config['kind'] == 'ridge':
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=config['alpha']))
    else:
        estimator = RandomForestRegressor(n_estimators=80, max_depth=config['depth'],
                                         min_samples_leaf=3, random_state=17, n_jobs=1)
    estimator.fit(x, y)
    prior = z.tolist()
    points = []
    for c in future:
        value = float(estimator.predict(features(prior, c).reshape(1, -1))[0])
        value = max(0., min(value, math.log1p(max(history.max()*100, 1000))))
        prior.append(value)
        points.append(math.expm1(value))
    return points
