"""Frozen numerical search diagnostic, with choices isolated from current targets."""
from copy import deepcopy
from datetime import datetime
import math
from statistics import mean

CONFIGS = [
    {'model':'seasonal','season':7}, {'model':'seasonal','season':14},
    {'model':'seasonal','season':28},
    {'model':'ridge','window':90,'lags':7,'alpha':10.},
    {'model':'ridge','window':90,'lags':28,'alpha':100.},
    {'model':'ridge','window':365,'lags':14,'alpha':10.},
    {'model':'ridge','window':365,'lags':28,'alpha':10.},
    {'model':'ridge','window':730,'lags':28,'alpha':1.},
    {'model':'ridge','window':730,'lags':56,'alpha':1000.},
    {'model':'random_forest','window':365,'lags':14,'depth':4},
    {'model':'random_forest','window':730,'lags':28,'depth':8},
]
ENDS = (688,702,716,730)
POLICIES = ('current_cv','recent_history','lifetime_history','lifetime_support')


def instant(value):
    result=datetime.fromisoformat(value)
    if result.utcoffset() is None:raise ValueError('Explicit timezone required')
    return result


def scores(values):
    if (type(values) is not list or len(values)!=len(CONFIGS)
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values)):
        raise ValueError('Complete finite eleven-configuration scores required')
    return values


def rmsle(point, actual):
    if type(point) is not list or type(actual) is not list or len(point)!=14 or len(actual)!=14:
        raise ValueError('Fourteen forecast points and actuals required')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in point+actual):
        raise ValueError('Finite nonboolean values required')
    if any(v<0 for v in actual):raise ValueError('Negative actuals rejected')
    return math.sqrt(mean((math.log1p(max(p,0))-math.log1p(a))**2 for p,a in zip(point,actual,strict=True)))


def requests_for_job(job):
    """Replicate frozen core.request_at using observed rows and known covariates.

    Production actuals are never consumed here. Request season remains the
    task metadata value 7; the separate model configuration owns model season.
    """
    r=job['request']
    if (len(r['history'])!=730 or len(r['timestamps'])!=730 or len(r['past_covariates'])!=730
            or len(r['future_timestamps'])!=14 or len(r['future_covariates'])!=14
            or r['horizon']!=14 or r['cutoff']!=job['origin']
            or r['timestamps'][-1]!=job['origin'] or r['series_id']!=job['series_id']):
        raise ValueError('Frozen production task shape and identity required')
    if r['past_covariate_names']!=r['future_covariate_names']:
        raise ValueError('Shared covariate names required')
    times=r['timestamps']+r['future_timestamps'];cov=r['past_covariates']+r['future_covariates']
    if any(instant(b)<=instant(a) for a,b in zip(times,times[1:])):
        raise ValueError('Strictly ordered explicit timestamps required')
    result={}
    for end in ENDS:
        at=times[end-1]
        request={'history':[float(v) for v in r['history'][:end]],'timestamps':times[:end],
            'future_timestamps':times[end:end+14],
            'past_covariates':deepcopy(cov[:end]),'future_covariates':deepcopy(cov[end:end+14]),
            'past_covariate_names':list(r['past_covariate_names']),
            'future_covariate_names':list(r['future_covariate_names']),
            'horizon':14,'series_id':r['series_id'],'unit':r['unit'],
            'cutoff':at,'frequency':'D','season':7,'known_time_cutoff':at}
        actual=[float(v) for v in r['history'][end:end+14]] if end<730 else None
        if actual is not None and instant(request['future_timestamps'][-1])>instant(job['origin']):
            raise ValueError('Current CV targets are not yet visible')
        result[end]={'request':request,'actual':actual}
    return result


def choose(current_cv, prior, origin, *, series_id):
    """Current production scores/targets are deliberately absent from this API."""
    scores(current_cv); now=instant(origin); visible=[];seen=set()
    excluded={'source':0,'recording':0,'not_prior':0}
    for row in prior:
        if row['series_id']!=series_id:raise ValueError('Cross-series history rejected')
        at=instant(row['origin'])
        if at in seen:raise ValueError('Duplicate historical origin')
        seen.add(at);scores(row['scores'])
        source=instant(row['target_end']);recording=instant(row['recorded_at'])
        if source<=at or recording<source:raise ValueError('Invalid horizon availability')
        blocked=False
        for key,failed in (('source',source>now),('recording',recording>now),('not_prior',at>=now)):
            if failed:excluded[key]+=1;blocked=True
        if not blocked:visible.append(row)
    visible.sort(key=lambda r:instant(r['origin']))
    cv=min(range(len(CONFIGS)),key=lambda i:current_cv[i])
    choices={p:cv for p in POLICIES}
    if len(visible)>=4:
        means=[mean(r['scores'][i] for r in visible) for i in range(len(CONFIGS))]
        recent=[mean(r['scores'][i] for r in visible[-4:]) for i in range(len(CONFIGS))]
        best=min(range(len(CONFIGS)),key=lambda i:means[i])
        choices['recent_history']=min(range(len(CONFIGS)),key=lambda i:recent[i])
        choices['lifetime_history']=best
        wins=sum(r['scores'][best]<r['scores'][cv] for r in visible)
        if means[best]<means[cv] and 2*wins>=len(visible):choices['lifetime_support']=best
    return {'choices':choices,'visible_origins':[r['origin'] for r in visible],
            'matched_origins':len(visible),'exclusions':excluded,
            'tie_policy':'first_in_frozen_configuration_order','current_production_outcomes_used':False}
