"""Causal selection of a fixed evidence-strength grid from executed trials."""
from datetime import datetime
import math

STRENGTHS=(0.,.25,.5,.75,1.)
KEYS=('0','0.25','0.5','0.75','1')
TIE_ORDER=('0.5','0.25','0.75','0','1')


def blend_inputs(current,past,strength):
    if isinstance(strength,bool) or strength not in STRENGTHS:raise ValueError('Strength must be one of0,.25,.5,.75,1')
    if len(current)!=3 or (past and len(past)!=16):raise ValueError('Three current folds and either zero or sixteen matched past pairs required')
    effective=float(strength) if past else 0.;pairs=[];masses=[]
    if effective<1:pairs.extend(current);masses.extend([(1-effective)/3]*3)
    if effective>0:pairs.extend(past);masses.extend([effective/16]*16)
    return {'pairs':pairs,'masses':masses,'requested_strength':float(strength),'effective_strength':effective}


def instant(value):
    at=datetime.fromisoformat(value)
    if at.tzinfo is None:raise ValueError('Explicit timezone required')
    return at


def risk(point,actual):
    if len(point)!=24 or len(actual)!=24 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in list(point)+list(actual)):
        raise ValueError('Complete finite nonnegative24-step candidate/actual pairs required')
    return math.sqrt(math.fsum((math.log1p(p)-math.log1p(y))**2 for p,y in zip(point,actual,strict=True))/24)


def choose(current,records,neighbors):
    now=instant(current['origin']);pool={}
    for r in records:
        if r['arm']!=current['arm'] or r['domain']!=current['domain']:continue
        origin,close,source,recorded=(instant(r[k]) for k in ('origin','last_target','source_available_at','recorded_at'))
        if origin>=now or close>now or source>now or recorded>now:continue
        if close<=origin or source<close or recorded<close:raise ValueError('Inconsistent mature outcome chronology')
        identity=r['series_id'],origin
        if identity in pool:raise ValueError('Duplicate eligible trial identity')
        pool[identity]=r
    keys=[(n['series_id'],instant(n['origin'])) for n in neighbors]
    if len(set(keys))!=len(keys) or len(keys)>16:raise ValueError('Unique bounded neighbor cohort required')
    # Insufficient trial coverage is explicit; never evaluate an unmatched subset.
    missing=[{'series_id':s,'origin':o.isoformat()} for s,o in keys if (s,o) not in pool]
    ready=len(keys)==16 and len({o for _,o in keys})>=3 and not missing
    result={'requested_strength':.5,'selection_basis':'insufficient_matched_history','ready':ready,'missing_trial_records':missing,
        'candidate_score_means':None,'ties':[],'cohort':[],'eligible_records':len(pool),'provider_calls':0,'new_weight_fits':0}
    if not ready:return result
    scores={k:[] for k in KEYS}
    for key in keys:
        r=pool[key]
        if set(r['candidates'])!=set(KEYS):raise ValueError('Every matched record must contain all five executed candidates')
        ids=[]
        for candidate in KEYS:
            trial=r['candidates'][candidate]
            if not isinstance(trial['execution_id'],str) or not trial['execution_id']:raise ValueError('Executed candidate identity required')
            if instant(trial['forecast_recorded_at'])>instant(r['origin']):raise ValueError('Candidate was not recorded by its forecast origin')
            if isinstance(trial['requested_strength'],bool):raise ValueError('Numeric candidate strength required')
            if trial['task_id']!=r['task_id'] or trial['arm']!=r['arm'] or trial['requested_strength']!=float(candidate):raise ValueError('Candidate/task identity mismatch')
            scores[candidate].append(risk(trial['point'],r['actual']));ids.append(trial['execution_id'])
        # Cache reuse may share a physical result, but each logical candidate has
        # its own execution identity and requested-strength provenance.
        if len(set(ids))!=5:raise ValueError('Distinct logical candidate executions required')
        result['cohort'].append({k:r[k] for k in ('task_id','series_id','arm','domain','origin','last_target','source_available_at','recorded_at')})
    means={k:math.fsum(v)/len(v) for k,v in scores.items()};minimum=min(means.values());ties=[k for k in TIE_ORDER if means[k]==minimum]
    result.update(requested_strength=float(ties[0]),selection_basis='lowest_matched_historical_candidate_rmsle',candidate_score_means=means,ties=ties)
    return result
