"""Shared065 sequential search; optional retrieval of prior executed backtests."""
from datetime import datetime
import hashlib
import json
import math
import numpy as np
from .configured_hourly import catalog,configuration,config_id,revision

RIDGE=.05
EXPLORATION=.5
NUMERICAL_LIMIT=60
STARTER_COST=24
EXTRA_CONFIGURATIONS=11


def admit_backtest(used):
    if not isinstance(used,int) or isinstance(used,bool) or used<0 or used+3+1>NUMERICAL_LIMIT:
        raise ValueError('Three backtest attempts and one final attempt must remain')


def vector(raw):
    r=configuration(raw);kind=r['kind']
    return [float(kind==k) for k in ('seasonal','weekly_mean','ridge','forest')]+[
        (r.get('window',336)-336)/(730-336),
        math.log(r.get('lags',24)/24)/math.log(7),
        (math.log10(r.get('alpha',.01))+2)/6,
        (r.get('depth',3)-3)/9,
        math.log(r.get('season',24)/24)/math.log(7)]


def checked_backtests(rows):
    out={}
    for row in rows:
        config=configuration(row['config']);key=config_id(config);value=row['cv_rmsle']
        if row['config_id']!=key or key in out:raise ValueError('Duplicate or inconsistent configuration identity')
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative observed CV RMSLE required')
        out[key]={'config':config,'config_id':key,'cv_rmsle':float(value)}
    starters=[r['config_id'] for r in catalog()[:6]]
    if not set(starters)<=set(out):raise ValueError('Six complete common starter backtests required')
    base=np.log1p([out[k]['cv_rmsle'] for k in starters]);location=float(base.mean());scale=max(.01,float(base.std()))
    return out,location,scale


def visible_history(current,history):
    now=datetime.fromisoformat(current['origin'])
    if now.tzinfo is None:raise ValueError('Explicit current origin required')
    pool=[];seen=set();expected_revision=revision()
    for row in history:
        if row['domain']!=current['domain'] or row['arm']!=current['arm']:continue
        origin,source,recorded=(datetime.fromisoformat(row[k]) for k in ('origin','source_available_at','recorded_at'))
        if any(t.tzinfo is None for t in (origin,source,recorded)):raise ValueError('Explicit evidence clocks required')
        if origin>=now or source>now or recorded>now:continue
        key=row['series_id'],origin
        if key in seen or source>origin or recorded<origin:raise ValueError('Duplicate or inconsistent backtest recording')
        if row['evidence_kind']!='executed_backtests' or row['revision']!=expected_revision:raise ValueError('Comparable executed backtest revision required')
        seen.add(key);pool.append(row)
    return sorted(pool,key=lambda r:(r['origin'],r['series_id']))


def neighborhood(current,history):
    pool=visible_history(current,history);query=np.asarray(current['features'],dtype=float)
    if query.shape!=(12,) or not np.isfinite(query).all():raise ValueError('Twelve predecision query features required')
    ready=len(pool)>=16 and len({r['origin'] for r in pool})>=3
    if not pool:return [],{'ready':False,'eligible':[],'selected':[],'scale':[1.]*12}
    x=np.asarray([r['features'] for r in pool],dtype=float)
    if x.shape!=(len(pool),12) or not np.isfinite(x).all():raise ValueError('Finite prior predecision features required')
    spread=np.maximum(.1,x.std(axis=0));distances=np.sum(((x-query)/spread)**2,axis=1)
    candidates=[{'series_id':r['series_id'],'origin':r['origin'],'arm':r['arm'],'distance':float(d)} for r,d in zip(pool,distances,strict=True)]
    order=sorted(range(len(pool)),key=lambda i:(distances[i],pool[i]['origin'],pool[i]['series_id']))[:16] if ready else []
    return [pool[i] for i in order],{'ready':ready,'eligible':candidates,'selected':[candidates[i] for i in order],'scale':spread.tolist()}


def kernel(config_a,context_a,config_b,context_b,scale):
    distance=np.sum((config_a[:,None,:]-config_b[None,:,:])**2,axis=2)
    context_distance=np.mean(((context_a[:,None,:]-context_b[None,:,:])/scale)**2,axis=2)
    return np.exp(-2*distance-.5*context_distance)


def posterior(configs,contexts,targets,masses,query_configs,query_context,scale):
    c=np.asarray(configs,dtype=float);x=np.asarray(contexts,dtype=float);y=np.asarray(targets,dtype=float);q=np.asarray(masses,dtype=float)
    qc=np.asarray(query_configs,dtype=float);qx=np.tile(np.asarray(query_context,dtype=float),(len(qc),1));scale=np.asarray(scale,dtype=float)
    if c.shape!=(len(y),9) or x.shape!=(len(y),12) or qc.ndim!=2 or qc.shape[1]!=9 or qx.shape!=(len(qc),12) or q.shape!=(len(y),) or scale.shape!=(12,):raise ValueError('Complete search training arrays required')
    if any(not a.size or not np.isfinite(a).all() for a in (c,x,y,q,qc,qx,scale)) or (q<=0).any() or (scale<=0).any() or abs(q.sum()-1)>1e-10:raise ValueError('Valid finite normalized training mass required')
    gram=kernel(c,x,c,x,scale)+np.diag(RIDGE/q);cross=kernel(c,x,qc,qx,scale)
    solved=np.linalg.solve(gram,np.column_stack((y,cross)))
    mean=cross.T@solved[:,0];variance=1-np.sum(cross*solved[:,1:],axis=0)
    if variance.min() < -1e-8 or variance.max()>1+1e-8:raise ValueError('Invalid kernel predictive variance')
    return mean,np.sqrt(np.clip(variance,0,1))


def suggest(current,backtests,history=()):
    tested,location,target_scale=checked_backtests(backtests);inventory=catalog();available=[r for r in inventory if r['config_id'] not in tested]
    if not available:raise ValueError('No untested configurations remain')
    past,retrieval=neighborhood(current,history);features=[];contexts=[];targets=[];masses=[];refs=[]
    def add(episode,observed,total_mass,kind):
        records,center,spread=checked_backtests(observed)
        for key,row in sorted(records.items()):
            features.append(vector(row['config']));contexts.append(list(episode['features']));targets.append((math.log1p(row['cv_rmsle'])-center)/spread)
            masses.append(total_mass/len(records));refs.append({'series_id':episode['series_id'],'origin':episode['origin'],'arm':episode['arm'],'config_id':key,'evidence_kind':kind})
    add(current,backtests,.5 if past else 1.,'current_backtest')
    for episode in past:add(episode,episode['backtests'],.5/len(past),'prior_executed_backtest')
    training={'config_features':features,'context_features':contexts,'targets':targets,'masses':masses,'refs':refs,'context_scale':retrieval['scale']}
    mean,spread=posterior(features,contexts,targets,masses,[vector(r['config']) for r in available],current['features'],retrieval['scale'])
    ranking=[{'config_id':r['config_id'],'config':r['config'],'predicted_standardized_cv':float(m),'kernel_spread':float(s),'acquisition':float(m-EXPLORATION*s)} for r,m,s in zip(available,mean,spread,strict=True)]
    order={r['config_id']:i for i,r in enumerate(inventory)};ranking.sort(key=lambda r:(r['acquisition'],order[r['config_id']]))
    return {'next_config':ranking[0]['config'],'next_config_id':ranking[0]['config_id'],'ranking':ranking,'retrieval':retrieval,
        'training_sha256':hashlib.sha256(json.dumps(training,separators=(',',':')).encode()).hexdigest(),
        'training_records':len(refs),'current_backtests':len(tested),'prior_episodes':len(past),
        'current_target_location':location,'current_target_scale':target_scale,'kernel_ridge':RIDGE,'exploration':EXPLORATION,
        'provider_calls':0,'surrogate_solves':1,'uncertainty_scope':'kernel_spread_not_calibrated_confidence_interval',
        'evidence_scope':'executed_current_and_prior_backtests_not_unexecuted_or_future_production_outcomes'}


def select(backtests):
    tested,_,_=checked_backtests(backtests);order={r['config_id']:i for i,r in enumerate(catalog())}
    return min(tested.values(),key=lambda r:(r['cv_rmsle'],order[r['config_id']]))
