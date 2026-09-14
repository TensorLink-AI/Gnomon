"""Shared forecast-conditioned correction with strong-convexity certificate."""
import hashlib,json
import numpy as np
from scipy.optimize import minimize

MODELS=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest','search_selected')
PENALTY=.1


def raw_features(point,weights):
    p=np.asarray([point[m] for m in MODELS],dtype=float).T;w=np.asarray(weights,dtype=float)
    if p.shape!=(24,7) or w.shape!=(4,7) or not np.isfinite(p).all() or not np.isfinite(w).all() or (p<0).any() or (w<0).any() or np.max(np.abs(w.sum(axis=1)-1))>1e-8:raise ValueError('Complete predictions and four simplex weight rows required')
    logs=np.log1p(p);base=np.sum(logs*np.repeat(w,6,axis=0),axis=1)
    return np.column_stack((base,logs-base[:,None])),base


def design(raw,location,scale):
    z=(raw-np.asarray(location))/np.asarray(scale);h=np.arange(24)*2*np.pi/24
    return np.concatenate((np.ones((*raw.shape[:-1],1)),z,np.broadcast_to(np.column_stack((np.sin(h),np.cos(h))),(*raw.shape[:-2],24,2))),axis=-1)


def objective(theta,x,residual,masses):
    error=np.einsum('nhj,j->nh',x,theta)-residual;norm=np.sqrt(np.mean(error**2,axis=1)+1e-12)
    value=float(masses@norm+PENALTY*np.dot(theta,theta));gradient=np.einsum('n,nh,nhj->j',masses/(24*norm),error,x)+2*PENALTY*theta
    return value,gradient


def fit(pairs,masses,weights):
    q=np.asarray(masses,dtype=float)
    if not pairs or q.shape!=(len(pairs),) or not np.isfinite(q).all() or (q<0).any() or abs(q.sum()-1)>1e-10:raise ValueError('Normalized nonnegative case masses required')
    raw=[];bases=[];actual=[]
    for pair in pairs:
        z,b=raw_features(pair['point'],weights);y=np.asarray(pair['actual'],dtype=float)
        if y.shape!=(24,) or not np.isfinite(y).all() or (y<0).any():raise ValueError('Complete finite nonnegative targets required')
        raw.append(z);bases.append(b);actual.append(np.log1p(y))
    raw=np.asarray(raw);bases=np.asarray(bases);actual=np.asarray(actual);location=np.einsum('n,nhj->j',q,raw)/24
    variance=np.einsum('n,nhj->j',q,np.square(raw-location))/24;scale=np.maximum(np.sqrt(variance),.1)
    x=design(raw,location,scale);residual=actual-bases;zero=np.zeros(11)
    result=minimize(lambda t:objective(t,x,residual,q),zero,jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-14,'gtol':1e-9})
    theta=result.x;value,gradient=objective(theta,x,residual,q);initial=objective(zero,x,residual,q)[0];bound=float(np.dot(gradient,gradient)/(4*PENALTY))
    info={'coefficients':theta.tolist(),'location':location.tolist(),'scale':scale.tolist(),'weights':weights,'objective':value,'initial_objective':initial,'gradient':gradient.tolist(),'suboptimality_bound':bound,'strong_convexity':2*PENALTY,'penalty':PENALTY,'success':bool(result.success),'solver_status':int(result.status),'solver_message':str(result.message),'iterations':int(result.nit),'function_evaluations':int(result.nfev),'input_sha256':hashlib.sha256(json.dumps({'pairs':pairs,'masses':masses,'weights':weights},separators=(',',':')).encode()).hexdigest()}
    if not np.isfinite(theta).all() or not result.success or not np.isfinite(bound) or bound>1e-8 or value>initial+1e-10:
        error=ValueError('Conditional correction failed frozen convergence requirements');error.certificate=info;raise error
    return info


def apply(point,fitted):
    raw,base=raw_features(point,fitted['weights']);x=design(raw,fitted['location'],fitted['scale']);correction=x@np.asarray(fitted['coefficients']);logs=base+correction;p=np.expm1(np.maximum(logs,0.));original=np.asarray([point[m] for m in MODELS])
    if not np.isfinite(p).all():raise ValueError('Nonfinite derived prediction')
    return {'point':p.tolist(),'base_log':base.tolist(),'correction_log':correction.tolist(),'raw_predicted_log':logs.tolist(),'clipped_leads':np.where(logs<0)[0].tolist(),'outside_model_range_leads':np.where((p<original.min(axis=0)-1e-10)|(p>original.max(axis=0)+1e-10))[0].tolist()}
