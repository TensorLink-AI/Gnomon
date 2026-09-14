"""Ridge regression of fixed-model next-day signed errors on observed CV errors."""
import hashlib,json
import numpy as np

MODELS=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
PENALTY=.1


def errors(point,actual):
    if set(point)!=set(MODELS):raise ValueError('Exactly six fixed configurations required')
    p=np.asarray([point[m] for m in MODELS],dtype=float);y=np.asarray(actual,dtype=float)
    if p.shape!=(6,24) or y.shape!=(24,) or not np.isfinite(p).all() or not np.isfinite(y).all() or (p<0).any() or (y<0).any():raise ValueError('Complete finite nonnegative model/actual pairs required')
    return np.log1p(y)[None,:]-np.log1p(p)


def fit(transitions,masses):
    q=np.asarray(masses,dtype=float);previous=np.asarray([r['previous_error'] for r in transitions],dtype=float);following=np.asarray([r['next_error'] for r in transitions],dtype=float)
    if not transitions or q.shape!=(len(transitions),) or previous.shape!=(len(transitions),6,24) or following.shape!=previous.shape or not np.isfinite(q).all() or not np.isfinite(previous).all() or not np.isfinite(following).all() or (q<0).any() or abs(q.sum()-1)>1e-10:raise ValueError('Complete six-model error transitions with normalized masses required')
    systems={}
    for i,m in enumerate(MODELS):
        x=np.stack((np.ones_like(previous[:,i]),previous[:,i]),axis=-1);y=following[:,i];gram=np.einsum('n,nhi,nhj->ij',q/24,x,x);rhs=np.einsum('n,nhi,nh->i',q/24,x,y);matrix=gram+PENALTY*np.eye(2);theta=np.linalg.solve(matrix,rhs);residual=np.einsum('nhi,i->nh',x,theta)-y
        value=float(q@np.mean(residual**2,axis=1)+PENALTY*(theta@theta));initial=float(q@np.mean(y**2,axis=1));gradient=2*(matrix@theta-rhs);eigen=np.linalg.eigvalsh(matrix)
        if not np.isfinite(theta).all() or eigen.min()<PENALTY-1e-10 or np.max(np.abs(gradient))>1e-9 or value>initial+1e-9:raise ValueError('Invalid ridge normal-equation certificate')
        systems[m]={'coefficients':theta.tolist(),'gram':gram.tolist(),'rhs':rhs.tolist(),'regularized_matrix':matrix.tolist(),'eigenvalues':eigen.tolist(),'gradient':gradient.tolist(),'objective':value,'zero_objective':initial}
    return {'systems':systems,'case_count':len(transitions),'rows_per_model':24*len(transitions),'penalty':PENALTY,'input_sha256':hashlib.sha256(json.dumps({'transitions':transitions,'masses':masses},separators=(',',':')).encode()).hexdigest()}


def apply(point,last_error,weights,fitted):
    p=np.asarray([point[m] for m in MODELS],dtype=float);e=np.asarray(last_error,dtype=float);w=np.asarray(weights,dtype=float)
    if p.shape!=(6,24) or e.shape!=p.shape or w.shape!=(6,) or not np.isfinite(p).all() or not np.isfinite(e).all() or not np.isfinite(w).all() or (p<0).any() or (w<0).any() or abs(w.sum()-1)>1e-8:raise ValueError('Complete prediction/error and simplex weights required')
    correction=np.array([fitted['systems'][m]['coefficients'][0]+fitted['systems'][m]['coefficients'][1]*e[i] for i,m in enumerate(MODELS)]);raw=np.log1p(p)+correction;logs=np.maximum(raw,0.);combined=np.expm1(w@logs)
    if not np.isfinite(combined).all():raise ValueError('Nonfinite corrected forecast')
    return {'point':combined.tolist(),'predicted_error':{m:correction[i].tolist() for i,m in enumerate(MODELS)},'corrected_model_points':{m:np.expm1(logs[i]).tolist() for i,m in enumerate(MODELS)},'clipped_leads':{m:np.where(raw[i]<0)[0].tolist() for i,m in enumerate(MODELS)}}
