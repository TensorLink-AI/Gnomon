"""Numerical exact-RMSLE solver with smooth search and an exact-objective bound."""
import hashlib
import json
import numpy as np
from scipy.optimize import minimize, brentq
from .evidence_ensemble import arrays, objective as exact_objective

EPSILON=1e-6


def evaluate(w,prepared,masses):
    smooth=1e-6*np.square(w-1/6).sum();gradient=2e-6*(w-1/6);slack=0.
    for (point,actual),mass in zip(prepared,masses,strict=True):
        error=point@w-actual;n2=float(np.mean(error**2));n=float(np.sqrt(n2));s=float(np.sqrt(n2+EPSILON**2))
        smooth += mass*s;gradient += mass*(point.T@error)/(24*s)
        # The chosen norm support vector has length<=1, including exact fits.
        slack += mass*(n-n2/s)
    gap=float(w@gradient-gradient.min()+slack)
    return float(smooth),gradient,gap,float(slack)


def fit(pairs,masses):
    q=np.asarray(masses,dtype=float)
    if not pairs or q.shape!=(len(pairs),) or not np.isfinite(q).all() or (q<0).any() or abs(q.sum()-1)>1e-10:raise ValueError('Normalized nonnegative masses required')
    prepared=[arrays(p) for p in pairs];initial=np.full(6,1/6)
    result=minimize(lambda w:evaluate(w,prepared,q)[:2],initial,method='SLSQP',jac=True,
        bounds=[(0.,1.)]*6,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(6)},
        options={'ftol':1e-14,'maxiter':500})
    w=result.x
    if not np.isfinite(w).all() or w.min() < -1e-8 or abs(w.sum()-1)>1e-8:raise ValueError('Infeasible global mixture')
    w=np.maximum(w,0.);w/=w.sum();start=w.tolist();refinements=[]
    for _ in range(32):
        _,gradient,gap,_=evaluate(w,prepared,q)
        if gap<=1e-5:break
        direction=np.eye(6)[gradient.argmin()]-w
        def derivative(step):return float(evaluate(w+step*direction,prepared,q)[1]@direction)
        if derivative(0.)>=0:break
        step=1. if derivative(1.)<=0 else brentq(derivative,0.,1.,xtol=1e-15,rtol=1e-14)
        w=w+step*direction;refinements.append({'gap_before':gap,'step':float(step)})
    smooth,_,gap,slack=evaluate(w,prepared,q);value=exact_objective(w,prepared,q)[0];initial_value=exact_objective(initial,prepared,q)[0]
    info={'weights':w.tolist(),'objective':value,'initial_objective':initial_value,'convex_gap_bound':gap,
        'dual_support_slack':slack,'search_objective':smooth,'search_epsilon':EPSILON,'maximum_search_perturbation':EPSILON,
        'certificate_kind':'exact_objective_norm_dual_support','success':bool(result.success),'solver_status':int(result.status),
        'solver_message':str(result.message),'iterations':int(result.nit),'refinement_start_weights':start,'certificate_refinements':refinements,
        'input_sha256':hashlib.sha256(json.dumps({'pairs':pairs,'masses':masses},separators=(',',':')).encode()).hexdigest()}
    if not result.success or gap>1e-5 or gap < -1e-10 or value>initial_value+1e-8 or not(-1e-10<=smooth-value<=EPSILON+1e-10):
        error=ValueError('Exact global mixture certificate failed');error.certificate=info;raise error
    return info
