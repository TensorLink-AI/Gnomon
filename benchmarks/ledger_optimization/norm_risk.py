"""Conditional error Gram targets scaled by each training case's anchor norm."""
import hashlib
import json
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .guarded_correction import MODELS, RECIPE, features
from .local_risk import gram_targets, predict_matrices, combine

SMOOTHING=1e-6


def targets(point,actual,anchor):
    grams=gram_targets(point,actual)
    a=np.asarray(anchor,dtype=float)
    if a.shape!=(6,) or not np.isfinite(a).all() or (a<0).any() or abs(a.sum()-1)>1e-8:
        raise ValueError('Finite simplex anchor required')
    errors=np.log1p(np.asarray([point[m] for m in MODELS],dtype=float).T)-np.log1p(np.asarray(actual,dtype=float))[:,None]
    norm=float(np.sqrt(np.mean((errors@a)**2)+SMOOTHING**2))
    scale=1/(2*norm)
    return grams*scale,{'anchor_smoothed_rmsle':norm,'gram_scale':scale,
                         'majorizer_constant':norm/2+SMOOTHING**2/(2*norm)}


def fit(pairs,masses,anchor):
    q=np.asarray(masses,dtype=float)
    if not pairs or q.shape!=(len(pairs),) or not np.isfinite(q).all() or (q<0).any() or abs(q.sum()-1)>1e-10:
        raise ValueError('Normalized nonnegative case masses required')
    x=np.concatenate([features(p['point'],anchor)[0] for p in pairs])
    prepared=[targets(p['point'],p['actual'],anchor) for p in pairs]
    y=np.concatenate([p[0] for p in prepared]);model=ExtraTreesRegressor(**RECIPE)
    model.fit(x,y.reshape(-1,36),sample_weight=np.repeat(q/24,24));trees=[]
    for estimator in model.estimators_:
        t=estimator.tree_;trees.append({k:getattr(t,k).tolist() for k in ('children_left','children_right','feature','threshold','value','n_node_samples','weighted_n_node_samples')})
    info={'anchor':anchor,'recipe':model.get_params(),'case_count':len(pairs),'row_count':len(x),
          'tree_count':len(trees),'trees':trees,'case_scaling':[p[1] for p in prepared],
          'smoothing_epsilon':SMOOTHING,
          'input_sha256':hashlib.sha256(json.dumps({'pairs':pairs,'masses':masses,'anchor':anchor},separators=(',',':')).encode()).hexdigest()}
    return model,info
