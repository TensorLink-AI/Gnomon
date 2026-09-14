"""Per-step conditional error Gram matrices with simplex forecast combination."""
import hashlib,json
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .guarded_correction import MODELS,RECIPE,features
from .conditional_risk import fit_weights


def gram_targets(point,actual):
    p=np.asarray([point[m] for m in MODELS],dtype=float).T;y=np.asarray(actual,dtype=float)
    if p.shape!=(24,6) or y.shape!=(24,) or not np.isfinite(p).all() or not np.isfinite(y).all() or (p<0).any() or (y<0).any():raise ValueError('Complete finite nonnegative24-step pairs required')
    e=np.log1p(p)-np.log1p(y)[:,None];return np.einsum('hi,hj->hij',e,e)


def fit(pairs,masses,anchor):
    q=np.asarray(masses,dtype=float)
    if not pairs or q.shape!=(len(pairs),) or not np.isfinite(q).all() or (q<0).any() or abs(q.sum()-1)>1e-10:raise ValueError('Normalized nonnegative case masses required')
    x=np.concatenate([features(p['point'],anchor)[0] for p in pairs]);y=np.concatenate([gram_targets(p['point'],p['actual']) for p in pairs]);model=ExtraTreesRegressor(**RECIPE);model.fit(x,y.reshape(-1,36),sample_weight=np.repeat(q/24,24));trees=[]
    for estimator in model.estimators_:
        t=estimator.tree_;trees.append({k:getattr(t,k).tolist() for k in ('children_left','children_right','feature','threshold','value','n_node_samples','weighted_n_node_samples')})
    return model,{'anchor':anchor,'recipe':model.get_params(),'case_count':len(pairs),'row_count':len(x),'tree_count':len(trees),'trees':trees,'input_sha256':hashlib.sha256(json.dumps({'pairs':pairs,'masses':masses,'anchor':anchor},separators=(',',':')).encode()).hexdigest()}


def predict_matrices(model,point,anchor):
    x,_=features(point,anchor);matrices=model.predict(x).reshape(24,6,6)
    if not np.isfinite(matrices).all() or not np.allclose(matrices,matrices.transpose(0,2,1),atol=1e-10,rtol=1e-10) or np.linalg.eigvalsh(matrices).min()<-1e-9:raise ValueError('Invalid conditional error Gram matrix')
    return matrices


def combine(point,anchor,matrices):
    g=np.asarray(matrices,dtype=float)
    if g.shape!=(24,6,6):raise ValueError('One six-model Gram matrix per lead required')
    certificates=[fit_weights(matrix,anchor) for matrix in g];logs=np.log1p(np.asarray([point[m] for m in MODELS],dtype=float)).T;weights=np.asarray([r['weights'] for r in certificates]);forecast=np.expm1(np.sum(logs*weights,axis=1))
    if not np.isfinite(forecast).all():raise ValueError('Nonfinite combination')
    return {'point':forecast.tolist(),'weight_fits':certificates,'weight_fit_count':24,'weight_iterations':sum(r['iterations'] for r in certificates)}
