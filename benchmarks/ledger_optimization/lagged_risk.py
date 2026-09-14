"""Conditional model risk with a visible predecessor's signed error context."""
from datetime import datetime, timedelta
import hashlib
import json
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .guarded_correction import MODELS, RECIPE, features as base_features
from .local_risk import gram_targets, combine


def validate_predecessor(meta, origin, config_ids):
    """Check metadata before reading predecessor payloads; absence is explicit."""
    if meta is None:
        return False
    def instant(s):
        d=datetime.fromisoformat(s)
        if d.tzinfo is None:
            raise ValueError('Predecessor and query require explicit timezones')
        return d
    now=instant(origin)
    old=instant(meta['origin']);end=instant(meta['last_target'])
    source=instant(meta['source_available_at']);recorded=instant(meta['recorded_at'])
    if not (old<end==now and end-old==timedelta(hours=24) and end<=source<=now and end<=recorded<=now):
        raise ValueError('Predecessor must be fully mature at this forecast origin')
    if set(config_ids)!=set(MODELS) or any(not isinstance(v,str) or not v for v in config_ids.values()) or meta['config_ids']!=config_ids:
        raise ValueError('Predecessor must use exactly the same model configurations')
    return True


def features(point, anchor, predecessor):
    base,_=base_features(point,anchor)
    if predecessor is None:
        prior=np.zeros((24,6));present=np.zeros((24,1))
    else:
        # Reuse the strict finite/nonnegative shape contract before log arithmetic.
        gram_targets(predecessor['point'],predecessor['actual'])
        prior=np.log1p(np.asarray([predecessor['point'][m] for m in MODELS],dtype=float).T)-np.log1p(np.asarray(predecessor['actual'],dtype=float))[:,None]
        present=np.ones((24,1))
    return np.concatenate((base,prior,present),axis=1)


def fit(pairs, masses, anchor):
    q=np.asarray(masses,dtype=float)
    if not pairs or q.shape!=(len(pairs),) or not np.isfinite(q).all() or (q<0).any() or abs(q.sum()-1)>1e-10:
        raise ValueError('Normalized nonnegative case masses required')
    x=np.concatenate([features(p['point'],anchor,p['predecessor']) for p in pairs])
    y=np.concatenate([gram_targets(p['point'],p['actual']) for p in pairs])
    model=ExtraTreesRegressor(**RECIPE)
    model.fit(x,y.reshape(-1,36),sample_weight=np.repeat(q/24,24));trees=[]
    for estimator in model.estimators_:
        t=estimator.tree_;trees.append({k:getattr(t,k).tolist() for k in ('children_left','children_right','feature','threshold','value','n_node_samples','weighted_n_node_samples')})
    evidence={'anchor':anchor,'recipe':model.get_params(),'case_count':len(pairs),
              'row_count':len(x),'feature_count':17,'tree_count':len(trees),'trees':trees,
              'input_sha256':hashlib.sha256(json.dumps({'pairs':pairs,'masses':masses,'anchor':anchor},separators=(',',':')).encode()).hexdigest()}
    return model,evidence


def predict_matrices(model, point, anchor, predecessor):
    matrix=model.predict(features(point,anchor,predecessor)).reshape(24,6,6)
    if not np.isfinite(matrix).all() or not np.allclose(matrix,matrix.transpose(0,2,1),atol=1e-10,rtol=1e-10) or np.linalg.eigvalsh(matrix).min()<-1e-9:
        raise ValueError('Invalid conditional error Gram matrices')
    return matrix
