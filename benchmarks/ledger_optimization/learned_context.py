"""Outcome-supervised historical relevance; query features never contain outcomes."""
import hashlib
import json
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .conditional_risk import MODELS, SETTINGS, visible


def contrast(pair):
    """Centered production-minus-CV RMSLE, from one matured historical case."""
    actual = np.asarray(pair['actual'], dtype=float)
    points = np.asarray([pair['point'][m] for m in MODELS], dtype=float)
    cv = np.asarray([pair['cv'][m] for m in MODELS], dtype=float)
    if actual.shape != (24,) or points.shape != (6, 24) or cv.shape != (6,) or any(
        not np.isfinite(x).all() or (x < 0).any() for x in (actual, points, cv)
    ):raise ValueError('Complete finite nonnegative historical evidence required')
    residual = np.sqrt(np.mean((np.log1p(points)-np.log1p(actual))**2, axis=1))-cv
    return (residual-residual.mean()).tolist()


def train_at(contexts, raw, domain, origin):
    eligible = visible(contexts, domain, origin)
    ready = len(eligible) >= 32 and len({r['origin'] for r in eligible}) >= 3
    record = {'domain': domain, 'origin': origin, 'effective_source_as_of': origin,
        'effective_recorded_as_of': origin, 'ready': ready,
        'eligible': [{'series_id': r['series_id'], 'origin': r['origin']} for r in eligible]}
    if not ready:return None, record
    x = np.asarray([r['features'] for r in eligible], dtype=float)
    if x.shape != (len(eligible), 12) or not np.isfinite(x).all():raise ValueError('Twelve finite predecision features required')
    # Only look up labels AFTER visibility filtering, including recording maturity.
    y = np.asarray([contrast(raw[r['series_id'], r['origin']]) for r in eligible])
    model = ExtraTreesRegressor(**SETTINGS).fit(x, y)
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append({k: getattr(tree, k).tolist() for k in ('children_left', 'children_right', 'feature', 'threshold')})
    training = {'features': x.tolist(), 'targets': y.tolist()}
    record.update(training, settings=SETTINGS, trees=trees,
        input_sha256=hashlib.sha256(json.dumps(training, separators=(',', ':')).encode()).hexdigest())
    return model, record


def retrieve(model, training, features):
    current = np.asarray(features, dtype=float)
    if current.shape != (12,) or not np.isfinite(current).all():raise ValueError('Twelve finite query features required')
    if model is None:return {'ready': False, 'current_features': current.tolist(), 'record_weights': [], 'leaves': []}
    x = np.asarray(training['features'], dtype=float); weights = np.zeros(len(x)); leaves = []
    for tree in model.estimators_:
        leaf = int(tree.apply(current.reshape(1, -1))[0]); matched = tree.apply(x) == leaf
        if not matched.any():raise ValueError('Unsupported query leaf')
        weights[matched] += 1/(len(model.estimators_)*int(matched.sum())); leaves.append(leaf)
    expected = weights @ np.asarray(training['targets'])
    if not np.allclose(expected, model.predict(current.reshape(1, -1))[0], atol=1e-10, rtol=1e-9):raise ValueError('Record weights do not reproduce forest prediction')
    return {'ready': True, 'current_features': current.tolist(), 'record_weights': weights.tolist(),
        'leaves': leaves, 'predicted_contrast': expected.tolist(),
        'effective_records': float(1/np.square(weights).sum())}
