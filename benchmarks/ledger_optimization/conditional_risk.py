"""Conditional historical error Gram matrices with auditable record weights."""
from datetime import datetime
import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')
SETTINGS = dict(n_estimators=64, max_depth=4, min_samples_leaf=8, max_features=1.,
                bootstrap=False, random_state=17, n_jobs=1, criterion='squared_error')


def visible(episodes, domain, origin):
    now = datetime.fromisoformat(origin); rows = []; seen = set()
    if now.tzinfo is None:raise ValueError('Explicit current cutoff required')
    for row in episodes:
        if row['domain'] != domain:continue
        at, close, recorded = (datetime.fromisoformat(row[k]) for k in ('origin', 'last_target', 'outcome_recorded_at'))
        if any(t.tzinfo is None for t in (at, close, recorded)):raise ValueError('Explicit episode clocks required')
        if at >= now or close > now or recorded > now:continue
        key = row['series_id'], at
        if key in seen or close <= at or recorded < close:raise ValueError('Inconsistent or duplicate episode')
        seen.add(key); rows.append(row)
    return sorted(rows, key=lambda row: (row['origin'], row['series_id']))


def gram(pair):
    points = np.asarray([pair['point'][m] for m in MODELS], dtype=float).T
    actual = np.asarray(pair['actual'], dtype=float)
    if points.shape != (24, 6) or actual.shape != (24,) or not np.isfinite(points).all() or not np.isfinite(actual).all() or (points < 0).any() or (actual < 0).any():
        raise ValueError('Finite nonnegative matched forecasts required')
    errors = np.log1p(points)-np.log1p(actual)[:, None]
    return errors.T@errors/24


def train(features, matrices):
    x = np.asarray(features, dtype=float); targets = np.asarray(matrices, dtype=float)
    if x.ndim != 2 or x.shape[1] != 12 or targets.shape != (len(x), 6, 6) or len(x) < 32 or not np.isfinite(x).all() or not np.isfinite(targets).all():
        raise ValueError('Complete visible features and matrices required')
    model = ExtraTreesRegressor(**SETTINGS).fit(x, targets.reshape(len(x), 36))
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append({'children_left': tree.children_left.tolist(), 'children_right': tree.children_right.tolist(),
                      'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist()})
    return model, {'settings': SETTINGS, 'features': x.tolist(), 'matrices': targets.tolist(), 'trees': trees,
        'input_sha256': hashlib.sha256(json.dumps({'features': x.tolist(), 'matrices': targets.tolist()}, separators=(',', ':')).encode()).hexdigest()}


def predict(model, training, features):
    x = np.asarray(training['features'], dtype=float); current = np.asarray(features, dtype=float)
    if current.shape != (12,) or not np.isfinite(current).all():raise ValueError('Twelve finite query features required')
    weights = np.zeros(len(x)); leaves = []
    for tree in model.estimators_:
        leaf = int(tree.apply(current.reshape(1, -1))[0]); matching = tree.apply(x) == leaf
        if not matching.any():raise ValueError('Query leaf has no training support')
        weights[matching] += 1/(len(model.estimators_)*int(matching.sum())); leaves.append(leaf)
    matrices = np.asarray(training['matrices'])
    matrix = np.einsum('n,nij->ij', weights, matrices)
    predicted = model.predict(current.reshape(1, -1))[0].reshape(6, 6)
    if not np.allclose(matrix, predicted, atol=1e-10, rtol=1e-9):raise ValueError('Historical record weights do not reproduce model')
    return {'matrix': matrix.tolist(), 'record_weights': weights.tolist(), 'leaves': leaves,
            'current_features': current.tolist()}


def fit_weights(matrix, anchor):
    g = np.asarray(matrix, dtype=float); a = np.asarray(anchor, dtype=float)
    if g.shape != (6, 6) or not np.isfinite(g).all() or not np.allclose(g, g.T, atol=1e-10, rtol=1e-10):raise ValueError('Symmetric finite Gram required')
    eigenvalues = np.linalg.eigvalsh(g)
    if eigenvalues[0] < -1e-9:raise ValueError('Gram matrix is not PSD')
    if a.shape != (6,) or not np.isfinite(a).all() or (a < 0).any() or abs(a.sum()-1) > 1e-8:raise ValueError('Simplex anchor required')
    def objective(w):return float(w@g@w+.001*np.square(w-a).sum()), 2*g@w+.002*(w-a)
    result = minimize(objective, a, jac=True, method='SLSQP', bounds=[(0., 1.)]*6,
        constraints={'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(6)},
        options={'ftol': 1e-12, 'maxiter': 500})
    w = result.x
    if not np.isfinite(w).all() or w.min() < -1e-8 or abs(w.sum()-1) > 1e-8:raise ValueError('Infeasible quadratic weights')
    w = np.maximum(w, 0.); w /= w.sum(); value, gradient = objective(w)
    gap = float(w@gradient-gradient.min())
    info = {'weights': w.tolist(), 'anchor': a.tolist(), 'matrix': g.tolist(), 'minimum_eigenvalue': float(eigenvalues[0]),
        'objective': value, 'initial_objective': objective(a)[0], 'convex_gap_bound': gap,
        'success': bool(result.success), 'solver_status': int(result.status), 'solver_message': str(result.message), 'iterations': int(result.nit)}
    if not result.success or gap > 1e-5 or value > info['initial_objective']+1e-8:
        error = ValueError('Quadratic fit failed frozen acceptance'); error.certificate = info; raise error
    return info
