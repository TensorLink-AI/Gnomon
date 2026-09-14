"""Four shared six-hour mixtures over six originals and the search-selected slot."""
import hashlib
import json
import numpy as np
from scipy.optimize import brentq, minimize

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest', 'search_selected')
PENALTY = .01
SMOOTHING = 1e-6


def prepare(pairs):
    points = np.asarray([[pair['point'][m] for m in MODELS] for pair in pairs], dtype=float).transpose(0, 2, 1)
    actual = np.asarray([pair['actual'] for pair in pairs], dtype=float)
    if points.shape != (len(pairs), 24, 7) or actual.shape != (len(pairs), 24) or not np.isfinite(points).all() or not np.isfinite(actual).all() or (points < 0).any() or (actual < 0).any():
        raise ValueError('Complete finite nonnegative 24-step matched pairs required')
    return np.log1p(points).reshape(-1, 4, 6, 7), np.log1p(actual)


def objective(flat, prepared, masses, anchor, certificate=False):
    w = flat.reshape(4, 7); points, actual = prepared
    error = np.einsum('nbhm,bm->nbh', points, w).reshape(-1, 24)-actual
    # Smooth the norm at exact fits. Its excess over RMSLE is in [0, 1e-6].
    norm = np.sqrt(np.mean(error**2, axis=1)+SMOOTHING**2)
    loss = masses@norm+PENALTY/4*np.square(w-anchor).sum()
    scales = masses/(24*norm)
    gradient = np.einsum('n,nbhm,nbh->bm', scales, points, error.reshape(-1, 4, 6))
    gradient += 2*PENALTY/4*(w-anchor)
    if certificate:return float(loss), gradient, 0.
    return float(loss), gradient.ravel()


def fit(pairs, masses, anchor):
    q = np.asarray(masses, dtype=float); a = np.asarray(anchor, dtype=float)
    if not pairs or q.shape != (len(pairs),) or not np.isfinite(q).all() or (q < 0).any() or abs(q.sum()-1) > 1e-10:
        raise ValueError('Normalized nonnegative case masses required')
    if a.shape != (7,) or not np.isfinite(a).all() or (a < 0).any() or abs(a.sum()-1) > 1e-8:
        raise ValueError('Simplex anchor required')
    prepared = prepare(pairs); initial = np.tile(a, (4, 1)).ravel()
    constraints = np.kron(np.eye(4), np.ones((1, 7)))
    result = minimize(lambda w: objective(w, prepared, q, a), initial, method='SLSQP', jac=True,
        bounds=[(0., 1.)]*28, constraints={'type': 'eq', 'fun': lambda w: constraints@w-1,
            'jac': lambda w: constraints}, options={'ftol': 1e-14, 'maxiter': 500})
    weights = result.x.reshape(4, 7)
    if not np.isfinite(weights).all() or weights.min() < -1e-8 or np.max(np.abs(weights.sum(axis=1)-1)) > 1e-8:
        raise ValueError('Infeasible intraday weights')
    weights = np.maximum(weights, 0.); weights /= weights.sum(axis=1, keepdims=True)
    refinements = []; refinement_start = weights.tolist()
    # Tiny objective changes can stop SLSQP before the first-order certificate
    # passes near an exact fit. Refine on the same convex objective and simplex.
    for _ in range(32):
        _, g, _ = objective(weights.ravel(), prepared, q, a, certificate=True)
        before = float(np.sum(weights*g)-g.min(axis=1).sum())
        if before <= 1e-5:break
        vertex = np.eye(7)[g.argmin(axis=1)]; direction = vertex-weights
        def derivative(step):
            _, along = objective((weights+step*direction).ravel(), prepared, q, a)
            return float(along@direction.ravel())
        step = 1. if derivative(1.) <= 0 else brentq(derivative, 0., 1., xtol=1e-15, rtol=1e-14)
        weights = weights+step*direction
        refinements.append({'gap_before': before, 'step': float(step)})
    value, gradient, omitted = objective(weights.ravel(), prepared, q, a, certificate=True)
    gap = float(np.sum(weights*gradient)-gradient.min(axis=1).sum()+omitted)
    info = {'weights': weights.tolist(), 'anchor': a.tolist(), 'objective': value,
        'initial_objective': objective(initial, prepared, q, a)[0], 'convex_gap_bound': gap,
        'near_zero_loss_bound': omitted, 'success': bool(result.success), 'solver_status': int(result.status),
        'smoothing_epsilon': SMOOTHING, 'maximum_objective_smoothing_error': SMOOTHING,
        'solver_message': str(result.message), 'iterations': int(result.nit),
        'certificate_refinements': refinements,
        'refinement_start_weights': refinement_start,
        'input_sha256': hashlib.sha256(json.dumps({'pairs': pairs, 'masses': masses, 'anchor': anchor}, separators=(',', ':')).encode()).hexdigest()}
    if not result.success or gap > 1e-5 or value > info['initial_objective']+1e-8:
        error = ValueError('Intraday optimizer failed frozen acceptance requirements'); error.certificate = info
        raise error
    return info


def combine(points, weights):
    p = np.asarray([points[m] for m in MODELS], dtype=float).T
    w = np.asarray(weights, dtype=float)
    if p.shape != (24, 7) or w.shape != (4, 7) or not np.isfinite(p).all() or not np.isfinite(w).all() or (p < 0).any() or (w < 0).any() or np.max(np.abs(w.sum(axis=1)-1)) > 1e-8:
        raise ValueError('Valid forecasts and four simplex weight vectors required')
    return np.expm1(np.sum(np.log1p(p)*np.repeat(w, 6, axis=0), axis=1)).tolist()
