"""Shared convex forecast combination; training pairs must already be visible."""
import hashlib
import json
import numpy as np
from scipy.optimize import minimize

MODELS = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def arrays(pair):
    p = np.asarray([pair['point'][m] for m in MODELS], dtype=float).T
    y = np.asarray(pair['actual'], dtype=float)
    if p.shape != (24, 6) or y.shape != (24,) or not np.isfinite(p).all() or not np.isfinite(y).all() or (p < 0).any() or (y < 0).any():
        raise ValueError('Complete nonnegative 24-step matched pairs required')
    return np.log1p(p), np.log1p(y)


def objective(weights, prepared, masses):
    loss = 1e-6*np.square(weights-1/6).sum()
    gradient = 2e-6*(weights-1/6)
    for (point, actual), mass in zip(prepared, masses, strict=True):
        error = point@weights-actual
        rmsle = float(np.sqrt(np.mean(error**2)))
        loss += mass*rmsle
        if rmsle > 0:
            gradient = gradient+mass*(point.T@error)/(24*rmsle)
    return float(loss), gradient


def fit(pairs, masses):
    if len(pairs) != len(masses) or not pairs or any(m < 0 for m in masses) or not np.isclose(sum(masses), 1.):
        raise ValueError('Normalized positive case masses required')
    prepared = [arrays(pair) for pair in pairs]
    initial = np.full(6, 1/6)
    result = minimize(lambda w: objective(w, prepared, masses), initial, method='SLSQP', jac=True,
        bounds=[(0., 1.)]*6, constraints={'type': 'eq', 'fun': lambda w: w.sum()-1,
                                       'jac': lambda w: np.ones(6)},
        options={'maxiter': 500, 'ftol': 1e-10})
    w = result.x
    if not np.isfinite(w).all() or w.min() < -1e-8 or abs(w.sum()-1) > 1e-8:
        raise ValueError('Infeasible solver result')
    w = np.maximum(w, 0.); w /= w.sum()
    value, _ = objective(w, prepared, masses)
    gradient = 2e-6*(w-1/6); omitted = 0.
    # A near-zero case norm has a global lower bound of zero. Charge its full
    # current loss rather than divide by a numerically tiny residual norm.
    for (point, actual), mass in zip(prepared, masses, strict=True):
        error = point@w-actual; norm = float(np.sqrt(np.mean(error**2)))
        if norm <= 1e-8:
            omitted += mass*norm
        else:
            gradient += mass*(point.T@error)/(24*norm)
    gap = float(w@gradient-gradient.min()+omitted)
    certificate = {'weights': w.tolist(), 'objective': value, 'initial_objective': objective(initial, prepared, masses)[0],
        'convex_gap_bound': gap, 'near_zero_loss_bound': omitted,
        'success': bool(result.success), 'solver_status': int(result.status),
        'solver_message': str(result.message), 'iterations': int(result.nit),
        'input_sha256': hashlib.sha256(json.dumps({'pairs': pairs, 'masses': masses}, separators=(',', ':')).encode()).hexdigest()}
    if not result.success or gap > 1e-5 or value > certificate['initial_objective']+1e-8:
        error = ValueError('Optimizer failed frozen convergence requirements')
        error.certificate = certificate
        raise error
    return certificate


def combine(point, weights):
    p = np.asarray([point[m] for m in MODELS], dtype=float).T
    w = np.asarray(weights, dtype=float)
    if p.shape != (24, 6) or not np.isfinite(p).all() or (p < 0).any() or w.shape != (6,) or not np.isfinite(w).all() or (w < 0).any() or abs(w.sum()-1) > 1e-8:
        raise ValueError('Invalid combination request')
    return np.expm1(np.log1p(p)@w).tolist()
