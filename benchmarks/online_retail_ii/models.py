"""Identical numerical candidates for direct and Gnomon execution paths."""
from datetime import timedelta
import math
import warnings

import numpy as np

CANDIDATES = ('last_value', 'seasonal_naive_7', 'mean_28', 'weekday_mean_4',
              'auto_ets_7', 'auto_arima_7', 'auto_theta_7', 'croston_sba',
              'ridge_log', 'hist_gradient_boosting_log')


def ml_features(history, day):
    values = np.asarray(history, dtype=float)
    return [*[values[-k] for k in (1, 7, 14, 28)],
            *[float(np.mean(values[-k:])) for k in (7, 28, 56)],
            math.sin(2*math.pi*day.weekday()/7), math.cos(2*math.pi*day.weekday()/7),
            math.sin(2*math.pi*day.timetuple().tm_yday/365.25),
            math.cos(2*math.pi*day.timetuple().tm_yday/365.25)]


def predict(name, history, horizon, last_date):
    y = np.asarray(history, dtype=float)
    if name not in CANDIDATES or len(y) < 84 or horizon < 1 or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError('Unknown model or invalid/nonnegative history contract')
    if name == 'last_value':
        point = np.repeat(y[-1], horizon)
    elif name == 'seasonal_naive_7':
        point = np.resize(y[-7:], horizon)
    elif name == 'mean_28':
        point = np.repeat(np.mean(y[-28:]), horizon)
    elif name == 'weekday_mean_4':
        point = np.resize(y[-28:].reshape(4, 7).mean(axis=0), horizon)
    elif name in ('ridge_log', 'hist_gradient_boosting_log'):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        z = np.log1p(y)
        start = last_date-timedelta(days=len(y)-1)
        X = np.asarray([ml_features(z[:i], start+timedelta(days=i)) for i in range(56, len(y))])
        model = make_pipeline(StandardScaler(), Ridge(alpha=10)) if name == 'ridge_log' else HistGradientBoostingRegressor(
            max_iter=150, max_leaf_nodes=15, learning_rate=.1, min_samples_leaf=20, random_state=7)
        model.fit(X, z[56:]); extended = list(z); forecasts = []
        for k in range(1, horizon+1):
            value = float(model.predict([ml_features(extended, last_date+timedelta(days=k))])[0])
            extended.append(value); forecasts.append(value)
        point = np.expm1(forecasts)
    else:
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, CrostonSBA
        model = {'auto_ets_7': lambda: AutoETS(season_length=7),
                 'auto_arima_7': lambda: AutoARIMA(season_length=7),
                 'auto_theta_7': lambda: AutoTheta(season_length=7),
                 'croston_sba': CrostonSBA}[name]()
        point = model.forecast(y=y, h=horizon)['mean']
    point = np.asarray(point, dtype=float)
    if point.shape != (horizon,) or not np.all(np.isfinite(point)):
        raise ValueError('Numerical provider returned invalid forecast')
    return point


def forecast(name, history, horizon, last_date):
    """Keep attempted failures and warnings; apply a common disclosed fallback."""
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            point = predict(name, history, horizon, last_date)
        except Exception as exc:
            error = {'category': type(exc).__name__, 'message': str(exc)[:300]}
            point = predict('seasonal_naive_7', history, horizon, last_date)
    clipped = int(np.sum(point < 0)); point = np.maximum(point, 0)
    return {'provider': name, 'point': point.tolist(), 'fallback_used': error is not None,
            'executed_provider': 'seasonal_naive_7' if error else name, 'error': error,
            'clipped_points': clipped, 'warnings': sorted({type(w.message).__name__ for w in caught})}


def metrics(point, actual, history):
    p, a, y = (np.asarray(v, dtype=float) for v in (point, actual, history))
    if p.shape != a.shape or not len(a) or not all(np.all(np.isfinite(v)) for v in (p, a, y)) or any(np.any(v < 0) for v in (p, a, y)):
        raise ValueError('Invalid scoring pairs')
    error = p-a; scale = float(np.mean(np.abs(y[7:]-y[:-7]))) if len(y) > 7 else 0
    mae = float(np.mean(np.abs(error)))
    return {'rmsle': float(np.sqrt(np.mean((np.log1p(p)-np.log1p(a))**2))),
            'mae': mae, 'mase': mae/scale if scale > 0 else None, 'bias': float(np.mean(error)),
            'absolute_error_sum': float(np.abs(error).sum()), 'actual_sum': float(a.sum()),
            'points': len(a), 'mase_scale': scale}
