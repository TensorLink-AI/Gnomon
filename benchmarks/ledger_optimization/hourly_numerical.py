"""Prospectively fixed nominal-hour recipes for development screen 038."""
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RECIPES = {
    'daily': {'kind': 'seasonal', 'season': 24},
    'weekly': {'kind': 'seasonal', 'season': 168},
    'weekly_mean': {'kind': 'weekly_mean'},
    'ridge_short': {'kind': 'ridge', 'window': 336, 'lags': 48, 'alpha': 10.0},
    'ridge_long': {'kind': 'ridge', 'window': 730, 'lags': 168, 'alpha': 10.0},
    'forest': {'kind': 'forest', 'window': 730, 'lags': 48, 'depth': 6},
}


def calendar_features(labels):
    return np.asarray([[math.sin(2*math.pi*t.hour/24), math.cos(2*math.pi*t.hour/24),
                        math.sin(2*math.pi*t.weekday()/7), math.cos(2*math.pi*t.weekday()/7)]
                       for t in labels])


def predict(history, history_labels, future_labels, name):
    config = RECIPES[name]
    history = np.asarray(history, dtype=float)
    if len(history) != len(history_labels) or not np.isfinite(history).all() or (history < 0).any():
        raise ValueError('Invalid aligned observed history')
    horizon = len(future_labels)
    if horizon != 24 or len(history) < 658:
        raise ValueError('Expected 24 targets and at least 658 observations')
    if config['kind'] == 'seasonal':
        season = config['season']
        return [float(history[-season+i % season]) for i in range(horizon)]
    if config['kind'] == 'weekly_mean':
        return [float(np.mean([history[-168*k+i] for k in range(1, 4)])) for i in range(horizon)]
    history = history[-config['window']:]
    labels = history_labels[-len(history):]
    cov = calendar_features(labels)
    future = calendar_features(future_labels)
    z = np.log1p(history)
    lag = config['lags']

    def features(prior, c):
        return np.r_[prior[-lag:][::-1], np.mean(prior[-24:]), np.mean(prior[-168:]), c]

    x = np.asarray([features(z[:i], cov[i]) for i in range(lag, len(z))])
    y = z[lag:]
    if config['kind'] == 'ridge':
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=config['alpha']))
    else:
        estimator = RandomForestRegressor(n_estimators=80, max_depth=config['depth'],
                                         min_samples_leaf=3, random_state=17, n_jobs=1)
    estimator.fit(x, y)
    prior = z.tolist()
    points = []
    for c in future:
        value = float(estimator.predict(features(prior, c).reshape(1, -1))[0])
        value = max(0., min(value, math.log1p(max(history.max()*100, 1000))))
        prior.append(value)
        points.append(math.expm1(value))
    return points
