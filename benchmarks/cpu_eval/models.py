"""Nine CPU configurations, explicit families; no hidden AutoML search or API calls."""
import numpy as np

FAMILIES = {'last_value':'naive', 'seasonal_naive':'seasonal naive', 'drift':'linear drift',
            'auto_ets':'exponential smoothing', 'theta':'Theta', 'croston_sba':'intermittent demand',
            'ridge':'linear autoregression', 'random_forest':'bagged trees',
            'hist_gradient_boosting':'boosted trees'}


def forecast(name, history, horizon, season, seed):
    y = np.asarray(history, dtype=float)
    if name == 'last_value':
        return np.repeat(y[-1], horizon)
    if name == 'seasonal_naive':
        return np.resize(y[-min(season,len(y)):], horizon)
    if name == 'drift':
        return y[-1] + np.arange(1,horizon+1)*(y[-1]-y[0])/max(1,len(y)-1)
    if name in ('auto_ets','theta','croston_sba'):
        from statsforecast.models import AutoETS, Theta, CrostonSBA
        if name == 'croston_sba' and np.any(y < 0):
            raise ValueError('Croston requires nonnegative history; declared fallback applies')
        model = {'auto_ets':lambda: AutoETS(season_length=season, model='ZZZ'),
                 'theta':lambda: Theta(season_length=season), 'croston_sba':CrostonSBA}[name]()
        return model.forecast(y=y, h=horizon)['mean']
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lag = min(48, max(8,season), len(y)//3)
    windows = np.lib.stride_tricks.sliding_window_view(y, lag+1)
    model = {'ridge':lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
             'random_forest':lambda: RandomForestRegressor(n_estimators=32, max_depth=8, min_samples_leaf=2,
                                                          random_state=seed, n_jobs=1),
             'hist_gradient_boosting':lambda: HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15,
                                                                           random_state=seed)}[name]()
    model.fit(windows[:,:-1], windows[:,-1])
    state, result = list(y[-lag:]), []
    for _ in range(horizon):
        point = float(model.predict(np.array(state[-lag:])[None,:])[0])
        result.append(point)
        state.append(point)
    return np.array(result)
