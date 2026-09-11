"""Exploratory log-bias calibration beyond the fixed candidate-selection pool.

Not counted toward the registered selection objective. Every automatic rule uses
only finalized past outcomes. Both control and ledger get the same transform;
control estimates it from the latest two archived CV-equivalent origins.
Initial two origins use raw predictions for all deployable policies because the
pre-evaluation CV vectors are not in this archive. Hindsight bounds are explicitly
non-deployable. Original forecasts are never rewritten.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import mean

import numpy as np


GRID = np.linspace(-2, 2, 81)


def losses(point, actual, shifts=GRID):
    predicted = np.log1p(np.maximum(point, 0))
    truth = np.log1p(actual)
    return np.sqrt(np.mean((np.maximum(predicted[None, :]+shifts[:, None], 0)-truth[None, :])**2, axis=1))


def fit(history, provider):
    values = np.mean([losses(c['predictions'][provider], c['actual']) for c in history], axis=0)
    best = min(range(len(GRID)), key=lambda i:(values[i], abs(GRID[i])))
    return float(GRID[best])


def run(cases):
    rows, prior = [], []
    for c in sorted(cases, key=lambda c:(c['origin'],c['series_id'])):
        now = datetime.fromisoformat(c['origin'])
        visible = [h for h in prior if h['series_id']==c['series_id']
                   and datetime.fromisoformat(h['outcome_recorded_at'])<=now
                   and max(datetime.fromisoformat(t) for t in h['future_timestamps'])<=now]
        provider = min(c['current_card'], key=lambda p:c['current_card'][p]['cv_rmsle'])
        cv_bias = fit(visible[-2:], provider) if len(visible)>=2 else 0
        policies = {'raw_cv_choice': 0, 'cv_calibrated_control': cv_bias}
        for window in (4,12,0):
            for weight in (0,.5):
                history = visible[-window:] if window else visible
                history_bias = fit(history, provider) if len(visible)>=4 else cv_bias
                policies[f'ledger_window{window}_cv{weight}'] = (1-weight)*history_bias+weight*cv_bias
        scores = {name: float(losses(c['predictions'][provider],c['actual'],np.array([b]))[0]) for name,b in policies.items()}
        scores['hindsight_calibrated_oracle'] = min(float(losses(point,c['actual']).min()) for point in c['predictions'].values())
        rows.append({'series_id':c['series_id'],'round':c['round'],'provider':provider,
                     'matured_origins':len(visible),'log_shifts':policies,'rmsle':scores})
        prior.append(c)
    aggregates = {name:mean(r['rmsle'][name] for r in rows) for name in rows[0]['rmsle']}
    return {'scope':'exploratory expanded capability; NOT the registered fixed-portfolio agent objective',
        'transform':'max(exp(log1p(prediction)+bias)-1,0)', 'bias_bounds':[-2,2], 'grid_step':.05,
        'control':'Same CV-selected provider with bias estimated from latest two archived mature origins',
        'cold_start':'First two archived origins remain raw in both deployable arms',
        'hindsight_bound':'Uses future actuals; not deployable and never a treatment score',
        'api_calls':0, 'original_forecasts_modified':False, 'mean_case_rmsle':aggregates,'cases':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(json.loads(args.cases.read_text()))
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    main()
