"""Locked052 validation adapter: six recipes, matched050 mixtures, paired interval."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

from .broad_screen import compute_case, rmsle
from .broad_intraday import summarize
from .context_ensemble import retrieve
from .ensemble_refinement import isolated
from .evidence_ensemble import combine as global_combine, MODELS
from .intraday_ensemble import fit as block_fit, combine as block_combine
from .warm_context import features

ARMS = ('global_cv', 'block_cv', 'block_ledger')
DOMAINS = ('electricity', 'pedestrian')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cv_pairs(source):
    pairs = []
    for i, end in enumerate((658,682,706)):
        folds = [source['folds'][m][i] for m in MODELS]
        if any(f['end'] != end or f['actual'] != folds[0]['actual'] for f in folds):
            raise ValueError('Unmatched CV fold or shifted lead hour')
        pairs.append({'point': {m: source['folds'][m][i]['point'] for m in MODELS}, 'actual': folds[0]['actual']})
    return pairs


def context(row, observed):
    # Deliberately excludes realized scores/actuals from the retrieval interface.
    return {**{k: row[k] for k in ('series_id','origin','last_target')},
        'domain': row['series_id'].split(':')[0], 'outcome_recorded_at': row['last_target'],
        'features': features(row['cv'], observed)}


def fit_case(source, current, contexts, past_rows, global_fit, before_fit=None):
    """Fit using current CV and strictly earlier matured evidence only."""
    cv = cv_pairs(source); retrieved = retrieve(current, contexts)
    if before_fit:before_fit('global_cv')
    anchor = global_fit(cv, [1/3]*3)
    if before_fit:before_fit('block_cv')
    control = block_fit(cv, [1/3]*3, anchor['weights'])
    past = []
    now = datetime.fromisoformat(source['origin'])
    for ref in retrieved['selected']:
        row = past_rows[ref['series_id'], ref['origin']]
        at, close = map(datetime.fromisoformat, (row['origin'], row['last_target']))
        if not (at < now and close <= now and at.timetz() == now.timetz()):
            raise ValueError('Invisible or lead-hour mismatched history')
        if row['series_id'].split(':')[0] != current['domain']:raise ValueError('Wrong domain')
        past.append({'point': row['point'], 'actual': row['actual']})
    if retrieved['ready']:
        if len(past) != 16:raise ValueError('Expected sixteen contexts')
        if before_fit:before_fit('block_ledger')
        ledger = block_fit(cv+past, [1/6]*3+[.5/16]*16, anchor['weights'])
    else:
        ledger = control  # Frozen052 cold-start handling; task stays in denominator.
    points = {'global_cv': global_combine(source['point'], anchor['weights']),
        'block_cv': block_combine(source['point'], control['weights']),
        'block_ledger': block_combine(source['point'], ledger['weights'])}
    return {'fits': {'global_cv': anchor, 'block_cv': control, 'block_ledger': ledger},
        'point': points, 'retrieval': retrieved, 'current_features': current['features'],
        'fallback': None if retrieved['ready'] else 'insufficient_visible_context',
        'weight_fits': 3 if retrieved['ready'] else 2}


def resample_indices(rng, repeats=10000):
    """Each domain independently: eight series, seven circular four-origin blocks."""
    series = rng.integers(0, 8, size=(repeats,2,8))
    starts = rng.integers(0,26,size=(repeats,2,7))
    origins = ((starts[...,None]+np.arange(4)) % 26).reshape(repeats,2,28)[...,:26]
    return series, origins


def uncertainty(rows):
    array = np.empty((2,8,26,3)); identities = {}
    for d, domain in enumerate(DOMAINS):
        ids = sorted({r['series_id'] for r in rows if r['series_id'].startswith(domain+':')})
        if len(ids) != 8:raise ValueError('Eight locked series per domain required')
        identities[domain] = ids
        index = {(r['series_id'],r['round']):r for r in rows if r['series_id'] in ids}
        if len(index) != 208:raise ValueError('All twenty-six origins required')
        for s, name in enumerate(ids):
            for t in range(26):array[d,s,t] = [index[name,t]['scores'][a] for a in ARMS]
    if not np.isfinite(array).all() or (array < 0).any():raise ValueError('Invalid scores')
    si, oi = resample_indices(np.random.default_rng(20260914))
    means = np.zeros((len(si),3))
    for d in range(2):
        sampled = array[d, si[:,d,:,None], oi[:,d,None,:], :]
        means += sampled.mean(axis=(1,2))/2
    if (means[:,:2] <= 0).any():raise ValueError('Relative reduction undefined for zero control')
    reductions = 1-means[:,2,None]/means[:,:2]
    return {'seed':20260914,'replicates':len(si),'series_order':identities,
        'interval_95': {a: np.quantile(reductions[:,i],[.025,.975]).tolist() for i,a in enumerate(ARMS[:2])},
        'method':'Paired domain-stratified series clusters and shared circular four-origin blocks',
        'limitation':'Sixteen series, dependent origins; numerical validation, not agent or final evaluation.'}, si, oi, reductions


def run(source, identity, output):
    source, identity, output = map(Path,(source,identity,output)); here = Path(__file__).parent
    read = lambda p: json.loads(p.read_text())
    lock = read(identity/'manifest.json'); receipt = read(here/'evidence/broad-validation-source-053.json')
    for name, sha in receipt['files'].items():
        if digest(source/name) != sha:raise ValueError('Source053 changed: '+name)
    for name, sha in lock['locked_implementation_sha256'].items():
        if digest(here/name) != sha:raise ValueError('Locked numerical implementation changed: '+name)
    if digest(here/'BROAD_VALIDATION_052.md') != lock['protocol_sha256']:raise ValueError('Protocol changed')
    scored = read(source/'scored-spans.json'); warm = read(source/'warmup-spans.json')
    tasks = read(source/'warmup-boundaries.json'); done = read(source/'COMPLETED.json')
    if len(scored) != 16 or set(scored) != set(warm) or done['usable_warmup_cases'] != 117:raise ValueError('Source cohort mismatch')
    output.mkdir(parents=True,exist_ok=False); (output/'raw').mkdir(); (output/'cases').mkdir()
    def save(name,value):(output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('manifest.json',{'protocol':'BROAD_VALIDATION_052.md','source_receipt_sha256':digest(here/'evidence/broad-validation-source-053.json'),
        'identity_manifest_sha256':digest(identity/'manifest.json'), 'api_calls':0,'reserved_future_values_read':0,
        'code_sha256':{n:digest(here/n) for n in ('validation_run.py','broad_screen.py','broad_intraday.py',*lock['locked_implementation_sha256'])},
        'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn')},
        'planned_scored_cases':416,'planned_warmup_cases':117,'planned_forecast_computations':533*24,
        'planned_estimator_fits':533*12,'availability_assumption':'Nominal period-end source/recording availability; UTC surrogate labels',
        'global_ftol':1e-12, 'private_extra_provider_calls':0})
    save('excluded-warmup.json',[t for t in tasks if not t['ready']])
    numerical = isolated('benchmarks.ledger_optimization._ensemble_validation054',here/'evidence_ensemble.py')
    def refined(*args,**kwargs):
        kwargs['options'] = {**kwargs['options'],'ftol':1e-12}
        return minimize(*args,**kwargs)
    numerical.minimize = refined
    started = time.monotonic(); cpu = time.process_time(); raw = {}; contexts = []; rows = []
    costs = {'forecast_computations':0,'estimator_fits':0,'weight_fits':0,'weight_fits_started':0,'completed_weight_iterations':0}
    pending = None
    def charge(model):
        costs['forecast_computations'] += 1
        costs['estimator_fits'] += model in ('ridge_short','ridge_long','forest')
    def charge_fit(arm):
        costs['weight_fits_started'] += 1
        pending['arm'] = arm
    def status(phase):
        save('status.json',{'phase':phase,'completed_raw_cases':len(raw),'completed_scored_cases':len(rows),
            **costs,'seconds':time.monotonic()-started,'pending':pending})
    try:
        plan = [(t['series_id'],t['round'],t['round']+8,warm[t['series_id']],t['origin']) for t in tasks if t['ready']]
        plan += [(s,r,r,span,None) for s,span in sorted(scored.items()) for r in range(26)]
        for series, round_number, index, span, expected_origin in plan:
            pending = {'series_id':series,'round':round_number}; status('forecasting')
            row = compute_case(span,index,[],charge); row.update(series_id=series,round=round_number)
            if expected_origin and row['origin'].replace('+00:00','') != expected_origin:raise ValueError('Warm origin shifted')
            key = series,row['origin']
            if key in raw:raise ValueError('Duplicate raw case')
            raw[key] = row
            low,high = row['history_indices']; contexts.append(context(row,span['values'][low:high]))
            save(f'raw/{series}-{round_number:03d}.json',row); status('forecasting')
        if len(raw) != 533 or costs['forecast_computations'] != 12792:raise ValueError('Forecast cohort/cost mismatch')
        save('contexts.json',contexts); context_map = {(r['series_id'],r['origin']):r for r in contexts}
        for row in sorted((r for r in raw.values() if r['round'] >= 0),key=lambda r:(r['origin'],r['series_id'])):
            key = row['series_id'],row['origin']; pending = {'series_id':key[0],'round':row['round']}; status('fitting_mixtures')
            # Do not pass current production actuals/scores into the decision step.
            proposal = fit_case({k:v for k,v in row.items() if k not in ('actual','scores')},context_map[key],contexts,
                {k:v for k,v in raw.items() if datetime.fromisoformat(v['origin']) < datetime.fromisoformat(row['origin'])
                    and datetime.fromisoformat(v['last_target']) <= datetime.fromisoformat(row['origin'])},numerical.fit,charge_fit)
            costs['weight_fits'] += proposal['weight_fits']
            fitted_arms = ARMS if proposal['fallback'] is None else ARMS[:2]
            costs['completed_weight_iterations'] += sum(proposal['fits'][a]['iterations'] for a in fitted_arms)
            result = {**{k:row[k] for k in ('series_id','round','origin','last_target')},**proposal,
                'actual':row['actual'],'scores':{a:rmsle(p,row['actual']) for a,p in proposal['point'].items()}}
            save(f'cases/{key[0]}-{row["round"]:02d}.json',result)
            rows.append({k:result[k] for k in ('series_id','round','scores','fallback')}); status('fitting_mixtures')
        if len(rows) != 416:raise ValueError('All scored tasks required')
        interval, series_indices, origin_indices, reductions = uncertainty(rows)
        np.savez_compressed(output/'bootstrap.npz',series_indices=series_indices,origin_indices=origin_indices,reductions=reductions)
        overall = summarize(rows); domains = {d:summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in DOMAINS}
        passed = (overall['primary_reduction'] >= .2 and overall['reduction_vs_global_guard'] >= .2
            and all(x[0] > 0 for x in interval['interval_95'].values())
            and all(v['primary_reduction'] > 0 and v['reduction_vs_global_guard'] > 0 for v in domains.values()))
        report = {'overall':overall,'domains':domains,'uncertainty':interval,'validation_gate_passed':passed,
            'cold_start_fallbacks':sum(r['fallback'] is not None for r in rows),'rows':rows,**costs,
            'usable_warmup_cases':117,'unavailable_warmup_cases':11,'api_calls':0,
            'seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,
            'goal_achieved':False,'limitation':'Disjoint-series numerical development validation; no agent evaluation or final-reserved access.'}
        save('report.json',report); status('complete'); return report
    except BaseException as error:
        save('FAILED.json',{'task':pending,'error':type(error).__name__,'message':str(error),
            'certificate':getattr(error,'certificate',None),**costs,'weight_fit_cost_note':'Only completed fit groups counted; failure certificate preserves a failed fit.',
            'completed_raw_cases':len(raw),'completed_scored_cases':len(rows),'seconds':time.monotonic()-started})
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source','identity','output'):p.add_argument(n)
    report = run(**vars(p.parse_args()));print(json.dumps({k:v for k,v in report.items() if k != 'rows'},indent=2))
