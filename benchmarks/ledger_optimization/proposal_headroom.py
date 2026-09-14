"""Hindsight-only ceilings for fixed saved proposals; never an executable policy."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

MODELS=('daily','weekly','weekly_mean','ridge_short','ridge_long','forest')
ARMS=('global_cv','block_cv','legacy_ledger','recent_ledger','lifetime_ledger','learned_ledger')
ORACLES=('provider_oracle','proposal_oracle','union_oracle','stepwise_envelope')
BASELINES=('selected_cv','block_cv')


def score(point,actual):
    if len(point)!=len(actual) or not actual or any(not math.isfinite(x) or x<0 for x in list(point)+list(actual)):
        raise ValueError('Equal nonempty finite nonnegative forecasts/actuals required')
    return math.sqrt(math.fsum((math.log1p(p)-math.log1p(y))**2 for p,y in zip(point,actual,strict=True))/len(actual))


def best(scores):
    if not scores or any(not math.isfinite(v) or v<0 for v in scores.values()):raise ValueError('Valid complete score map required')
    value=min(scores.values())
    return {'score':value,'selected':next(k for k,v in scores.items() if v==value),'ties':[k for k,v in scores.items() if v==value]}


def envelope(points,actual):
    for p in points.values():score(p,actual)
    if not points:raise ValueError('Model forecasts required')
    residual=[]
    for h,y in enumerate(actual):
        low=min(math.log1p(p[h]) for p in points.values());high=max(math.log1p(p[h]) for p in points.values());observed=math.log1p(y)
        residual.append(min(high,max(low,observed))-observed)
    return {'log_residual':residual,'score':math.sqrt(math.fsum(e*e for e in residual)/len(actual)),
        'scope':'independent_horizon_weights_outside_shared_block_action'}


def summary(rows):
    means={a:math.fsum(r['scores'][a] for r in rows)/len(rows) for a in (*BASELINES,*[a for a in ARMS if a!='block_cv'],*ORACLES)}
    comparisons={}
    for baseline in BASELINES:
        comparisons[baseline]={}
        for oracle in ORACLES:
            gain=None if means[baseline]==0 else 1-means[oracle]/means[baseline]
            comparisons[baseline][oracle]={'maximum_relative_improvement':gain,
                'twenty_percent_status':'impossible_in_stated_class' if means[oracle]>.8*means[baseline] else 'not_ruled_out_not_established',
                'fraction_of_oracle_gain_required':.2/gain if gain is not None and gain>0 else None}
    return {'cases':len(rows),'mean_rmsle':means,'comparisons':comparisons}


def run(original,proposals,output):
    original,proposals,output=map(Path,(original,proposals,output));here=Path(__file__).parent;receipts={}
    def load(root,name):
        p=here/'evidence'/name;receipts[name]=hashlib.sha256(p.read_bytes()).hexdigest();rows={}
        for n,h in json.loads(p.read_text())['files'].items():
            if not n.startswith(('electricity:','pedestrian:')):continue
            raw=(root/n).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source '+n)
            r=json.loads(raw);key=r['series_id'],r['origin']
            if key in rows:raise ValueError('Duplicate task')
            rows[key]=r
        if len(rows)!=416:raise ValueError('All416 cases required')
        return rows
    source=load(original,'broad-screen-038.json');saved=load(proposals,'broad-learned-retrieval-062.json')
    if set(source)!=set(saved):raise ValueError('Mismatched task identity')
    output.mkdir(parents=True,exist_ok=False)
    def write(n,v):(output/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    write('manifest.json',{'protocol':'PROPOSAL_HEADROOM_063.md','code_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('PROPOSAL_HEADROOM_063.md','proposal_headroom.py')},
        'source_receipts_sha256':receipts,'diagnostic_only':True,'uses_current_future_actuals':True,'validation_or_final_access':False,
        'forecasts_or_policies_changed':False,'api_calls':0,'provider_calls':0,'additional_weight_fits':0,'inherited_forecast_computations':25584})
    rows=[];start=time.monotonic();cpu=time.process_time()
    for key,r in sorted(source.items(),key=lambda p:(p[0][1],p[0][0])):
        prior=saved[key]
        if r['actual']!=prior['actual'] or r['round']!=prior['round'] or r['last_target']!=prior['last_target']:raise ValueError('Changed scoring task')
        models={m:score(r['point'][m],r['actual']) for m in MODELS};arms={a:score(prior['point'][a],r['actual']) for a in ARMS}
        selected=min(MODELS,key=lambda m:r['cv'][m])
        if r['selection']['cv']!=selected:raise ValueError('Changed CV selection')
        choices={'provider_oracle':best(models),'proposal_oracle':best(arms),'union_oracle':best({**{'provider:'+m:v for m,v in models.items()},**{'proposal:'+a:v for a,v in arms.items()}})}
        bounded=envelope(r['point'],r['actual'])
        scores={'selected_cv':models[selected],**arms,**{k:v['score'] for k,v in choices.items()},'stepwise_envelope':bounded['score']}
        row={**{k:r[k] for k in ('series_id','origin','round','last_target')},'scores':scores,'provider_scores':models,'selected_cv_provider':selected,
            'hindsight':choices,'stepwise_envelope':bounded,'diagnostic_only':True,'uses_current_future_actuals':True}
        rows.append(row);write(f'{key[0]}-{r["round"]:02d}.json',row)
    report={'overall':summary(rows),'domains':{d:summary([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')},
        'phases':{'early_0_7':summary([r for r in rows if r['round']<8]),'later_8_25':summary([r for r in rows if r['round']>=8])},
        'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'provider_calls':0,'api_calls':0,'weight_fits':0,
        'inherited_forecast_computations':25584,'diagnostic_only':True,'uses_current_future_actuals':True,
        'limitation':'Reused development hindsight ceilings, not practical policy skill, confidence evidence or an agent evaluation.'}
    write('report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','proposals','output'):p.add_argument(n)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
