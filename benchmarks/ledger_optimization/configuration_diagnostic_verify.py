"""Independent audit of067 from all saved066 production requests and decisions."""
from collections import Counter
import hashlib,json,math
from pathlib import Path
import sys
import numpy as np


def verify(root,source):
    root,source=Path(root),Path(source);checks=0;rows=[];source_hashes=json.loads((source/'SHA256SUMS.json').read_text())
    def check(v,message):
        nonlocal checks
        checks+=1
        if not v:raise AssertionError(message)
    def load(p):return json.loads(p.read_text())
    def near(a,b):check(np.allclose(a,b,rtol=1e-12,atol=1e-12),'Numerical disagreement')
    def read(name):
        p=source/name;check(hashlib.sha256(p.read_bytes()).hexdigest()==source_hashes[name],'Source hash');return load(p)
    def error(p,y):return float(np.sqrt(np.mean(np.square(np.log1p(np.asarray(p))-np.log1p(np.asarray(y))))))
    for p in sorted(root.glob('*.json')):
        if p.name in ('report.json','manifest.json','verification.json'):continue
        r=load(p);tid=r['task_id'];o=read(f'outcomes/{tid}.json');y=o['actual'];produced={}
        check(not o['warmup'] and r['round']==o['round'] and r['origin']==o['origin'] and r['series_id']==o['series_id'],'Scored task identity')
        for forecast in sorted((source/'forecasts'/tid).glob('*.json')):
            f=read(str(forecast.relative_to(source)))
            if f['request']['history_end']==730:produced[f['request']['config_id']]=f['point']
        independent={cid:error(point,y) for cid,point in produced.items()}
        check(set(r['production_scores'])==set(independent),'Exactly all produced configurations')
        for cid,v in independent.items():near(r['production_scores'][cid],v)
        originals=None
        for arm in ('control','ledger'):
            d=read(f'cases/{tid}/{arm}/decision.json');b=d['backtests'];a=r['arms'][arm];base=min(b[:6],key=lambda x:x['cv_rmsle']);originals={x['config_id'] for x in b[:6]};selected=d['selected']['config_id']
            check(a['selected_config_id']==selected and a['original_cv_selected']==base['config_id'],'Prospective decisions unaltered')
            check(a['selected_extra']==(selected not in originals),'Extra selection label')
            near(a['original_cv_rmsle'],base['cv_rmsle']);near(a['expanded_cv_rmsle'],min(x['cv_rmsle'] for x in b));near(a['cv_improvement'],a['original_cv_rmsle']-a['expanded_cv_rmsle'])
            near(a['selected_score'],independent[selected]);near(a['original_cv_selected_score'],independent[base['config_id']])
            change=independent[base['config_id']]-independent[selected];near(a['production_change_vs_original_cv'],change)
            check(a['production_comparison']==('improved' if change>1e-12 else 'worsened' if change< -1e-12 else 'tied'),'Production comparison label')
            available=originals|{selected};check(a['produced_configurations']==len(available) and a['tested_configurations']==17 and a['unobserved_production_configurations']==17-len(available),'Incomplete production coverage disclosed')
            oracle=r['hindsight_choices'][arm];minimum=min(independent[cid] for cid in available);near(oracle['score'],minimum)
            check(oracle['ties']==sorted(cid for cid in available if math.isclose(independent[cid],minimum,abs_tol=1e-14,rel_tol=0)),'Hindsight ties')
            near(a['selection_regret'],independent[selected]-minimum);near(r['scores']['oracle_'+arm],minimum);near(r['scores'][arm],independent[selected])
        for label,available in [('original_six',originals),('union',set(produced))]:
            minimum=min(independent[cid] for cid in available);near(r['scores']['oracle_'+label],minimum);near(r['hindsight_choices'][label]['score'],minimum)
            check(r['hindsight_choices'][label]['selected'] in available,'Hindsight choice available')
        for arm in ('strong_block_cv','lifetime_ledger'):near(r['scores'][arm],error(o['point'][arm],y))
        near(r['scores']['original_cv'],r['arms']['control']['original_cv_selected_score'])
        check(r['same_selection']==(r['arms']['control']['selected_config_id']==r['arms']['ledger']['selected_config_id']),'Shared selection')
        check(r['diagnostic_only'] and r['uses_current_future_actuals'],'Hindsight warning')
        rows.append(r)
    report=load(root/'report.json');check(len(rows)==416 and len({r['task_id'] for r in rows})==416,'Full cohort')
    def summary(group,observed):
        check(observed['cases']==len(group),'Summary count');means={a:float(np.mean([r['scores'][a] for r in group])) for a in group[0]['scores']}
        for a,v in means.items():near(observed['mean_rmsle'][a],v)
        check(observed['same_selection_cases']==sum(r['same_selection'] for r in group),'Shared choice count')
        for a in ('control','ledger'):
            out=observed['arms'][a];check(out['extra_selections']==sum(r['arms'][a]['selected_extra'] for r in group),'Extra count')
            check(out['production_comparisons']==dict(Counter(r['arms'][a]['production_comparison'] for r in group)),'Comparison counts')
            check(out['configuration_counts']==dict(Counter(r['arms'][a]['selected_config_id'] for r in group)),'Configuration frequencies')
            for f in ('cv_improvement','selection_regret'):near(out['mean_'+f],np.mean([r['arms'][a][f] for r in group]))
        for a,c in observed['ceiling_vs_strong_block_cv'].items():
            near(c['maximum_reduction'],1-means[a]/means['strong_block_cv']);check(c['twenty_percent_status']==('impossible_for_these_saved_choices' if means[a]>.8*means['strong_block_cv'] else 'not_ruled_out_not_established'),'Ceiling interpretation')
    summary(rows,report['overall'])
    for domain in ('electricity','pedestrian'):summary([r for r in rows if r['series_id'].startswith(domain+':')],report['domains'][domain])
    summary([r for r in rows if r['round']<8],report['phases']['early_0_7']);summary([r for r in rows if r['round']>=8],report['phases']['later_8_25'])
    check(report['provider_calls']==report['api_calls']==report['new_weight_fits']==0,'Read-only cost');check(report['inherited_forecast_computations']==24032+12984+12600,'Inherited computations')
    manifest=load(root/'manifest.json');here=Path(__file__).parent
    for n,h in manifest['code_sha256'].items():check(hashlib.sha256((here/n).read_bytes()).hexdigest()==h,'Frozen diagnostic')
    result={'checks':checks,'failures':0,'cases':len(rows),'provider_calls':0,'api_calls':0,'scope':'Recompute all production scores from saved forecasts, all original/expanded choices, available-choice ceilings and aggregates. No claim about unproduced forecasts.',
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
