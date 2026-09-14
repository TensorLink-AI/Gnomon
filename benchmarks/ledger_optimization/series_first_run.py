"""Frozen071 isolated068runner with only mature-neighbor ordering changed."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
from .series_first_context import retrieve


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def run(source,anchors,incumbent,output):
    here=Path(__file__).parent;output=Path(output);incumbent=Path(incumbent)
    if output.exists():raise FileExistsError(output)
    receiptpath=here/'evidence/search-ensemble-068.json';receipt=json.loads(receiptpath.read_text());old={}
    for name,h in receipt['files'].items():
        if len(Path(name).stem)!=64:continue
        p=incumbent/name
        if sha(p)!=h:raise ValueError('Changed068incumbent')
        r=json.loads(p.read_text());old[r['task_id']]=r
    if len(old)!=416:raise ValueError('Full incumbent cohort required')
    spec=importlib.util.spec_from_file_location('benchmarks.ledger_optimization._search_ensemble_071',here/'search_ensemble_run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.retrieve=retrieve
    amendment={'protocol':'SERIES_FIRST_071.md','code_sha256':{n:sha(here/n) for n in ('SERIES_FIRST_071.md','series_first_context.py','series_first_run.py','lifetime_context.py','search_ensemble_run.py')},
        'incumbent_receipt_sha256':sha(receiptpath),'block_simplexes':4,'retrieval_policy':'same_series_then_context_distance','weights_per_simplex':7,'base068modified':False,'validation_or_final_access':False,
        'provider_calls':0,'api_calls':0,'additional_inherited_068comparison_weight_fits':832}
    try:
        base=runner.run(source,anchors,output);write(output/'base-report.json',base);rows=[]
        for tid,prior in sorted(old.items()):
            current=json.loads((output/(tid+'.json')).read_text())
            if any(current[k]!=prior[k] for k in ('series_id','origin','round','actual')):raise ValueError('Incumbent task identity mismatch')
            rows.append({**{k:current[k] for k in ('task_id','series_id','origin','round')},'scores':{**current['scores'],
                'incumbent_search_control':prior['scores']['control'],'incumbent_search_ledger':prior['scores']['ledger']}})
        def summarize(rs):
            means={a:mean(r['scores'][a] for r in rs) for a in rs[0]['scores']}
            return {'cases':len(rs),'mean_rmsle':means,'ledger_reduction':{a:1-means['ledger']/v for a,v in means.items() if a!='ledger'}}
        report={**base,'overall':summarize(rows),'domains':{d:summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')},
            'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),'later_8_25':summarize([r for r in rows if r['round']>=8])},
            'additional_inherited_068comparison_weight_fits':832,'amendment':'SERIES_FIRST_071.md'}
        g=report['overall']['ledger_reduction'];guards=('control','strong_block_cv','lifetime_ledger','incumbent_search_ledger')
        report['development_gate_passed']=g['control']>=.2 and g['strong_block_cv']>=.2 and g['lifetime_ledger']>0 and g['incumbent_search_ledger']>0 and all(d['ledger_reduction'][a]>0 for d in report['domains'].values() for a in guards)
        write(output/'comparison.json',rows);write(output/'report.json',report);return report
    except BaseException as e:
        if output.exists():write(output/'amendment-failure.json',{'error':type(e).__name__,'message':str(e)})
        raise
    finally:
        if output.exists():write(output/'amendment.json',amendment)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','anchors','incumbent','output'):p.add_argument(n)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
