"""Paired regression summaries; preserve series together in time-block resampling."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np

if __package__:
    from .run import TASKS
else:
    from run import TASKS


def score(values, baseline, *, relative=False):
    if relative:
        den = np.nansum(baseline, axis=0)
        use = den > 0
        return float(np.mean(np.nansum(values, axis=0)[use]/den[use]))
    return float(np.nanmean(values))


def paired(reference, candidate, *, relative=False, block=28, draws=2000):
    old = {(d['origin'],d['series_id']):d for d in reference}
    new = {(d['origin'],d['series_id']):d for d in candidate}
    if old.keys() != new.keys():
        raise ValueError('Arms do not have the same scored origins and series')
    origins, series = sorted({k[0] for k in old}), sorted({k[1] for k in old})
    ti, si = {v:i for i,v in enumerate(origins)}, {v:i for i,v in enumerate(series)}
    A, B, base = [np.full((len(origins),len(series)),np.nan) for _ in range(3)]
    for k,d in old.items():
        i,j=ti[k[0]],si[k[1]]
        if d['losses'] != new[k]['losses']:
            raise ValueError('Candidate forecasts or targets changed between arms')
        A[i,j],B[i,j] = d['served_loss'],new[k]['served_loss']
        # replay loss dictionaries are in baseline-first provider order.
        base[i,j] = next(iter(d['losses'].values()))
    a,b=score(A,base,relative=relative),score(B,base,relative=relative)
    rng=np.random.default_rng(0);boot=[]
    for _ in range(draws):
        starts=rng.integers(len(origins),size=int(np.ceil(len(origins)/block)))
        ix=((starts[:,None]+np.arange(block))%len(origins)).ravel()[:len(origins)]
        boot.append(score(B[ix],base[ix],relative=relative)-score(A[ix],base[ix],relative=relative))
    lo,hi = map(float,np.quantile(boot,[.025,.975]))
    out={'current_score':a,'candidate_score':b,'percent_change':100*(b/a-1),'difference':b-a,
         'ci95_time_block':[lo,hi],'block_origins':block,'origins':len(origins),'series':len(series),
         'paired_forecasts':len(old),'changed_choice_share':sum(old[k]['served']!=new[k]['served'] for k in old)/len(old),
         'verdict':'lower_error' if hi<0 else 'higher_error' if lo>0 else 'uncertain'}
    if relative:
        delta=np.nansum(B,axis=0)/np.nansum(base,axis=0)-np.nansum(A,axis=0)/np.nansum(base,axis=0)
        out['ci95_series']=list(map(float,np.quantile([rng.choice(delta,len(delta)).mean() for _ in range(draws)],[.025,.975])))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args()
    reports={};hashes=set()
    def read(task,arm):
        meta=json.loads((args.output/f'{task}.{arm}.json').read_text())
        hashes.add(meta['source_sha256'])
        with gzip.open(args.output/f'{task}.{arm}.decisions.json.gz','rt') as f:rows=json.load(f)
        return meta,rows
    for task in TASKS:
        m,old=read(task,'current');arms=['no_memory','bounded','fase']+(['v2_profile'] if task in ('alpha','favorita') else [])
        reports[task]={'input_sha256':m['input_sha256'],'baseline_old_code_checks':len(m['baseline_old_code_checks']),'arms':{}}
        assert len(m['baseline_old_code_checks'])>=30
        for arm in arms:
            n,new=read(task,arm);assert n['input_sha256']==m['input_sha256']
            reports[task]['arms'][arm]=paired(old,new,relative=task=='favorita',block=112 if task=='favorita' else 28)
            reports[task]['arms'][arm]['switch_rate']=n['switch_rate']
            if task=='alpha':
                for label,before in [('regression',True),('extension',False)]:
                    reports[task]['arms'][arm][label]=paired([d for d in old if (d['origin']<'2026-08-01')==before],
                        [d for d in new if (d['origin']<'2026-08-01')==before])
    assert len(hashes)==1, 'Different source versions were evaluated'
    result={'source_sha256':hashes.pop(),'scope':'previously inspected regression windows; exploratory, unadjusted intervals', 'tasks':reports}
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('task | arm | error change vs current | paired 95% interval | verdict')
    for task,r in reports.items():
        for arm,v in r['arms'].items():
            print(f"{task} | {arm} | {v['percent_change']:+.2f}% | {v['ci95_time_block']} | {v['verdict']}")


if __name__=='__main__':main()
