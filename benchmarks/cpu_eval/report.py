"""Aggregate all planned configurations without silently dropping failed datasets."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import PROTOCOL, read, write


def report(root, mode):
    protocol=read(PROTOCOL)
    expected=[s['id'] for s in protocol['datasets'] if mode=='evaluation' or s['id'] in protocol['pilot_datasets']]
    summaries,missing=[],[]
    for name in expected:
        path=Path(root)/name.replace('/','_')/'summary.json'
        if not path.exists():
            missing.append(name)
            continue
        result=read(path)
        if result['dataset']!=name or result['mode']!=mode:
            raise ValueError('Unexpected dataset/mode in summary')
        summaries.append(result)
    complete=not missing and all(s['expected_scored_folds']==s['complete_scored_folds'] for s in summaries)
    macro={}
    if complete:
        macro={a:float(np.mean([s['scores_mase'][a] for s in summaries])) for a in protocol['arms']}
    failures=sum(s['model_failures'].get(m,0) for s in summaries for m in protocol['models'])
    calls=sum(s['model_calls'] for s in summaries)
    return {'mode':mode,'planned_configurations':expected,'missing_configurations':missing,
            'complete':complete,'macro_mase':macro,'datasets':summaries,
            'candidate_failures':failures,'candidate_calls':calls,
            'candidate_failure_fraction':failures/calls if calls else None,
            'all_candidate_dispatch_seconds':sum(s['all_candidate_dispatch_seconds'] for s in summaries),
            'pilot_feasible':mode=='pilot' and complete and calls>0 and failures/calls<=.05,
            'promotion':'none; review paired per-configuration results, coverage and failure rates',
            'limits':'No pooled significance claim across six configurations; intervals are exploratory except the declared primary pair. Not official GIFT-Eval scores.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--mode',choices=['pilot','evaluation'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=report(args.runs,args.mode)
    write(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
