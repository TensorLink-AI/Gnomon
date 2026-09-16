"""Real Hermes, synthetic transport, and the exact previously failing case."""
import argparse,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from . import run as controller
from benchmarks.online_retail_ii.resume_ledger import build_history
from benchmarks.online_retail_ii.data import dump


def main():
    p=argparse.ArgumentParser()
    for name in ('repo','baseline','output','plain-python','gnomon-python'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.output);root.mkdir()
    controller.build_history=build_history
    frozen=controller.source_files(Path(args.repo));rows=[]
    for origin in ('2011-03-27','2011-04-10'):
        case=Path(args.baseline)/'agent-cases'/('90214M-'+origin)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(controller.execute_one,args,case,arm,7,'',frozen,True) for arm in controller.ARMS]
            group=[f.result() for f in futures]
        assert all(r['resolved'] and r['exit_code']==0 and r['native_memory_calls']>=1 and r['api_errors']==0 for r in group)
        assert all(r['point']==group[0]['point'] for r in group)
        rows.extend(group)
        report=json.loads((root/'seed-7/ledger/90214M'/origin/'ledger/report.json').read_text())
        past=json.loads((case/'matured-outcomes.json').read_text())['records']
        assert report['matched_origins']==len(past)
        for rank in report['ranking']:
            expected=sum(r['scores'][rank['provider']] for r in past)/len(past)
            assert abs(rank['mean_rmsle']-expected)<1e-12
    dump(root/'rows.json',rows)
    dump(root/'FINISHED.json',{'stub':True,'sessions':6,'resolved':6,'paid_calls':0,
        'source_hashes':frozen,'before_after_dst_rankings_match_original_pairs':True})
    print('6/6 real Hermes preflight sessions passed; zero paid calls',flush=True)

if __name__=='__main__':main()
