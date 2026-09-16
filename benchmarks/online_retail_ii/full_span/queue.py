"""One-shot sequel: waits for successful terminal predecessor before data access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.online_retail_ii.data import dump,sha


def predecessor_status(base):
    base=Path(base);exit_path=base/'launch-001/exit.json';finished=base/'paid-001/FINISHED.json'
    if not exit_path.exists():return 'waiting'
    receipt=json.loads(exit_path.read_text())
    if receipt.get('exit_code')!=0 or not finished.exists():return 'blocked'
    result=json.loads(finished.read_text())
    if result.get('sessions')!=3744 or result.get('stub') is not False:return 'blocked'
    return 'ready'


def run(args):
    root=Path(args.root).resolve();repo=root/'code';out=root/'queue-001'
    out.mkdir()
    inventory=json.loads((repo/'inventory.json').read_text())
    def verify():
        if any(sha(repo/p)!=digest for p,digest in inventory.items()):raise ValueError('Frozen sequel bundle changed')
    verify()
    dump(out/'plan.json',{'predecessor':args.predecessor,'source_inventory_sha256':sha(repo/'inventory.json'),
        'archive_sha256':args.archive_sha256,'planned_periods':43,'planned_sessions':12384,
        'fresh_memory':True,'wait_for_success':True,'source_archive_not_yet_opened':True,
        'scope':'separate_full_span_exploratory_replay'})
    while True:
        status=predecessor_status(args.predecessor)
        dump(out/'state.json',{'status':status,'checked_at_epoch':time.time(),'source_archive_not_yet_opened':True})
        if status=='blocked':raise ValueError('Predecessor did not complete successfully; sequel not started')
        if status=='ready':break
        time.sleep(30)
    verify()
    if sha(args.archive)!=args.archive_sha256:raise ValueError('Source ZIP mismatch')
    predecessor=Path(args.predecessor)
    dump(out/'predecessor-receipt.json',{'finished_sha256':sha(predecessor/'paid-001/FINISHED.json'),
        'exit_sha256':sha(predecessor/'launch-001/exit.json')})
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def command(name,argv):
        verify()
        dump(out/'state.json',{'status':'running','stage':name,'started_at_epoch':time.time()})
        dump(out/(name+'-command.json'),{'argv':argv,'cwd':str(repo)})
        with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
            proc=subprocess.Popen(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
            dump(out/(name+'-process.json'),{'pid':proc.pid,'started_at_epoch':time.time()})
            code=proc.wait()
        dump(out/(name+'-exit.json'),{'exit_code':code,'finished_at_epoch':time.time()})
        if code:raise ValueError(name+' failed; no automatic retry or subsequent stage')
        verify()
    py=sys.executable;panel=root/'panel-001';baseline=root/'baselines-001'
    command('prepare',[py,'-m','benchmarks.online_retail_ii.full_span.cli','prepare','--archive',args.archive,'--output',str(panel)])
    command('baselines',[py,'-m','benchmarks.online_retail_ii.full_span.cli','baselines','--panel',str(panel),'--output',str(baseline)])
    common=[py,'-m','benchmarks.online_retail_ii.full_span.controller','--repo',str(repo),
        '--baseline',str(baseline),'--manifest',str(panel/'manifest.json'),'--panel',str(panel),
        '--plain-python',args.plain_python,'--gnomon-python',args.gnomon_python,'--workers','6']
    preflight=root/'preflight-001'
    command('preflight',[*common,'--output',str(preflight),'--stub'])
    receipt=json.loads((preflight/'FINISHED.json').read_text())
    if receipt!={'stub':True,'sessions':6,'resolved':6,'paid_calls':0}:raise ValueError('Sequel preflight incomplete')
    rows=[json.loads(line) for line in (preflight/'scores.jsonl').read_text().splitlines()]
    if len(rows)!=6 or not all(r['resolved'] and r['exit_code']==0 and r['native_memory_calls']>=1 for r in rows):
        raise ValueError('Native memory/execution preflight failed')
    for day in {r['origin'] for r in rows}:
        group=[r for r in rows if r['origin']==day]
        if len(group)!=3 or any(r['point']!=group[0]['point'] for r in group):raise ValueError('Backend forecast parity failed')
    command('paid',[*common,'--output',str(root/'paid-001'),'--credentials',args.credentials])
    dump(out/'FINISHED.json',{'status':'complete','paid_receipt_sha256':sha(root/'paid-001/FINISHED.json')})


def main():
    p=argparse.ArgumentParser()
    for name in ('root','predecessor','archive','archive-sha256','plain-python','gnomon-python','credentials'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        path=Path(args.root)/'queue-001'
        if path.exists():dump(path/'INCOMPLETE.json',{'cause':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':main()
