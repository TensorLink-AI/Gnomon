"""Read-only live status for the guarded pilot; never restart or mutate the run."""
import argparse
import re
import subprocess

REMOTE = r'''
import json
from pathlib import Path
from statistics import mean
from datetime import datetime,timezone
base=Path('/root/gnomon-ledger-ml-v3/code/results')
launch=base/'LAUNCH_LABEL';root=base/'RUN_LABEL'
identity=json.loads((launch/'launch.json').read_text())
stat_path=Path('/proc')/str(identity['pid'])/'stat'
stat=stat_path.read_text().split(') ')[1].split() if stat_path.exists() else None
live=bool(stat and stat[0]!='Z' and stat[19]==identity['start_ticks'] and
          Path('/proc/sys/kernel/random/boot_id').read_text().strip()==identity['boot_id'])
rows=[json.loads(p.read_text()) for p in root.glob('*/*/round-*/grade.json')]
keyed={(r['arm'],r['series_id'],r['round']):r for r in rows}
arms=('plain','gnomon','ledger')
matched=sorted({(r['series_id'],r['round']) for r in rows if all((a,r['series_id'],r['round']) in keyed for a in arms)})
requests=list(root.glob('**/api-*-forwarded.json'));responses=list(root.glob('**/api-*-response.json'))
usage=[];errors=0
for p in responses:
    try:r=json.loads(p.read_text())
    except ValueError:r={'error':'unparseable'}
    usage.append(r.get('usage') or {});errors+=bool(r.get('error'))
probes=[json.loads(p.read_text()) for p in root.glob('**/service-admission/probe-*-receipt.json')]
print(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'controller_live':live,'pid':identity['pid'],
 'finished':(launch/'FINISHED.json').exists(),'incomplete':(launch/'INCOMPLETE.json').exists(),
 'completed':len(rows),'planned':json.loads((root/'manifest.json').read_text())['planned'],'arms':{a:{'completed':sum(r['arm']==a for r in rows),
 'valid':sum(r['arm']==a and r['valid'] for r in rows),
 'full_workflows':sum(r['arm']==a and r['workflow_complete'] for r in rows)} for a in arms},
 'matched_cases':len(matched),'matched_mean_rmsle':{a:mean(keyed[a,s,n]['rmsle'] for s,n in matched) if matched else None for a in arms},
 'forwarded_requests':len(requests),'returned_responses':len(responses),'api_errors':errors,
 'reported_tokens':sum(u.get('total_tokens',0) for u in usage),
 'attempts_without_reported_usage_including_inflight':len(requests)-sum('total_tokens' in u for u in usage),
 'readiness_requests':len(probes),'readiness_reported_tokens':sum((p.get('usage') or {}).get('total_tokens',0) for p in probes),
 'readiness_unknown_usage':sum('total_tokens' not in (p.get('usage') or {}) for p in probes),
 'scope':'Development monitoring; matched cases require all three arms, without success filtering. Not final efficacy evidence.'},indent=2))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='guarded-history-093-pilot-001')
    parser.add_argument('--launch',default='guarded-history-093-launch-001')
    args=parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',s) for s in (args.run,args.launch)):
        parser.error('Use directory labels inside the remote results directory')
    script=REMOTE.replace('LAUNCH_LABEL',args.launch).replace('RUN_LABEL',args.run)
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
                    '-i','/root/.ssh/targon_gnomon_arena','wrk-kadzj08j3t1o@ssh.deployments.targon.com',
                    'python3 -'],input=script,text=True,check=True)


if __name__=='__main__':main()
