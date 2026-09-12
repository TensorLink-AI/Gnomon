"""Read-only live status or non-destructive evidence mirroring for ML v3.

Does not restart a job. A status file alone never establishes liveness.
"""
import argparse
import json
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[2]
HOST='wrk-kadzj08j3t1o@ssh.deployments.targon.com'
REMOTE='/root/gnomon-ledger-ml-v3'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
     '-i','/root/.ssh/targon_gnomon_arena']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['status','mirror'])
    args=p.parse_args()
    script='''
from pathlib import Path
import json,os
base=Path("/root/gnomon-ledger-ml-v3")
root=base/"code/results/hermes-ml-checkpoint-120-v3"
value={}
launch=base/"launch.json"
if launch.exists():
    record=json.loads(launch.read_text());value["launch"]=record
    root=Path(record["argv"][record["argv"].index("--root")+1])
    proc=Path("/proc")/str(record["pid"])
    try:
        stat=(proc/"stat").read_text().rsplit(")",1)[1].split()
        value["process_verified_live"]=(Path("/proc/sys/kernel/random/boot_id").read_text().strip()==record["boot_id"] and stat[19]==record["start_ticks"] and stat[0] not in ("Z","X"))
    except FileNotFoundError:value["process_verified_live"]=False
else:value["process_verified_live"]=False
value["result_root"]=str(root)
for n in ("pipeline-status.json","GATE.json","FINISHED.json","BLOCKED.json","pilot/status.json","evaluation/status.json"):
    file=root/n
    if file.exists():value[n]=json.loads(file.read_text())
print(json.dumps(value,indent=2))
'''
    value=subprocess.check_output([*SSH,HOST,'python3','-'],input=script,text=True)
    if args.operation=='mirror':
        remote_root=json.loads(value)['result_root']
        assert remote_root.startswith(REMOTE+'/code/results/hermes-ml-checkpoint-120-v3')
        target=REPO/'results'/Path(remote_root).name
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az','--partial','-e',' '.join(SSH),
                        HOST+':'+remote_root+'/',str(target)+'/'],check=True)
        print(json.dumps({'mirrored_to':str(target)}))
        return
    print(value,end='')


if __name__=='__main__':main()
