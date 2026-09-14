"""Read-only process verification and non-destructive evidence mirroring for 092."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from .pod_ml_v4 import SSH, HOST, REPO

REMOTE = '/root/gnomon-ledger-ml-v3/code/results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('status', 'mirror'))
    args = parser.parse_args()
    script = '''
from pathlib import Path
import json
base=Path('/root/gnomon-ledger-ml-v3/code/results')
value={}
for name,launch in [('preflight','corrected-history-092-setup-002/preflight-launch.json'),
                    ('trial','corrected-history-092-launch.json')]:
    path=base/launch
    if not path.exists():
        value[name]={'launched':False};continue
    record=json.loads(path.read_text());proc=Path('/proc')/str(record['pid'])
    try:
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        live=(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
              and stat[19]==record['start_ticks'] and stat[0] not in ('Z','X'))
    except FileNotFoundError:live=False
    value[name]={'launched':True,'process_verified_live':live,'launch':record}
root=base/'corrected-history-092-run-001'
for name in ('pipeline-status.json','GATE.json','FINISHED.json','BLOCKED.json','INCOMPLETE.json',
             'INTEGRITY_STOP.json','pilot/status.json','evaluation/status.json'):
    file=root/name
    if file.exists():value[name]=json.loads(file.read_text())
preflight=base/'corrected-history-092-preflight-001/passed.json'
if preflight.exists():
    passed=json.loads(preflight.read_text())
    value['preflight_result']={'passed':passed['passed'],'checks':len(passed['checks'])}
admissions=[]
for file in root.glob('**/service-admission/status.json'):
    state=json.loads(file.read_text())
    if not state['ready']:admissions.append({'path':str(file),'probe_count':state['probe_count'],
        'waiting':state['waiting'],'terminal_cause':state['terminal_cause']})
value['unready_service_admissions']=admissions
print(json.dumps(value,indent=2))
'''
    result = subprocess.run([*SSH, HOST, 'python3', '-'], input=script, text=True,
                            capture_output=True, timeout=40, check=True)
    if args.operation == 'status':
        print(result.stdout, end='')
        return
    state = json.loads(result.stdout)
    names = ['corrected-history-092-setup', 'corrected-history-092-setup-002',
             'corrected-history-092-preflight-001']
    if state['trial']['launched']:
        names += ['corrected-history-092-run-001', 'corrected-history-092-launch.json']
    for name in names:
        subprocess.run(['rsync', '-az', '--partial', '-e', shlex.join(SSH),
                        HOST + ':' + REMOTE + '/' + name, str(REPO / 'results') + '/'], check=True)
    print(json.dumps({'mirrored': names, 'state': state}))


if __name__ == '__main__':
    main()
