"""Build a fresh, isolated candidate-100 dispatch bundle without launching it."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from .contrast_plan_100 import read,sha
from .launch_contrast_100 import verify_inputs

MODULES=('control_contrast_100.py','launch_contrast_100.py','contrast_plan_100.py',
         'contrast_readiness_100.py','recheck_predecessor_100.py','launch_workflow_097.py',
         'control_continuation_097.py','control_collection_096.py','costs_guarded_093.py',
         'continue_collection_096.py','continue_guarded_093.py','launch_collection_096.py')


def build(output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh bundle directory required')
    output.mkdir(parents=True)
    payload=output/'payload';code=payload/'code/benchmarks/ledger_optimization';code.mkdir(parents=True)
    (code/'__init__.py').write_text('');(code.parent/'__init__.py').write_text('')
    for name in MODULES:shutil.copyfile(Path(__file__).with_name(name),code/name)
    copies={'results/contrast-capsule-100-offline-003/capsule':'capsule',
            'results/workflow-097-offline-002/capsule':'previous-capsule'}
    for source,name in copies.items():
        if any(p.is_symlink() for p in Path(source).rglob('*')):raise ValueError('Source contains symlink')
        shutil.copytree(source,payload/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files={'results/contrast-100-prospective-plan-003/plan.json':'plan.json',
           'results/workflow-097-prospective-plan-001/plan.json':'parent-plan.json',
           'results/contrast-100-launch-integration-001/normalized-preflight.json':'preflight.json',
           'results/contrast-100-continuation-preflight-001/passed.json':'continuation-preflight.json',
           'results/ledger-ml-continuous-022/export-002/host-jobs.json':'host-jobs.json'}
    files.update({'results/contrast-capsule-100-worker-003/'+name:'worker-proof/'+name for name in ('passed.json','report.json','manifest.json')})
    for source,name in files.items():
        (payload/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,payload/name)
    verify_inputs(payload/'parent-plan.json',payload/'capsule',payload/'worker-proof',payload/'host-jobs.json',payload/'plan.json',payload/'preflight.json')
    continuation=read(payload/'continuation-preflight.json')
    if not continuation['passed'] or continuation['engy_calls']!=0:raise ValueError('Continuation proof did not pass')
    for name,digest in continuation['continuation_sources'].items():
        if sha(code/name)!=digest:raise ValueError('Continuation helper changed')
    if continuation['frozen_sources']!=read(payload/'capsule/capsule.json')['sources']:
        raise ValueError('Continuation tested a different worker')
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    checks=[]
    for module in ('control_contrast_100','launch_contrast_100','recheck_predecessor_100','continue_collection_096','control_continuation_097'):
        argv=[sys.executable,'-m','benchmarks.ledger_optimization.'+module,'--help']
        result=subprocess.run(argv,cwd=payload/'code',env=env,text=True,capture_output=True)
        (output/(module+'.stdout')).write_text(result.stdout);(output/(module+'.stderr')).write_text(result.stderr)
        checks.append({'argv':argv,'cwd':str(payload/'code'),'exit_status':result.returncode});result.check_returncode()
    script="from pathlib import Path;from benchmarks.ledger_optimization.launch_contrast_100 import verify_inputs;p=Path('..');verify_inputs(p/'parent-plan.json',p/'capsule',p/'worker-proof',p/'host-jobs.json',p/'plan.json',p/'preflight.json');print('copied inputs verified')"
    result=subprocess.run([sys.executable,'-c',script],cwd=payload/'code',env=env,capture_output=True,text=True)
    (output/'copied-inputs.stdout').write_text(result.stdout);(output/'copied-inputs.stderr').write_text(result.stderr);result.check_returncode()
    inventory={str(p.relative_to(payload)):sha(p) for p in payload.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (payload/'SHA256SUMS.json').write_text(json.dumps(inventory,indent=2)+'\n')
    with tarfile.open(output/'dispatch.tar.gz','x:gz') as tar:
        for name in sorted([*inventory,'SHA256SUMS.json']):tar.add(payload/name,arcname=name,recursive=False)
    receipt={'status':'built_not_deployed_or_launched','files':len(inventory),'archive_bytes':(output/'dispatch.tar.gz').stat().st_size,
             'archive_sha256':sha(output/'dispatch.tar.gz'),'plan_sha256':sha(payload/'plan.json'),
             'continuation_preflight_sha256':sha(payload/'continuation-preflight.json'),'builder_sha256':sha(__file__),
             'help_import_checks':checks,'copied_input_validation':True,'engy_calls':0,'final_gate_opened':False,
             'requires':'terminal 097 evidence, independent re-audit, runtime and source checks before a fresh pilot'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.output),indent=2))
