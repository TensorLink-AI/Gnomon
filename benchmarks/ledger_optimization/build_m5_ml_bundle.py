"""Package fixed development inputs and isolated host code; never dispatch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from . import launch_m5_ml as launch
from . import m5_ml_launch_inputs as inputs

PREPARATION_SOURCE_SHA = '3ac1b25471c405fea8c4ffa7aa115e396ab1e5f64885b2642328630826937eff'
SEEDS = {
    7: ('results/m5-ml-capsule-offline-001/capsule',
        'results/m5-ml-capsule-offline-001/synthetic-worker-002'),
    19: ('results/m5-ml-seed-19-offline-001/capsule',
         'results/m5-ml-seed-19-offline-001/probe'),
}


def checked_copy(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if any(p.is_symlink() for p in (source, *source.parents)) or not source.is_file():
        raise ValueError('Regular explicit source file required')
    raw = source.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Source hash mismatch: '+source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(raw)


def build(output, *, host_preflight=None):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh bundle directory required')
    here = Path(__file__).parent
    tested = launch.source_identity()
    if inputs.sha(here/'m5_prepare.py') != PREPARATION_SOURCE_SHA:
        raise ValueError('Original tested preparation dependency changed')
    # All fixed input authentication precedes copying; credentials are not inputs.
    manifest = Path('results/ledger-optimization/m5-panel-014/manifest.json')
    jobs = Path('results/m5-ml-development-prepare-001/prepared/development-jobs.json')
    parent = Path('results/contrast-100-prospective-plan-003/plan.json')
    plans = {}
    for seed, (capsule, worker) in SEEDS.items():
        plan = Path(f'results/m5-ml-launch-inputs-001/plan-seed-{seed}.json')
        if inputs.sha(plan) != launch.PLAN_HASHES[seed]:
            raise ValueError('Fixed seed plan changed')
        registry = Path(launch.read(plan)['dispatch_registry'])
        plans[seed] = inputs.verify_plan(plan, parent, capsule, worker,
            manifest.read_bytes(), jobs.read_bytes(), seed=seed, registry=registry)
        if host_preflight is not None:
            launch.verify_host_preflight(host_preflight, plans[seed])
    output.mkdir(parents=True)
    payload = output/'payload'; code = payload/'code'
    sources = {}

    def add(source, name, digest=None):
        source = Path(source)
        digest = digest or inputs.sha(source)
        checked_copy(source, payload/name, digest)
        sources[str(source)] = digest

    for name, digest in tested.items():
        destination = Path('code/benchmarks/ledger_optimization')/name
        add(here/name, destination, digest)
    add(here/'m5_prepare.py', 'code/benchmarks/ledger_optimization/m5_prepare.py', PREPARATION_SOURCE_SHA)
    add(__file__, 'code/benchmarks/ledger_optimization/build_m5_ml_bundle.py')
    for package in ('benchmarks', 'benchmarks/ledger_optimization', 'benchmarks/tests'):
        p = code/package/'__init__.py';p.parent.mkdir(parents=True, exist_ok=True);p.write_text('')
    for source, name in ((manifest,'panel-manifest.json'),(jobs,'development-jobs.json'),(parent,'parent-plan.json')):
        add(source,name)
    for seed, (capsule, worker) in SEEDS.items():
        capsule, worker = Path(capsule), Path(worker)
        cap = plans[seed]['capsule']; prefix = Path(f'seed-{seed}')
        add(f'results/m5-ml-launch-inputs-001/plan-seed-{seed}.json',prefix/'plan.json',launch.PLAN_HASHES[seed])
        add(capsule/'capsule.json',prefix/'capsule/capsule.json',inputs.IDENTITIES[seed]['capsule.json'])
        add(capsule/'cohort-contract.json',prefix/'capsule/cohort-contract.json',cap['cohort_contract_sha256'])
        for name,digest in cap['sources'].items():
            if Path(name).name != name:
                raise ValueError('Capsule contains an unexpected nested source')
            add(capsule/'benchmarks/hermes_ml_checkpoint_v6'/name,
                prefix/'capsule/benchmarks/hermes_ml_checkpoint_v6'/name,digest)
        for name in ('passed.json','report.json','manifest.json'):
            add(worker/name,prefix/'worker-proof'/name,inputs.IDENTITIES[seed][name])
    if host_preflight is not None:
        add(host_preflight,'host-preflight.json')

    script = '''import json
from pathlib import Path
from benchmarks.ledger_optimization import launch_m5_ml as launch
from benchmarks.ledger_optimization import m5_ml_launch_inputs as inputs
p=Path('..'); plans=[]
for seed in (7,19):
 s=p/f'seed-{seed}'; plan=launch.read(s/'plan.json')
 plans.append(inputs.verify_plan(s/'plan.json',p/'parent-plan.json',s/'capsule',s/'worker-proof',
  (p/'panel-manifest.json').read_bytes(),(p/'development-jobs.json').read_bytes(),
  seed=seed,registry=Path(plan['dispatch_registry'])))
 if (p/'host-preflight.json').exists():launch.verify_host_preflight(p/'host-preflight.json',plan)
print(json.dumps({'host_sources':launch.source_identity(),'seeds':[p['requested_seed'] for p in plans],
 'inputs_verified':True,'dispatch_reserved':False,'engy_calls':0}))
'''
    (output/'verify_copied.py').write_text(script)
    env = dict(os.environ); env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
    commands = [([sys.executable,'-m','benchmarks.ledger_optimization.launch_m5_ml','--help'],'help'),
                ([sys.executable,'-c',script],'copied-inputs')]
    checks = []
    for argv,label in commands:
        result = subprocess.run(argv,cwd=code,env=env,text=True,capture_output=True)
        (output/(label+'.stdout')).write_text(result.stdout)
        (output/(label+'.stderr')).write_text(result.stderr)
        check={'argv':argv,'cwd':str(code),'exit_status':result.returncode}
        (output/(label+'.command.json')).write_text(json.dumps(check,indent=2)+'\n')
        checks.append(check);result.check_returncode()
    verified = json.loads((output/'copied-inputs.stdout').read_text())
    if verified['host_sources'] != tested or launch.source_identity() != tested:
        raise ValueError('Host sources changed during packaging')
    for source,digest in sources.items():
        if inputs.sha(source)!=digest:raise ValueError('Input changed during packaging')
    inventory = {str(p.relative_to(payload)):inputs.sha(p) for p in payload.rglob('*') if p.is_file()}
    (payload/'SHA256SUMS.json').write_text(json.dumps(inventory,indent=2)+'\n')
    with tarfile.open(output/'bundle.tar.gz','x:gz') as tar:
        for name in sorted([*inventory,'SHA256SUMS.json']):
            tar.add(payload/name,arcname=name,recursive=False)
    receipt={'status':'built_not_deployed_or_launched','host_preflight_included':host_preflight is not None,
             'host_sources':tested,'dependency_sources':{'m5_prepare.py':PREPARATION_SOURCE_SHA},
             'source_inputs':sources,'files':len(inventory),'archive_bytes':(output/'bundle.tar.gz').stat().st_size,
             'archive_sha256':inputs.sha(output/'bundle.tar.gz'),'checks':checks,
             'runtime_included':False,'credentials_included':False,'reserved_targets_included':False,
             'engy_calls':0,'provider_calls':0,'dispatch_reserved':False,'final_gate_opened':False,
             'requires':'Full exact-source host proof, terminal predecessor reconciliation, and actual remote runtime checks remain launch prerequisites.'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host-preflight',type=Path)
    args=p.parse_args();print(json.dumps(build(args.output,host_preflight=args.host_preflight),indent=2))
