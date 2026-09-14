"""Pilot gate, conditional fresh evaluation, independent audit and evidence archive."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from .run import HERE, OTHER, REPO, key, TASK_SOURCE, SOURCE_SHA, ARMS
from .transport import dump, sha


def gate(report):
    complete=report['complete'] and not report['audit_failures']
    arms={a:{'completed':report['arms'][a]['workflow_complete'],'tasks':report['arms'][a]['tasks']}
          for a in ('plain','gnomon','ledger')}
    passed=complete and all(v['tasks']==12 and v['completed']>=11 for v in arms.values())
    return {'passed':passed,'threshold':.90,'required_per_arm':'11 of 12 full workflows',
            'arms':arms,'integrity_passed':complete,'accuracy_used_for_promotion':False}


def archive(root):
    receipts=[json.loads(p.read_text()) for p in root.glob('**/service-admission/probe-*-receipt.json')]
    dump(root/'service-admission-costs.json',{'api_calls':len(receipts),
         'reported_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in receipts),
         'usage_complete':all('total_tokens' in (r.get('usage') or {}) for r in receipts),
         'scope':'All admission probes, including sessions never started; no excluded costs.'})
    credential=key().encode()
    files=[p for p in root.rglob('*') if p.is_file() and p.name not in ('evidence.tar.gz','SHA256SUMS.json')]
    for file in files:
        if credential in file.read_bytes():raise RuntimeError('Credential detected; archive withheld')
    inventory={str(f.relative_to(root)):sha(f) for f in files}
    dump(root/'SHA256SUMS.json',inventory)
    with tarfile.open(root/'evidence.tar.gz','w:gz') as tar:
        for name in [*inventory,'SHA256SUMS.json']:tar.add(root/name,arcname=name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    python=OTHER/'gnomon-venv/bin/python';env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        preflight_path=args.preflight.resolve() if args.preflight else root/'preflight-001/passed.json'
        preflight=json.loads(preflight_path.read_text())
        assert preflight['passed']
        assert all(sha(HERE/n)==h for n,h in preflight['tested_sources'].items())
        dump(root/'accepted-preflight.json',{'path':str(preflight_path),'sha256':sha(preflight_path)})
        assert sha(HERE/'numerical.py')==sha(REPO/'benchmarks/hermes_ml_iteration/numerical.py')
        assert ARMS==('plain','gnomon','ledger')
        assert sha(TASK_SOURCE)==SOURCE_SHA
        previous=REPO/'results/hermes-ml-checkpoint-120-v3-pod-002'
        preserved={n:sha(previous/n) for n in ('pilot/report.json','FINISHED.json','SHA256SUMS.json')}
        dump(root/'previous-evidence-hashes.json',preserved)
        frozen={p.name:sha(p) for p in HERE.iterdir() if p.is_file()}
        dump(root/'frozen-source-hashes.json',frozen)
        setup=root/'setup';setup.mkdir(exist_ok=False)
        # Preserve complete source pins and the wheel; don't mutate shared runtimes.
        for name in ('gnomon_forecast-1.2.0-py3-none-any.whl','hermes-frozen.tar.gz',
                     'gnomon-runtime.json','plain-runtime.json','requirements-matched.txt','branch-build.json'):
            shutil.copyfile(previous/'setup'/name,setup/name)
        shutil.copyfile(TASK_SOURCE,setup/'continuous-host-jobs.json')
        evaluated=False
        for phase in ('pilot','evaluation'):
            assert all(sha(HERE/n)==h for n,h in frozen.items())
            dump(root/'pipeline-status.json',{'phase':phase,'at':datetime.now(timezone.utc).isoformat()})
            argv=[str(python),'-m','benchmarks.hermes_ml_checkpoint_v5.run','--output',str(root/phase)]
            if phase=='pilot':argv.append('--pilot')
            dump(root/(phase+'-command.json'),{'argv':argv,'cwd':str(REPO)})
            with (root/(phase+'-run.log')).open('x') as f:
                result=subprocess.run(argv,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(phase+' runner exited '+str(result.returncode))
            with (root/(phase+'-audit.log')).open('x') as f:
                subprocess.run([str(python),'-m','benchmarks.hermes_ml_checkpoint_v5.analyze',str(root/phase)],
                               cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            report=json.loads((root/phase/'report.json').read_text())
            if not report['complete'] or report['audit_failures']:raise RuntimeError(phase+' integrity audit failed')
            if phase=='pilot':
                decision=gate(report);dump(root/'GATE.json',decision)
                if not decision['passed']:break
            else:evaluated=True
        assert all(sha(previous/n)==h for n,h in preserved.items())
        assert all(sha(HERE/n)==h for n,h in frozen.items())
        dump(root/'outcome.json',{'pilot_gate':json.loads((root/'GATE.json').read_text()),
                                'evaluation_run':evaluated,'previous_evidence_unchanged':True})
        archive(root)
        dump(root/'FINISHED.json',{'completed':True,'evaluation_run':evaluated,
                                 'at':datetime.now(timezone.utc).isoformat(),'archive_sha256':sha(root/'evidence.tar.gz')})
    except Exception as exc:
        dump(root/'BLOCKED.json',{'error':type(exc).__name__,'message':str(exc),'at':datetime.now(timezone.utc).isoformat(),'automatic_retry':False})
        archive(root)
        dump(root/'INCOMPLETE.json',{'completed':False,'archive_sha256':sha(root/'evidence.tar.gz'),'automatic_retry':False})
        raise


if __name__=='__main__':main()
