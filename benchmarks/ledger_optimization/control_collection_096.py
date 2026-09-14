"""Retain one gated collection pilot's process and evidence; never restart it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(pid):
    fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'start_ticks':fields[19],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def completion(output, exit_status):
    """A failed completion gate is an observed pilot outcome, not missing evidence."""
    if exit_status != 0:
        return {'complete':False,'cause':'pilot_process_failed','exit_status':exit_status}
    try:
        runner=json.loads((output/'runner-exit.json').read_text())
        report=json.loads((output/'report.json').read_text())
        gate=json.loads((output/'GATE.json').read_text())
        clean=(runner['exit_status']==0 and runner['continuation_launched'] is False
               and report['complete'] is True and not report['audit_failures']
               and not report['shutdown_record_gaps']
               and gate['accuracy_used_for_gate'] is False
               and isinstance(gate['passed'],bool))
        return {'complete':clean,'cause':None if clean else 'pilot_evidence_incomplete',
                'continuation_gate_passed':gate['passed'],
                'continuation_launched':False,'exit_status':exit_status}
    except (OSError,ValueError,KeyError,TypeError):
        return {'complete':False,'cause':'pilot_completion_receipts_missing_or_invalid',
                'exit_status':exit_status}


def archive_evidence(launch, output, forbidden=()):
    files={}
    for prefix,root in (('launch',launch),('pilot',output)):
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            # Do not follow runtime links out of the retained evidence tree.
            if path.is_symlink():
                raise ValueError('Evidence contains an unarchived symlink')
            if path.is_file():
                files[f'{prefix}/{path.relative_to(root)}']=path
    if any(value in path.read_bytes() for path in files.values() for value in forbidden):
        raise ValueError('Credential scan failed; archive withheld')
    inventory={name:sha(path) for name,path in files.items()}
    dump(launch/'SHA256SUMS.json',inventory)
    with tarfile.open(launch/'evidence.tar.gz','x:gz') as archive:
        for name,path in files.items():
            archive.add(path,arcname=name,recursive=False)
        archive.add(launch/'SHA256SUMS.json',arcname='SHA256SUMS.json')
    return {'archive_sha256':sha(launch/'evidence.tar.gz'),
            'inventory_sha256':sha(launch/'SHA256SUMS.json'),'files':len(files)}


def supervise(command, launch, output, credentials_file=None):
    launch,output=Path(launch).absolute(),Path(output).absolute()
    if (launch==output or launch in output.parents or output in launch.parents
            or launch.exists() or launch.is_symlink() or output.exists() or output.is_symlink()):
        raise ValueError('Require separate fresh launch and pilot directories')
    launch.mkdir(parents=True)
    dump(launch/'launch.json',{**identity(os.getpid()),
        'at':datetime.now(timezone.utc).isoformat(),'controller_sha256':sha(__file__),
        'continuation_authorized':False})
    dump(launch/'command.json',{'argv':command,'cwd':str(Path.cwd()),'output':str(output)})
    (launch/'controller-source.py').write_bytes(Path(__file__).read_bytes())
    # No retries, timeout extensions, shell execution, or credential copying.
    with (launch/'pilot.stdout').open('xb') as stdout, (launch/'pilot.stderr').open('xb') as stderr:
        try:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
        except OSError as exc:
            dump(launch/'spawn-error.json',{'type':type(exc).__name__})
            status=127
        else:
            dump(launch/'pilot-process.json',identity(child.pid))
            status=child.wait()
    dump(launch/'pilot-exit.json',{'exit_status':status})
    result=completion(output,status)
    dump(launch/'AUDITED.json',result)
    if not result['complete']:
        dump(launch/'INCOMPLETE.json',result)
    forbidden=[]
    # The launcher creates this receipt only after prerequisite/build validation,
    # immediately before reading the key. Rejected launches never read it here.
    if credentials_file is not None and (output/'accepted-launch.json').exists():
        for line in Path(credentials_file).read_text().splitlines():
            if line.startswith('ENGY_API_KEY='):
                value=line.split('=',1)[1].strip().strip('"').strip("'")
                if value:forbidden.append(value.encode())
        if not forbidden:
            raise ValueError('Credential scan unavailable; archive withheld')
    archived=archive_evidence(launch,output,forbidden)
    dump(launch/'FINISHED.json',{**result,**archived,
         'at':datetime.now(timezone.utc).isoformat(),'continuation_launched':False})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-directory',type=Path,required=True)
    fields=('capsule','plan','preflight','task-source','runtime','paid-predecessor',
            'seed-predecessor','output','credentials-file')
    for field in fields:
        parser.add_argument('--'+field,type=Path,required=True)
    args=parser.parse_args()
    command=[sys.executable,'-m','benchmarks.ledger_optimization.launch_collection_096']
    for field in fields:
        command+=['--'+field,str(getattr(args,field.replace('-','_')).absolute())]
    result=supervise(command,args.launch_directory,args.output,args.credentials_file)
    print(json.dumps(result))
    if not result['complete']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
