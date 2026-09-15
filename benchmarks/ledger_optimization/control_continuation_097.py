"""Archive one gated continuation, including failed sessions and all request costs.

The reused archive helper labels the complete run directory `pilot/` inside the
tar; the manifest distinguishes its 36 retained and 276 new sessions. No new
pilot, retry, final evaluation, or accuracy-dependent promotion is started here.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from . import control_collection_096 as archive
from . import costs_guarded_093 as cost_accounting


def completion(output, status):
    if status != 0:
        return {'complete': False, 'cause': 'continuation_process_failed', 'exit_status': status}
    try:
        read = lambda name: json.loads((output/name).read_text())
        manifest, done, runner, report = (read(n) for n in
            ('manifest.json', 'complete.json', 'runner-exit.json', 'report.json'))
        jobs = read('host-jobs.json')
        expected = {(a,s,j['round']) for s,values in jobs.items() for j in values
                    for a in ('plain','gnomon','ledger')}
        actual = {(r['arm'],r['series_id'],r['round']) for r in report['rows']}
        clean = (manifest['planned'] == done['completed'] == done['planned'] == 312
            and manifest['pilot_sessions_retained'] == done['retained_pilot_sessions'] == 36
            and manifest['new_sessions_planned'] == done['new_sessions'] == 276
            and done['source_unchanged'] is True and manifest['accuracy_used_for_promotion'] is False
            and runner == {'exit_status':0,'final_gate_opened':False}
            and report['complete'] is True and report['audit_failures'] == []
            and report['shutdown_record_gaps'] == [] and len(report['rows']) == len(expected) == 312
            and actual == expected and sum(n < 3 for _,_,n in actual) == 36)
        return {'complete':clean,'cause':None if clean else 'continuation_evidence_incomplete',
            'exit_status':status,'retained_pilot_sessions':36,'new_sessions':276,
            'final_gate_opened':False,'accuracy_used_for_completion':False}
    except (OSError, ValueError, KeyError, TypeError):
        return {'complete':False,'cause':'continuation_receipts_missing_or_invalid','exit_status':status}


def supervise(command, launch, output, credentials_file=None):
    launch, output = Path(launch).absolute(), Path(output).absolute()
    if (launch==output or launch in output.parents or output in launch.parents
            or launch.exists() or launch.is_symlink() or output.exists() or output.is_symlink()):
        raise ValueError('Require separate fresh controller and continuation directories')
    launch.mkdir(parents=True)
    archive.dump(launch/'launch.json', {**archive.identity(os.getpid()),
        'at':datetime.now(timezone.utc).isoformat(),'controller_sha256':archive.sha(__file__),
        'archive_helper_sha256':archive.sha(archive.__file__),
        'cost_accounting_sha256':archive.sha(cost_accounting.__file__),
        'automatic_retry':False,'final_gate_opened':False})
    archive.dump(launch/'command.json', {'argv':command,'cwd':str(Path.cwd()),'output':str(output)})
    (launch/'controller-source.py').write_bytes(Path(__file__).read_bytes())
    (launch/'archive-helper-source.py').write_bytes(Path(archive.__file__).read_bytes())
    (launch/'cost-accounting-source.py').write_bytes(Path(cost_accounting.__file__).read_bytes())
    with (launch/'development.stdout').open('xb') as stdout, (launch/'development.stderr').open('xb') as stderr:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        except OSError as exc:
            archive.dump(launch/'spawn-error.json', {'type':type(exc).__name__}); status=127
        else:
            archive.dump(launch/'development-process.json', archive.identity(child.pid))
            status = child.wait()
    archive.dump(launch/'development-exit.json', {'exit_status':status})
    result = completion(output, status)
    if output.exists():
        try:
            costs = cost_accounting.summarize(output)
            if result['complete'] and (costs['deduplicated_total']['sessions'] != 312
                    or costs['stages']['retained_pilot']['sessions'] != 36
                    or costs['stages']['continuation']['sessions'] != 276):
                result.update(complete=False,cause='cost_session_count_mismatch')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            costs = {'complete':False,'cause':'cost_accounting_failed','type':type(exc).__name__}
            result.update(complete=False,cause='cost_accounting_failed')
        archive.dump(launch/'costs.json', costs)
    archive.dump(launch/'AUDITED.json', result)
    if not result['complete']:
        archive.dump(launch/'INCOMPLETE.json', result)
    forbidden = []
    if credentials_file is not None and (output/'accepted-launch.json').exists():
        for line in Path(credentials_file).read_text().splitlines():
            if line.startswith('ENGY_API_KEY='):
                value=line.split('=',1)[1].strip().strip('"').strip("'")
                if value: forbidden.append(value.encode())
        if not forbidden:
            raise ValueError('Credential scan unavailable; archive withheld')
    archived = archive.archive_evidence(launch, output, forbidden)
    archive.dump(launch/'FINISHED.json', {**result,**archived,
        'at':datetime.now(timezone.utc).isoformat(),'further_execution_launched':False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-directory', type=Path, required=True)
    fields = ('capsule','plan','preflight','task-source','runtime','pilot-root','pilot-launch',
              'continuation-preflight','output','credentials-file')
    for field in fields:
        parser.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    command = [sys.executable,'-m','benchmarks.ledger_optimization.continue_collection_096']
    for field in fields:
        command += ['--'+field,str(getattr(args,field.replace('-','_')).absolute())]
    result = supervise(command,args.launch_directory,args.output,args.credentials_file)
    print(json.dumps(result))
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
