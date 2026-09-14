"""Host-only continuation of an independently accepted, immutable 093 pilot.

Never modifies the frozen worker or the original pilot. No credentials are read
until gate, source, copied-state, runtime and synthetic-preflight checks pass.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

from benchmarks.hermes_ml_checkpoint_v6 import run
from benchmarks.hermes_ml_checkpoint_v6.analyze import analyze

FIELDS = ('request', 'actual', 'origin', 'outcome_recorded_at',
          'future_timestamps', 'series_id', 'round')
PREFIX = 3


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(root):
    """Reject links/special files rather than copying anything outside this run."""
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
                raise ValueError('Non-regular pilot state: ' + str(path.relative_to(root)))
            if stat.S_ISREG(mode):
                result[str(path.relative_to(root))] = digest(path)
    return result


def require_terminal(identity, proc=Path('/proc')):
    path = proc / str(identity['pid']) / 'stat'
    if not path.exists():
        return
    boot = (proc / 'sys/kernel/random/boot_id').read_text().strip()
    fields = path.read_text().split(') ')[1].split()
    if (boot == identity['boot_id'] and fields[19] == str(identity['start_ticks'])
            and fields[0] != 'Z'):
        raise ValueError('Original pilot process is still live')


def jobs_from_source(source):
    if digest(source) != run.SOURCE_SHA:
        raise ValueError('Development task source hash changed')
    raw = read(source)
    jobs = {s: [{k: j[k] for k in FIELDS} for j in raw[s]] for s in sorted(raw)}
    if len(jobs) != 4 or any(len(v) != 26 for v in jobs.values()):
        raise ValueError('Require the frozen four-series, 26-origin development cohort')
    for series, values in jobs.items():
        for number, job in enumerate(values):
            if job['series_id'] != series or job['round'] != number:
                raise ValueError('Noncanonical series/origin ordering')
    return jobs


def verify_gate(pilot, launch, jobs):
    if (launch / 'INCOMPLETE.json').exists():
        raise ValueError('Pilot controller recorded an incomplete run')
    gate, finished = read(launch / 'GATE.json'), read(launch / 'FINISHED.json')
    if gate.get('passed') is not True or finished.get('pilot_exit_status') != 0:
        raise ValueError('Pilot completion gate did not pass')
    if read(launch / 'pilot-exit.json')['exit_status'] != 0:
        raise ValueError('Pilot process failed')
    identity = read(launch / 'launch.json')
    require_terminal(identity)
    child = {**read(launch / 'pilot-process.json'), 'boot_id': identity['boot_id']}
    require_terminal(child)
    sources = {p.name: digest(p) for p in run.HERE.iterdir() if p.is_file()}
    if sources != identity['source_hashes']:
        raise ValueError('Frozen 093 source mismatch')
    manifest, report = read(pilot / 'manifest.json'), read(pilot / 'report.json')
    if manifest['sources'] != sources or manifest['source_jobs_sha256'] != run.SOURCE_SHA:
        raise ValueError('Pilot manifest/source mismatch')
    if (manifest['planned'] != 36 or report.get('complete') is not True
            or report.get('audit_failures') != []):
        raise ValueError('Complete independent pilot audit required')
    if read(pilot / 'host-jobs.json') != {s: v[:PREFIX] for s, v in jobs.items()}:
        raise ValueError('Pilot does not match the exact development prefix')
    expected = {(a, s, n) for a in run.ARMS for s in jobs for n in range(PREFIX)}
    grades = [read(p) for p in pilot.glob('*/*/round-*/grade.json')]
    keys = {(g['arm'], g['series_id'], g['round']) for g in grades}
    if keys != expected or len(grades) != len(expected):
        raise ValueError('Missing, extra or duplicate pilot sessions')
    for arm in run.ARMS:
        values = [g for g in grades if g['arm'] == arm]
        full = sum(g['workflow_complete'] is True for g in values)
        if len(values) != 12 or full < 11:
            raise ValueError('Pilot full-workflow threshold failed: ' + arm)
        if report['arms'][arm]['workflow_complete'] != full:
            raise ValueError('Pilot report disagrees with retained grades')
    if digest(launch / 'evidence.tar.gz') != finished['archive_sha256']:
        raise ValueError('Pilot archive hash mismatch')
    files = inventory(pilot)
    if files != read(launch / 'SHA256SUMS.json'):
        raise ValueError('Pilot state differs from its terminal inventory')
    return files, sources, manifest


def copy_prefix(pilot, output, expected):
    if output.exists() or output == pilot or pilot in output.parents:
        raise ValueError('Require a fresh continuation root outside the original pilot')
    shutil.copytree(pilot, output, symlinks=False)
    if inventory(output) != expected or inventory(pilot) != expected:
        raise ValueError('Pilot copy/source identity changed')
    metadata = output / 'pilot-metadata'
    metadata.mkdir()
    # Keep historical completion markers, but never present them as completion
    # of the longer continuation. Session files remain at their original paths.
    for path in list(output.iterdir()):
        if path.is_file():
            path.rename(metadata / path.name)
    run.dump(output / 'pilot-prefix-inventory.json', expected)
    return metadata


def prior_from_prefix(root, arm, series, jobs):
    prior = []
    for job in jobs[:PREFIX]:
        grade = read(root / arm / series / f'round-{job["round"]}' / 'grade.json')
        if (grade['origin'], grade['series_id'], grade['arm'], grade['round']) != (
                job['origin'], series, arm, job['round']):
            raise ValueError('Pilot grade/task identity mismatch')
        prior.append({'series_id':series, 'unit':job['request']['unit'],
            'origin':job['origin'], 'future_timestamps':job['future_timestamps'],
            'outcome_recorded_at':job['outcome_recorded_at'], 'point':grade['point'],
            'actual':job['actual'], 'execution_id':grade['execution_id'],
            'config':grade['config'], 'fallback_used':grade['fallback_used'],
            'rmsle':grade['rmsle']})
    return prior


def continue_chain(index, series, jobs, root, runtimes, api_key):
    prior = {a:prior_from_prefix(root, a, series, jobs) for a in run.ARMS}
    result = []
    for job in jobs[PREFIX:]:
        offset = (index + job['round']) % len(run.ARMS)
        for arm in run.ARMS[offset:] + run.ARMS[:offset]:
            if run.STOP.is_set():
                raise RuntimeError('New sessions stopped after admission failure')
            now = datetime.fromisoformat(job['origin'])
            if any(datetime.fromisoformat(p['outcome_recorded_at']) > now for p in prior[arm]):
                raise ValueError('Continuation would expose an immature outcome')
            result.append(run.execute(root, arm, series, job, prior[arm], runtimes[arm], api_key))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--pilot-launch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--continuation-preflight', type=Path, required=True)
    args = parser.parse_args()
    pilot, launch, output = (p.resolve() for p in (args.pilot_root, args.pilot_launch, args.output))
    jobs = jobs_from_source(run.TASK_SOURCE)
    files, sources, manifest = verify_gate(pilot, launch, jobs)
    preflight = read(args.continuation_preflight)
    if (preflight.get('passed') is not True or preflight.get('engy_calls') != 0
            or preflight.get('continuation_source_sha256') != digest(Path(__file__))
            or preflight.get('frozen_sources') != sources
            or set(preflight.get('resumed_arms', [])) != set(run.ARMS)):
        raise ValueError('Exact-source synthetic continuation preflight required')
    runtime = run.runtime_inventory()
    if runtime != manifest['inventory'] or runtime != read(pilot / 'final-runtime-inventory.json'):
        raise ValueError('Runtime inventory differs from completed pilot')
    metadata = copy_prefix(pilot, output, files)
    # Re-run the frozen independent audit on the copied prefix. Original remains
    # immutable, and no credential or forecast request is needed for this audit.
    shutil.copyfile(metadata / 'manifest.json', output / 'manifest.json')
    shutil.copyfile(metadata / 'host-jobs.json', output / 'host-jobs.json')
    prefix_report = analyze(output)
    if not prefix_report['complete'] or prefix_report['audit_failures']:
        raise ValueError('Independent copied-prefix audit failed')
    for name in ('report.json', 'RESULTS.md'):
        (output / name).rename(metadata / ('rechecked-' + name))
    run.dump(output / 'manifest.json', {**manifest, 'pilot':False, 'planned':312,
        'pilot_sessions_retained':36, 'new_sessions_planned':276,
        'continuation_source_sha256':digest(Path(__file__)),
        'continuation_preflight_sha256':digest(args.continuation_preflight),
        'original_pilot':str(pilot), 'accuracy_used_for_promotion':False})
    run.dump(output / 'host-jobs.json', jobs)
    runtimes = {a:Path(p) for a,p in manifest['runtimes'].items()}
    results = []
    api_key = run.key()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(continue_chain, i, s, jobs[s], output, runtimes, api_key)
                       for i,s in enumerate(sorted(jobs))]
            for future in as_completed(futures):
                try:
                    results += future.result()
                except BaseException:
                    run.STOP.set()
                    raise
        if len(results) != 276:
            raise ValueError('Continuation session count mismatch')
        if {p.name:digest(p) for p in run.HERE.iterdir() if p.is_file()} != sources:
            raise ValueError('Frozen sources changed during continuation')
        final_runtime = run.runtime_inventory()
        run.dump(output / 'final-runtime-inventory.json', final_runtime)
        if runtime != final_runtime or inventory(pilot) != files:
            raise ValueError('Original pilot or runtime changed during continuation')
        run.dump(output / 'complete.json', {'completed':312, 'planned':312,
            'new_sessions':276, 'retained_pilot_sessions':36, 'source_unchanged':True})
    except BaseException:
        run.STOP.set()
        run.dump(output / 'CONTINUATION_INCOMPLETE.json', {
            'at':datetime.now(timezone.utc).isoformat(), 'automatic_retry':False,
            'scope':'Inspect retained session and host logs; no selective rerun.'})
        raise


if __name__ == '__main__':
    main()
