"""Host stage primitives for the frozen 72+240-session recipe-plan experiment.

No CLI or autonomous dispatch. A paid caller must separately verify the frozen
plan, full host preflight, terminal predecessor and exclusive reservation before
calling execute_stage. The synthetic integration invokes the same stage paths.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from .audit_planning_107 import audit_session
from . import costs_guarded_093 as costs
from .m5_ml_terminal_prefix import archived_prefix, inventory, terminal

ARMS = ('plain', 'gnomon', 'ledger')
FIELDS = ('request', 'actual', 'origin', 'outcome_recorded_at', 'future_timestamps', 'series_id', 'round')
PREFIX = 6


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value):
    with Path(path).open('x') as stream: stream.write(json.dumps(value, indent=2)+'\n')


def source_identity():
    names = ('planning_host_107.py', 'probe_planning_host_107.py', 'audit_planning_107.py', 'contrast_audit_100.py',
             'contrast_view_100.py', 'paired_consistency_098.py', 'current_cv_pair_100.py',
             'current_history_contrast_100.py', 'continue_collection_096.py', 'continue_guarded_093.py',
             'launch_collection_096.py', 'control_collection_096.py', 'costs_guarded_093.py',
             'm5_ml_terminal_prefix.py', 'm5_ml_stage_checks.py', 'm5_ml_development_contract.py',
             'm5_ml_prefix_identity.py')
    return {n: sha(Path(__file__).with_name(n)) for n in names}


def validate_jobs(jobs):
    if len(jobs) != 4 or any(len(rows) != 26 for rows in jobs.values()):
        raise ValueError('Exact four-series, 26-origin development shape required')
    for series, rows in jobs.items():
        for n, job in enumerate(rows):
            if (set(job) != set(FIELDS) or job['series_id'] != series or type(job['round']) is not int
                    or job['round'] != n or len(job['actual']) != 14
                    or len(job['request']['history']) != 730 or len(job['future_timestamps']) != 14
                    or job['request']['future_timestamps'] != job['future_timestamps']):
                raise ValueError('Exact ordered development task identity required')
            origin = datetime.fromisoformat(job['origin'])
            if origin.tzinfo is None or job['request']['timestamps'][-1] != job['origin']:
                raise ValueError('Explicit forecast origin matching history endpoint required')
            if n and datetime.fromisoformat(rows[n-1]['outcome_recorded_at']) > origin:
                raise ValueError('Prior outcomes must mature before the next origin')


def stage_checks(root, jobs, stage, report):
    if stage not in ('pilot', 'complete'): raise ValueError('Unknown development stage')
    validate_jobs(jobs); root = Path(root)
    chosen = {s: rows[:PREFIX] if stage == 'pilot' else rows for s, rows in jobs.items()}
    expected = {(a, s, j['round']): j for a in ARMS for s, rows in chosen.items() for j in rows}
    if (report.get('complete') is not True or report.get('audit_failures') != []
            or report.get('shutdown_record_gaps') != [] or read(root/'host-jobs.json') != chosen):
        raise ValueError('Complete audited fixed stage required')
    raw = {}
    for path in root.rglob('grade.json'):
        g = read(path); key = g['arm'], g['series_id'], g['round']
        if (key not in expected or key in raw or path.relative_to(root).parts !=
                (g['arm'], g['series_id'], f"round-{g['round']}", 'grade.json')):
            raise ValueError('Extra, duplicate or displaced stage grade')
        raw[key] = g
    audited = {(r['arm'], r['series_id'], r['round']): r for r in report['rows']}
    if set(raw) != set(expected) or set(audited) != set(expected) or len(report['rows']) != len(expected):
        raise ValueError('Every intended stage session must remain in the denominator')
    for key, g in raw.items():
        task = expected[key]
        if g['origin'] != task['origin'] or any(type(g.get(k)) is not bool for k in ('valid', 'workflow_complete', 'fallback_used')):
            raise ValueError('Grade identity or completion types differ')
        if (len(g['point']) != 14 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in g['point'])
                or any(audited[key].get(k) != v for k, v in g.items())):
            raise ValueError('Invalid forecast or grade/report disagreement')
        score = math.sqrt(sum((math.log1p(p)-math.log1p(a))**2 for p, a in zip(g['point'], task['actual'], strict=True))/14)
        if not math.isfinite(g['rmsle']) or abs(score-g['rmsle']) > 1e-12:
            raise ValueError('Stage score disagrees with retained forecast and fixed actuals')
    quality = True
    for arm in ARMS:
        rows = [g for k, g in raw.items() if k[0] == arm]
        counts = {'tasks': len(rows), 'valid': sum(r['valid'] for r in rows),
                  'workflow_complete': sum(r['workflow_complete'] for r in rows)}
        if any(report['arms'][arm][k] != v for k, v in counts.items()):
            raise ValueError('Per-arm counts disagree')
        if stage == 'pilot': quality &= counts['valid'] == 24 and counts['workflow_complete'] >= 22
    return {'sessions': len(raw), 'passed': True, 'pilot_quality_passed': quality if stage == 'pilot' else None,
            'accuracy_used_for_gate': False, 'scores_recomputed': len(raw)}


def accounting(root):
    result = costs.summarize(root)
    for row in result['sessions']: row['stage'] = 'retained_pilot' if row['round'] < PREFIX else 'continuation'
    result['stages'] = {s: {key: sum(r[key] for r in result['sessions'] if r['stage'] == s) for key in costs.COUNTS}
                        for s in ('retained_pilot', 'continuation')}
    if any(result['deduplicated_total'][key] != sum(v[key] for v in result['stages'].values()) for key in costs.COUNTS):
        raise ValueError('Cost stages do not reconcile')
    return result


def audit_recipes(root, output):
    output = Path(output); output.mkdir(exist_ok=False)
    rows = [{'session': str(p.parent.relative_to(root)), **audit_session(p.parent)}
            for p in sorted(Path(root).glob('*/*/round-*/grade.json'))]
    dump(output/'sessions.json', rows)
    summary = {'passed': True, 'sessions': len(rows), 'checks': sum(r['checks'] for r in rows),
               'annotations': sum(r['annotations'] for r in rows),
               'actionable_next_calls': sum(r['actionable_next_calls'] for r in rows),
               'engy_calls': 0, 'provider_calls': 0}
    dump(output/'report.json', summary); return summary


def verify_pilot(pilot, launch, jobs, plan, plan_sha, *, proc=Path('/proc')):
    pilot, launch = Path(pilot), Path(launch)
    # A copied process identity from another boot is not a terminal observation.
    for name in ('launch.json', 'pilot-process.json'):
        identity = read(launch/name)
        if identity['boot_id'] != (proc/'sys/kernel/random/boot_id').read_text().strip():
            raise ValueError('Pilot must be verified on its original host boot')
        terminal(identity, proc)
    if (launch/'INCOMPLETE.json').exists() or (pilot/'STAGE_INCOMPLETE.json').exists():
        raise ValueError('Incomplete pilot requires separate reconciliation')
    for name in ('FINISHED.json', 'AUDITED.json'):
        value = read(launch/name)
        if value.get('complete') is not True or value.get('continuation_gate_passed') is not True:
            raise ValueError('Pilot terminal gate did not pass')
    if (read(launch/'pilot-exit.json') != {'exit_status': 0}
            or read(pilot/'runner-exit.json') != {'exit_status': 0, 'continuation_launched': False}
            or read(pilot/'GATE.json') != {'passed': True, 'accuracy_used_for_gate': False}):
        raise ValueError('Pilot completion receipts disagree')
    manifest = read(pilot/'manifest.json')
    if (manifest['sources'] != plan['capsule']['sources'] or manifest['inventory'] != plan['runtime_inventory']
            or manifest['build'] != plan['build'] or manifest['source_jobs_sha256'] != plan['task_source_sha256']
            or manifest.get('requested_seed') != 7 or manifest['planned'] != 72
            or read(pilot/'final-runtime-inventory.json') != plan['runtime_inventory']
            or sha(pilot/'prospective-plan.json') != plan_sha
            or read(pilot/'accepted-launch.json')['plan_sha256'] != plan_sha):
        raise ValueError('Pilot plan, source or runtime changed')
    checked = stage_checks(pilot, jobs, 'pilot', read(pilot/'report.json'))
    if not checked['pilot_quality_passed']: raise ValueError('Pilot quality threshold failed')
    recipe = read(pilot/'recipe-audit/report.json')
    if recipe.get('passed') is not True or recipe.get('sessions') != 72: raise ValueError('Full pilot recipe audit required')
    return {'files': archived_prefix(pilot, launch), 'manifest': manifest}


def execute_stage(helper, plan_path, jobs, output, *, stage, credential,
                  pilot=None, pilot_launch=None):
    """Execute one admitted stage; no retries or automatic continuation.

    credential is invoked only after source/runtime and (for continuation)
    terminal archive and copied-prefix re-audits. This primitive is not itself
    a public paid admission interface.
    """
    plan_path = Path(plan_path); plan = read(plan_path); output = Path(output).absolute()
    run = helper.run; sources = source_identity(); validate_jobs(jobs)
    if (stage not in ('pilot', 'complete') or output.exists() or output.is_symlink()
            or plan['host_sources'] != sources or tuple(plan['arms']) != ARMS
            or plan['planned'] != {'pilot_sessions': 72, 'continuation_sessions': 240, 'total_sessions': 312}
            or plan['requested_seed'] != 7 or plan['final_gate_opened'] is not False):
        raise ValueError('Fresh output and exact prospective host contract required')
    jobs_hash = hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if jobs_hash != plan['jobs_sha256'] or plan['task_source_sha256'] != run.SOURCE_SHA:
        raise ValueError('Job values differ from the authenticated prospective source')
    frozen = {p.name: sha(p) for p in run.HERE.iterdir() if p.is_file()}
    runtime = run.runtime_inventory()
    runtimes = {a: run.OTHER/('plain-venv' if a == 'plain' else 'gnomon-venv')/'bin/python' for a in ARMS}
    build = json.loads(subprocess.check_output([str(runtimes['gnomon']), '-I', '-c',
        'import json;from gnomon.build_info import build_info;print(json.dumps(build_info()))'], text=True))
    if (frozen != plan['capsule']['sources'] or runtime != plan['runtime_inventory'] or build != plan['build']
            or build['package_version'] != '1.2.0' or build['source_sha256'] != run.BUILD_SHA):
        raise ValueError('Actual source/runtime/build differs from plan')
    prefix = None; selected_jobs = {s: rows[:PREFIX] if stage == 'pilot' else rows for s, rows in jobs.items()}
    if stage == 'complete':
        pilot, pilot_launch = Path(pilot).resolve(), Path(pilot_launch).resolve()
        if any(a == output or a in output.parents or output in a.parents for a in (pilot, pilot_launch)):
            raise ValueError('Continuation must be outside original evidence')
        prefix = verify_pilot(pilot, pilot_launch, jobs, plan, sha(plan_path))
        metadata = helper.copy_prefix(pilot, output, prefix['files'])
        for name in ('manifest.json', 'host-jobs.json'): shutil.copyfile(metadata/name, output/name)
        recheck_output = output/'prefix-workflow-audit'; recheck_output.mkdir()
        rechecked = helper.analyze(output, output=recheck_output)
        if not stage_checks(output, jobs, 'pilot', rechecked)['pilot_quality_passed']:
            raise ValueError('Copied pilot did not pass')
        # Preserve the copied original audit; write this new audit separately.
        audit_recipes(output, output/'prefix-recipe-audit')
        for name in ('manifest.json', 'host-jobs.json'): (output/name).unlink()
        (output/'recipe-audit').rename(output/'retained-pilot-recipe-audit')
    else:
        output.mkdir(parents=True)
        shutil.copytree(run.HERE, output/'frozen-source', ignore=shutil.ignore_patterns('__pycache__'))
    total = 72 if stage == 'pilot' else 312; new = 72 if stage == 'pilot' else 240
    manifest = {'planned': total, 'pilot': stage == 'pilot', 'build': build, 'inventory': runtime,
                'sources': frozen, 'model': run.MODEL, 'requested_seed': 7,
                'runtimes': {a: str(p) for a, p in runtimes.items()}, 'source_jobs_sha256': plan['task_source_sha256'],
                'pilot_sessions_retained': 0 if stage == 'pilot' else 72, 'new_sessions_planned': new,
                'accuracy_used_for_promotion': False, 'host_sources': sources}
    dump(output/'manifest.json', manifest); dump(output/'host-jobs.json', selected_jobs)
    (output/'prospective-plan.json').write_bytes(plan_path.read_bytes())
    dump(output/'accepted-launch.json', {'plan_sha256': sha(plan_path), 'host_sources': sources,
         'stage': stage, 'retained_sessions': 0 if stage == 'pilot' else 72, 'new_sessions': new, 'final_gate_opened': False})
    rows = []
    try:
        api_key = credential()
        # This is an explicit host-only prefix parameter; worker sources stay unchanged.
        helper.PREFIX = PREFIX
        chain = run.chain if stage == 'pilot' else helper.continue_chain
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(chain, i, s, selected_jobs[s], output, runtimes, api_key) for i, s in enumerate(sorted(jobs))]
            for future in as_completed(futures):
                try: rows += future.result()
                except BaseException:
                    run.STOP.set(); raise
        if len(rows) != new: raise ValueError('Missing or duplicated newly executed sessions')
        final_runtime = run.runtime_inventory(); dump(output/'final-runtime-inventory.json', final_runtime)
        if (runtime != final_runtime or frozen != {p.name: sha(p) for p in run.HERE.iterdir() if p.is_file()}
                or sources != source_identity() or (prefix and inventory(pilot) != prefix['files'])):
            raise ValueError('Runtime, sources or original pilot changed')
        report = helper.analyze(output)
        checked = stage_checks(output, jobs, stage, report)
        recipe = audit_recipes(output, output/'recipe-audit')
        if recipe['sessions'] != total: raise ValueError('Incomplete recipe audit')
        billed = accounting(output)
        if (billed['deduplicated_total']['sessions'] != total
                or billed['stages']['retained_pilot']['sessions'] != 72
                or billed['stages']['continuation']['sessions'] != (0 if stage == 'pilot' else 240)):
            raise ValueError('Cost session coverage mismatch')
        dump(output/'costs.json', billed)
        dump(output/'complete.json', {'completed': total, 'planned': total, 'new_sessions': new,
             'retained_pilot_sessions': 0 if stage == 'pilot' else 72, 'source_unchanged': True})
        # The shared archive supervisor reads the same completion contract for both stages.
        dump(output/'GATE.json', {'passed': checked['pilot_quality_passed'] if stage == 'pilot' else True,
                                'accuracy_used_for_gate': False})
        dump(output/'runner-exit.json', {'exit_status': 0, 'continuation_launched': False})
        return checked
    except BaseException as exc:
        run.STOP.set()
        dump(output/'STAGE_INCOMPLETE.json', {'type': type(exc).__name__, 'automatic_retry': False,
             'at': datetime.now(timezone.utc).isoformat(), 'final_gate_opened': False})
        dump(output/'runner-exit.json', {'exit_status': 1, 'continuation_launched': False})
        raise
