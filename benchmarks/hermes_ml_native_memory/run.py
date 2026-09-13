"""Only the native-memory Hermes control; reuse frozen checkpoint-v4 execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from benchmarks.hermes_ml_checkpoint_v4 import run as base
from benchmarks.hermes_ml_checkpoint_v4.pipeline import archive
from benchmarks.hermes_ml_checkpoint_v4.transport import dump, sha, MODEL

HERE = Path(__file__).resolve().parent
ORIGINAL_PREPARE = base.prepare


def prepare(work, job, prior, arm):
    assert arm == 'plain', 'This follow-up must never execute another arm'
    ORIGINAL_PREPARE(work, job, prior, arm)
    with (work / 'TASK.md').open('a') as f:
        f.write('\n' + (HERE / 'MEMORY_WORKFLOW.md').read_text())


def native_state(home):
    state = {}
    for directory in ('memories', 'skills'):
        for path in (home / directory).rglob('*'):
            if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                continue
            data = path.read_bytes()
            if len(data) > 4 * 1024 * 1024:
                raise ValueError('Native state exceeds the declared 4 MiB per-file evidence limit')
            import base64
            state[str(path.relative_to(home))] = {
                'sha256': sha(path), 'bytes': len(data),
                'base64': base64.b64encode(data).decode('ascii'),
            }
    return state


def chain(series, jobs, root, python, api_key):
    prior = []
    for job in jobs:
        if base.STOP.is_set():
            raise RuntimeError('Service admission stopped new sessions')
        home = root / 'plain' / series / 'home'
        before = native_state(home)
        base.execute(root, 'plain', series, job, prior, python, api_key)
        out = root / 'plain' / series / f'round-{job["round"]}'
        dump(out / 'native-state-before.json', before)
        dump(out / 'native-state-after.json', native_state(home))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--comparison-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = args.comparison_root.resolve()
    try:
        passed = json.loads(args.preflight.read_text())
        assert passed['passed']
        own = {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
        common = {p.name: sha(p) for p in base.HERE.iterdir() if p.is_file()}
        assert passed['tested_sources'] == own and passed['common_sources'] == common
        assert MODEL == 'deepseek-v4.1-flash' and sha(base.TASK_SOURCE) == base.SOURCE_SHA
        dump(root / 'accepted-preflight.json', {'path': str(args.preflight), 'sha256': sha(args.preflight)})
        shutil.copytree(HERE, root / 'followup-source', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(base.HERE, root / 'frozen-source', ignore=shutil.ignore_patterns('__pycache__'))
        # Waiting is outside every task and makes no model calls.
        while not (reference / 'FINISHED.json').exists():
            if (reference / 'BLOCKED.json').exists() or (reference / 'INCOMPLETE.json').exists():
                raise RuntimeError('Reference run terminated incomplete; follow-up has not started')
            dump(root / 'queue-status.json', {'state': 'waiting_for_existing_run',
                 'reference': str(reference), 'at': datetime.now(timezone.utc).isoformat(), 'api_calls': 0})
            time.sleep(30)
        receipt = json.loads((reference / 'FINISHED.json').read_text())
        assert receipt['completed'] and receipt['evaluation_run']
        originals = {n: sha(reference / n) for n in ('FINISHED.json', 'evaluation/report.json', 'SHA256SUMS.json')}
        dump(root / 'comparison-evidence-hashes.json', originals)
        raw = json.loads(base.TASK_SOURCE.read_text())
        jobs = {s: [{k: j[k] for k in ('request', 'actual', 'origin', 'outcome_recorded_at',
                   'future_timestamps', 'series_id', 'round')} for j in values] for s, values in raw.items()}
        assert len(jobs) == 4 and all(len(values) == 26 for values in jobs.values())
        inventory = base.runtime_inventory()
        assert inventory['plain']['gnomon'] is False
        python = base.OTHER / 'plain-venv/bin/python'
        dump(root / 'manifest.json', {'planned': 104, 'arms': ['plain'], 'model': MODEL,
             'sources': common, 'followup_sources': own, 'source_jobs_sha256': base.SOURCE_SHA,
             'runtimes': {'plain': str(python)}, 'inventory': inventory,
             'comparison_root': str(reference), 'followup': 'explicit_native_memory',
             'comparison_is_contemporaneous': False})
        dump(root / 'host-jobs.json', jobs)
        dump(root / 'queue-status.json', {'state': 'running', 'at': datetime.now(timezone.utc).isoformat()})
        base.prepare = prepare
        credential = base.key()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(chain, s, jobs[s], root, python, credential) for s in sorted(jobs)]
            for f in as_completed(futures):
                f.result()
        assert common == {p.name: sha(p) for p in base.HERE.iterdir() if p.is_file()}
        assert own == {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
        final_inventory = base.runtime_inventory()
        dump(root / 'final-runtime-inventory.json', final_inventory)
        assert inventory == final_inventory
        from .analyze import analyze
        report = analyze(root, reference)
        assert report['complete'] and not report['audit_failures']
        assert all(sha(reference / n) == h for n, h in originals.items())
        dump(root / 'complete.json', {'completed': 104, 'planned': 104, 'reference_unchanged': True})
        archive(root)
        dump(root / 'FINISHED.json', {'completed': True, 'archive_sha256': sha(root / 'evidence.tar.gz'),
             'at': datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        dump(root / 'BLOCKED.json', {'error': type(exc).__name__, 'message': str(exc), 'automatic_retry': False})
        archive(root)
        dump(root / 'INCOMPLETE.json', {'completed': False, 'automatic_retry': False,
             'archive_sha256': sha(root / 'evidence.tar.gz')})
        raise


if __name__ == '__main__':
    main()
