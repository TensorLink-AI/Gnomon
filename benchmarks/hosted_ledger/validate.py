"""Run with python -m benchmarks.hosted_ledger.validate OUTPUT_DIRECTORY.

Exercises the existing core across spawned processes, without an MCP server,
credentials, network calls or production data. Fixed clocks are synthetic test
fixtures, not a proposed hosted API. Timing results are observations, not an SLA.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sqlite3
import statistics
import time

from gnomon import GnomonSession, TemporalLedger
from gnomon.forecast_adapter import ForecastAdapterError
from gnomon.ids import FixedClock


ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
CUTOFF = (ORIGIN + timedelta(days=3)).isoformat()


def clock(days=0):
    return FixedClock(ORIGIN + timedelta(days=days))


def forecast(path):
    ledger = TemporalLedger(path, clock=clock())
    with GnomonSession.from_config(ledger=ledger) as session:
        execution = session.forecast('last_value', {
            'history': [10., 11., 12.], 'horizon': 2, 'series_id': 'sales', 'unit': 'widgets',
            'timestamps': [(ORIGIN - timedelta(days=d)).isoformat() for d in (2, 1, 0)],
            'future_timestamps': [(ORIGIN + timedelta(days=d)).isoformat() for d in (1, 2)],
            'cutoff': ORIGIN.isoformat(),
        })
    decision = ledger.record_decision_summary(execution_id=execution['execution_id'],
        rationale='Synthetic last-value baseline.', assumptions=['Level persists.'],
        invalidation_conditions=['Level changes.'], context=[])
    return {'execution_id': execution['execution_id'], 'decision_id': decision['decision_id']}


def outcomes(path):
    ledger = TemporalLedger(path, clock=clock(3), create=False)
    return ledger.append_actual(actuals=[{
        'series_id': 'sales', 'unit': 'widgets', 'valid_time': (ORIGIN + timedelta(days=d)).isoformat(),
        'value': value, 'source_available_at': CUTOFF, 'source_ref': 'synthetic-fixture',
    } for d, value in ((1, 13.), (2, 14.))])


def review(path, ids):
    ledger = TemporalLedger(path, clock=clock(3), create=False)
    execution = ledger.execution(ids['execution_id'])
    packet = ledger.review_decision(decision_id=ids['decision_id'],
                                   source_as_of=CUTOFF, recorded_as_of=CUTOFF)
    assert execution['result']['point'] == [12., 12.]
    return packet


def lesson(path, ids):
    ledger = TemporalLedger(path, clock=clock(3), create=False)
    saved = ledger.record_lesson(decision_id=ids['decision_id'],
        lesson='Synthetic baseline underpredicted both observations; cause unverified.',
        source_as_of=CUTOFF, recorded_as_of=CUTOFF)
    return ledger.export_lesson(lesson_id=saved['lesson_id'], recorded_as_of=CUTOFF)


def reject_premature_lesson(path, ids):
    try:
        lesson(path, ids)
    except ForecastAdapterError as error:
        assert 'complete matching-unit actuals' in str(error)
        return True
    raise AssertionError('a lesson was accepted before any outcomes arrived')


def revise(path):
    ledger = TemporalLedger(path, clock=clock(4), create=False)
    return ledger.append_actual(series_id='sales', unit='widgets',
        valid_time=(ORIGIN + timedelta(days=2)).isoformat(), value=20.,
        source_available_at=(ORIGIN + timedelta(days=4)).isoformat(), source_ref='synthetic-revision')


def verify_restore(path, ids, saved):
    ledger = TemporalLedger(path, clock=clock(5), create=False)
    old = ledger.review_decision(decision_id=ids['decision_id'], source_as_of=CUTOFF, recorded_as_of=CUTOFF)
    current = ledger.review_decision(decision_id=ids['decision_id'],
        source_as_of=(ORIGIN + timedelta(days=5)).isoformat(),
        recorded_as_of=(ORIGIN + timedelta(days=5)).isoformat())
    exported = ledger.export_lesson(lesson_id=saved['lesson_id'], recorded_as_of=CUTOFF)
    assert old['metrics']['mae'] == 1.5 and current['metrics']['mae'] == 4.5
    assert exported == saved
    return {'old_mae': old['metrics']['mae'], 'revised_mae': current['metrics']['mae'],
            'saved_lesson_unchanged': exported == saved}


def fresh(function, *args):
    """A distinct interpreter for each step: no shared caches or live objects."""
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn')) as pool:
        return pool.submit(function, *args).result(timeout=45)


def writer(path, worker, count):
    ledger = TemporalLedger(path, create=False)
    durations, identities = [], []
    for i in range(count):
        started = time.perf_counter()
        aid = ledger.append_actual(series_id='contended', unit='widgets',
            valid_time=ORIGIN.isoformat(), value=float(worker * count + i),
            source_available_at=CUTOFF, source_ref=f'worker-{worker}')
        # An exact retry must reuse the existing observation, even with competing writers.
        assert aid == ledger.append_actual(series_id='contended', unit='widgets',
            valid_time=ORIGIN.isoformat(), value=float(worker * count + i),
            source_available_at=CUTOFF, source_ref=f'worker-{worker}')
        durations.append((time.perf_counter() - started) * 1000)
        identities.append(aid)
    return {'milliseconds_per_write_and_retry': durations, 'ids': identities}


def snapshot_reader(path, ids, ready, resume, result):
    class PausingLedger(TemporalLedger):
        def _pairs(self, conn, req, source_as_of, recorded_as_of):
            pairs = super()._pairs(conn, req, source_as_of, recorded_as_of)
            ready.set()
            if not resume.wait(15):
                raise RuntimeError('snapshot fixture writer did not finish')
            again = super()._pairs(conn, req, source_as_of, recorded_as_of)
            assert again == pairs, 'one review observed two different actual revisions'
            return again
    try:
        ledger = PausingLedger(path, clock=clock(5), create=False)
        result.put(ledger.review_decision(decision_id=ids['decision_id'],
            source_as_of=(ORIGIN + timedelta(days=5)).isoformat(),
            recorded_as_of=(ORIGIN + timedelta(days=5)).isoformat()))
    except Exception as error:
        result.put({'fixture_error': repr(error)})


def snapshot_probe(path, ids):
    """WAL permits a writer commit while a review's read transaction stays open."""
    with sqlite3.connect(path) as conn:
        assert conn.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
    context = mp.get_context('spawn')
    ready, resume, result = context.Event(), context.Event(), context.Queue()
    reader = context.Process(target=snapshot_reader, args=(path, ids, ready, resume, result))
    reader.start()
    try:
        if not ready.wait(15):
            raise RuntimeError('snapshot reader did not reach barrier')
        # This commits before the paused review resumes its remaining reads.
        revise(path)
        resume.set()
        packet = result.get(timeout=15)
        assert 'fixture_error' not in packet, packet
        assert packet['metrics']['mae'] == 1.5
        assert [p['actual'] for p in packet['scored_pairs']] == [13., 14.]
        assert packet['coverage']['matched_steps'] == 2
        return {'writer_committed_during_review': True, 'review_mae': packet['metrics']['mae']}
    finally:
        resume.set()
        reader.join(5)
        if reader.is_alive():
            reader.terminate()
            reader.join()
        result.close()


def backup(source, destination):
    """SQLite online backup includes committed WAL content; no raw .db file copying."""
    if destination.exists():
        raise ValueError('backup destination already exists')
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
        assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert dst.execute('PRAGMA foreign_key_check').fetchall() == []


def run(directory, *, workers=4, writes=25, journal_mode='DELETE'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / 'ledger.db'
    ids = fresh(forecast, path)
    pending = fresh(review, path, ids)
    assert pending['scoring_status'] == 'pending'
    assert fresh(reject_premature_lesson, path, ids)
    fresh(outcomes, path)
    reviewed = fresh(review, path, ids)
    assert reviewed['metrics']['mae'] == 1.5
    saved = fresh(lesson, path, ids)
    with sqlite3.connect(path) as conn:
        assert conn.execute(f'PRAGMA journal_mode={journal_mode}').fetchone()[0].upper() == journal_mode
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(writer, path, i, writes) for i in range(workers)]
        results = [f.result(timeout=90) for f in futures]
    elapsed = time.perf_counter() - started
    durations = sorted(t for r in results for t in r['milliseconds_per_write_and_retry'])
    all_ids = [aid for r in results for aid in r['ids']]
    with sqlite3.connect(path) as conn:
        revisions = conn.execute("SELECT revision FROM actuals WHERE series_id='contended' ORDER BY revision").fetchall()
        assert [r[0] for r in revisions] == list(range(workers * writes))
        assert len(set(all_ids)) == workers * writes
        for sql in ("UPDATE actuals SET value=0", "DELETE FROM actuals"):
            try:
                conn.execute(sql)
            except sqlite3.IntegrityError as error:
                assert 'append-only' in str(error)
            else:
                raise AssertionError('append-only enforcement missing')
        conn.rollback()
    snapshot = snapshot_probe(path, ids)  # Separately labelled WAL snapshot test.
    restored = directory / 'restored' / 'ledger.db'
    restored.parent.mkdir()
    started = time.perf_counter()
    backup(path, restored)
    restored_result = fresh(verify_restore, restored, ids, saved)
    recovery_seconds = time.perf_counter() - started
    manifest = {'kind': 'hosted-ledger-validation/1', 'synthetic': True,
        'resources': ids, 'lesson_id': saved['lesson_id'], 'artifact_count': 0,
        'backup_sha256': hashlib.sha256(restored.read_bytes()).hexdigest()}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'kind': 'hosted-ledger-validation/1', 'status': 'passed',
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment': {'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version,
                        'platform': platform.platform()},
        'scope': 'Core SQLite only; no hosted server, authorization, MCP, Ditto or provider network calls.',
        'synthetic_clock': True, 'concurrency': {'journal_mode': journal_mode, 'workers': workers,
            'distinct_writes': workers * writes, 'exact_retries': workers * writes,
            'elapsed_seconds_including_process_startup': elapsed,
            'median_ms_write_and_retry': statistics.median(durations),
            'p95_ms_write_and_retry': durations[max(0, (95 * len(durations) + 99) // 100 - 1)],
            'max_ms_write_and_retry': max(durations), 'revision_sequence_complete': True},
        'handoff': {'pending_then_complete': True, 'premature_lesson_rejected': True,
                    'mae': reviewed['metrics']['mae']},
        'snapshot': {'journal_mode': 'WAL', **snapshot},
        'restore': {**restored_result, 'elapsed_seconds_including_verification_process': recovery_seconds,
                    'database_bytes': restored.stat().st_size},
        'limitations': ['Tiny synthetic ledger, not a capacity certification.',
            'Process reopening is tested; no actual hosted server exists in this probe.',
            'Backup contains no external artifacts or credentials.',
            'No kill-during-commit or provider/response fault injection.']}
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 17))
    parser.add_argument('--writes', type=int, default=25, choices=range(1, 10001))
    parser.add_argument('--journal-mode', choices=['DELETE', 'WAL'], default='DELETE')
    args = parser.parse_args()
    print(json.dumps(run(args.output, workers=args.workers, writes=args.writes,
                         journal_mode=args.journal_mode), indent=2))


if __name__ == '__main__':
    main()
