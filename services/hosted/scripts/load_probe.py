"""Bounded real-transport concurrency probe; reports measurements, not an SLA."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from hosted_probe_support import serving, call
from gnomon_hosted.storage import Store


def client(spec):
    url, token, index, at = spec
    elapsed, ids = [], []
    for i in range(10):
        args = {'operation': 'append_actual', 'series_id': 'shared', 'valid_time': at,
                'source_available_at': at, 'value': index * 10 + i, 'idempotency_key': f'row-{i}'}
        replies = []
        for _ in range(2):
            started = perf_counter()
            reply = call(url, token, 'gnomon_ledger', args)
            elapsed.append(perf_counter()-started)
            assert reply['status'] == 'ok' and reply['state'] == 'completed', reply
            replies.append(reply['request_id'])
        assert replies[0] == replies[1]
        ids.append(replies[0])
    return elapsed, ids


def main():
    with TemporaryDirectory(prefix='gnomon-hosted-load-') as tmp:
        store = Store.initialize(tmp)
        project = store.create_project('load')['project_id']
        tokens = [store.issue_token(project, f'client-{i}', ['actual.create'])['token'] for i in range(4)]
        at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        started = perf_counter()
        with serving(store.root) as url:
            with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
                results = list(pool.map(client, [(url, token, i, at) for i, token in enumerate(tokens)]))
        with store.ledger(project).transaction() as conn:
            rows = conn.execute('SELECT COUNT(*),COUNT(DISTINCT revision) FROM actuals').fetchone()
            assert tuple(rows) == (40, 40)
            assert conn.execute('SELECT COUNT(*) FROM hosted_requests').fetchone()[0] == 40
        times = sorted(t for r, _ in results for t in r)
        print(json.dumps({'clients': 4, 'unique_writes': 40, 'exact_retries': 40,
            'errors': 0, 'wall_seconds': round(perf_counter()-started, 3),
            'median_request_seconds': round(statistics.median(times), 4),
            'p95_request_seconds': round(times[int(len(times)*.95)-1], 4),
            'max_request_seconds': round(max(times), 4),
            'scope': 'fresh MCP sessions, local SQLite, synthetic actuals; not production capacity'}))


if __name__ == '__main__':
    main()
