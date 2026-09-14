"""Read-only cost accounting; copied pilot requests are counted exactly once."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ARMS = ('plain', 'gnomon', 'ledger')
COUNTS = ('sessions', 'forwarded_requests', 'returned_responses', 'reported_tokens',
          'requests_without_reported_usage', 'api_errors', 'readiness_requests',
          'readiness_reported_tokens', 'readiness_unknown_usage', 'orphan_responses')


def read(path):
    try:
        return json.loads(path.read_text())
    except (ValueError, FileNotFoundError):
        return None


def token_count(value):
    usage = value.get('usage') if isinstance(value, dict) else None
    total = usage.get('total_tokens') if isinstance(usage, dict) else None
    return total if type(total) is int and total >= 0 else None


def session_cost(folder):
    value = dict.fromkeys(COUNTS, 0)
    value['sessions'] = 1
    requests = {p.name.removesuffix('-forwarded.json'):p for p in folder.glob('api-*-forwarded.json')}
    responses = {p.name.removesuffix('-response.json'):p for p in folder.glob('api-*-response.json')}
    value['forwarded_requests'] = len(requests)
    value['returned_responses'] = len(requests.keys() & responses.keys())
    value['orphan_responses'] = len(responses.keys() - requests.keys())
    for identifier in requests:
        response = read(folder / (identifier + '-response.json'))
        tokens = token_count(response)
        if tokens is None:
            value['requests_without_reported_usage'] += 1
        else:
            value['reported_tokens'] += tokens
        receipt = read(folder / (identifier + '-receipt.json'))
        http_error = isinstance(receipt,dict) and type(receipt.get('status')) is int and receipt['status'] >= 400
        value['api_errors'] += bool(http_error or (isinstance(response,dict) and response.get('error')))
    for path in (folder / 'service-admission').glob('probe-*-receipt.json'):
        value['readiness_requests'] += 1
        tokens = token_count(read(path))
        if tokens is None:
            value['readiness_unknown_usage'] += 1
        else:
            value['readiness_reported_tokens'] += tokens
    return value


def summarize(root):
    if not root.is_dir():
        raise ValueError('Run directory does not exist')
    rows = []
    # Only canonical session paths, never work/home copies or an evidence archive.
    for folder in sorted(root.glob('*/*/round-*')):
        arm, series, name = folder.relative_to(root).parts
        if arm not in ARMS or not folder.is_dir():
            raise ValueError('Unexpected session path: ' + str(folder))
        number = int(name.removeprefix('round-'))
        if number < 0 or name != f'round-{number}':
            raise ValueError('Noncanonical round path')
        rows.append({'arm':arm, 'series_id':series, 'round':number,
            'stage':'retained_pilot' if number < 3 else 'continuation', **session_cost(folder)})
    def aggregate(selected):
        return {key:sum(row[key] for row in selected) for key in COUNTS}
    stages = {stage:aggregate([r for r in rows if r['stage']==stage])
              for stage in ('retained_pilot','continuation')}
    result = {'at':datetime.now(timezone.utc).isoformat(), 'root':str(root),
        'stages':stages, 'deduplicated_total':aggregate(rows),
        'arms':{a:aggregate([r for r in rows if r['arm']==a]) for a in ARMS},
        'billing_dollars':None, 'sessions':rows,
        'scope':'One canonical record per arm/series/origin. Unknown usage includes in-flight and unreported requests; it is not zero. Reported tokens include billed context again on each request and are not unique text tokens. Pilot and incremental continuation stages are disjoint.'}
    for key in COUNTS:
        if result['deduplicated_total'][key] != sum(v[key] for v in stages.values()):
            raise ValueError('Cost stage reconciliation failed')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(json.dumps(summarize(parser.parse_args().root),indent=2))
