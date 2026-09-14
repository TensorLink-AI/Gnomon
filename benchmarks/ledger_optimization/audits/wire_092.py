"""Offline 092 wire audit: retain failures and distinguish missing usage from zero.

Consumes an immutable snapshot-manifest.json listing completed session folders.
It neither queries models nor changes the trial's completion/scoring policy.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'snapshot-manifest.json').read_text())
    names = manifest['completed_session_folders']
    require(bool(names) and len(names) == len(set(names)), 'Unique completed folders required')
    sessions = []
    hashes = {}
    models = Counter()
    for name in names:
        folder = (root / name).resolve()
        require(folder.is_relative_to(root), 'Session outside snapshot')

        def read(path):
            raw = path.read_bytes()
            hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw)

        grade = read(folder / 'grade.json')
        files = {
            suffix: {p.name.removesuffix('-' + suffix + '.json'): p
                     for p in folder.glob('api-*-' + suffix + '.json')}
            for suffix in ('forwarded', 'response', 'receipt')
        }
        ids = set(files['forwarded'])
        require(ids == set(files['response']) == set(files['receipt']),
                'Every forwarded attempt needs its response and receipt')
        require(len(ids) == grade['api_calls'] and len(ids) <= 16,
                'Forwarded attempt count or original budget mismatch')
        usage = Counter()
        failures = []
        missing = []
        successes = 0
        for attempt in sorted(ids):
            request = read(files['forwarded'][attempt])
            response = read(files['response'][attempt])
            receipt = read(files['receipt'][attempt])
            require(request['model'] == 'deepseek-v4.1-flash'
                    and request['max_tokens'] == 3072, 'Frozen model settings changed')
            status = receipt['status']
            if status == 200:
                require(response.get('model') == 'deepseek-v4.1-flash',
                        'Successful response has unexpected model identity')
                models[response['model']] += 1
                successes += 1
            else:
                failures.append({'attempt': attempt, 'proxy_status': status})
            reported = response.get('usage')
            if reported is None:
                missing.append(attempt)
            else:
                require(all(type(reported.get(k)) is int and reported[k] >= 0
                            for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')),
                        'Usage present but malformed or incomplete')
                require(reported['total_tokens'] == reported['prompt_tokens']
                        + reported['completion_tokens'], 'Usage arithmetic mismatch')
                usage.update({k: reported[k] for k in
                              ('prompt_tokens', 'completion_tokens', 'total_tokens')})
        require(usage['total_tokens'] == grade['tokens'],
                'Grade differs from reported wire usage')
        sessions.append({
            'folder': name, 'forwarded_attempts': len(ids),
            'successful_responses': successes, 'failed_responses': failures,
            'reported_prompt_tokens': usage['prompt_tokens'],
            'reported_completion_tokens': usage['completion_tokens'],
            'reported_total_tokens': usage['total_tokens'],
            'attempts_without_usage': missing, 'usage_complete': not missing,
            'workflow_complete': grade['workflow_complete'], 'valid_forecast': grade['valid'],
        })
    return {
        'scope': 'All completed folders in supplied snapshot; no success-based exclusions.',
        'passed': True, 'sessions': sessions, 'input_hashes': hashes,
        'model_counts': dict(models),
        'forwarded_attempts': sum(s['forwarded_attempts'] for s in sessions),
        'successful_responses': sum(s['successful_responses'] for s in sessions),
        'failed_responses': sum(len(s['failed_responses']) for s in sessions),
        'reported_total_tokens': sum(s['reported_total_tokens'] for s in sessions),
        'attempts_without_usage': sum(len(s['attempts_without_usage']) for s in sessions),
        'usage_complete': all(s['usage_complete'] for s in sessions),
        'missing_usage_not_assumed_zero': True, 'billing_dollars': None,
        'readiness_probes_included': False, 'audit_api_calls': 0, 'audit_provider_calls': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sessions', 'input_hashes')}))


if __name__ == '__main__':
    main()
