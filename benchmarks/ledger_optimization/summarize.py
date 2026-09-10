"""Produce a compact, hash-linked report without changing experiment scores."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    records, paths = [], []
    for p in sorted(directory.glob('[0-9]*.json')):
        if p.name.endswith('.harness-error.json'):
            continue
        row = json.loads(p.read_text())
        records.append(row)
        paths.append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    by_case, by_arm = defaultdict(dict), defaultdict(list)
    for r in records:
        case = (r['case']['series_id'], r['case']['round'], r['seed'])
        by_case[case][r['arm']] = r
        by_arm[r['arm']].append(r)
    complete = {k: v for k, v in by_case.items() if set(v) == set(manifest['arms'])}
    scores = {arm: mean(r[arm]['rmsle'] for r in complete.values()) for arm in manifest['arms']} if complete else {}
    return {'scope': manifest['scope'], 'manifest': manifest, 'complete_matched_cases': len(complete),
        'incomplete_case_count': len(by_case)-len(complete), 'harness_failures': len(list(directory.glob('*.harness-error.json'))),
        'mean_case_rmsle': scores,
        'relative_improvement_vs_no_ledger': {arm: 1-score/scores['no_ledger'] if scores['no_ledger'] else None
                                             for arm, score in scores.items() if arm != 'no_ledger'},
        'target_established': False,
        'limitation': 'Development experiment, not an untouched final test; too few independent series to establish the objective.',
        'completion': {arm: {'recorded_decisions': len(rows), 'fallbacks': sum(r['fallback_used'] for r in rows),
                            'resolutions': dict(Counter(r['resolution']['status'] for r in rows)),
                            'forecast_attempts': sum(r['forecast_attempts'] for r in rows),
                            'successful_executions': sum(r['successful_executions'] for r in rows),
                            'api_calls': sum(r['api_calls'] for r in rows),
                            'prompt_tokens': sum(u.get('prompt_tokens', 0) for r in rows for u in r['api_usage']),
                            'completion_tokens': sum(u.get('completion_tokens', 0) for r in rows for u in r['api_usage']),
                            'reported_api_cost_usd': sum(r['api_cost_usd'] for r in rows) if all(r['api_cost_usd'] is not None for r in rows) else None}
                       for arm, rows in by_arm.items()},
        'per_case': [{**r['case'], 'arm': r['arm'], 'seed': r['seed'], 'rmsle': r['rmsle'], 'provider': r['provider'],
                      'fallback_used': r['fallback_used'], 'resolution_status': r['resolution']['status']} for r in records],
        'raw_results': paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('raw_results', 'per_case', 'manifest')}, indent=2))


if __name__ == '__main__':
    main()
