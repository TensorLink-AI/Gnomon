"""Audit complete equal-information trials without accepting summary claims alone."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def audit(directory, cases_path, memory_path):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['historical_information_contract'] != 'same_raw_matched_history_in_every_arm':
        raise ValueError('This audit applies only to equal-history trials')
    cases_bytes, memory_bytes = cases_path.read_bytes(), memory_path.read_bytes()
    if hashlib.sha256(cases_bytes).hexdigest() != manifest['snapshot_sha256'] or hashlib.sha256(memory_bytes).hexdigest() != manifest['memory_sha256']:
        raise ValueError('Trial input hashes differ from the manifest')
    cases = {(c['series_id'], c['round']): c for c in json.loads(cases_bytes)}
    memory = {(p['series_id'], p['round']): p for p in json.loads(memory_bytes)['packets']}
    expected = set(product(manifest['series'], manifest['rounds'], manifest['seeds_requested'], manifest['arms']))
    rows, exposures, hashes = {}, defaultdict(list), []
    max_delta = 0
    for path in sorted(directory.glob('[0-9]*.json')):
        if path.name.endswith('harness-error.json'):
            raise ValueError('Unresolved harness failure')
        row = json.loads(path.read_text())
        key = (row['case']['series_id'], row['case']['round'], row['seed'], row['arm'])
        if key in rows or key not in expected:
            raise ValueError('Duplicate or unexpected decision')
        rows[key] = row
        case, packet = cases[key[:2]], memory[key[:2]]
        system, user = row['transcript'][:2]
        if system['role'] != 'system' or user['role'] != 'user':
            raise ValueError('Unexpected initial message structure')
        prompt = json.loads(user['content'])
        if prompt['raw_matched_history'] != packet['raw_history'] or prompt['current_context'] != packet['current_context']:
            raise ValueError('Historical exposure differs from the frozen packet')
        if prompt['request_fingerprint'] != packet['request_fingerprint']:
            raise ValueError('Prompt request identity differs from the packet')
        for historical in prompt['raw_matched_history']['records']:
            if datetime.fromisoformat(historical['origin']) >= datetime.fromisoformat(case['origin']):
                raise ValueError('Current or future origin exposed as historical evidence')
        if row['arm'] == 'no_ledger' and 'historical_evidence' in prompt:
            raise ValueError('Control received treatment summaries')
        shared = {k: v for k, v in prompt.items() if k != 'historical_evidence'}
        exposures[key[:3]].append(canonical({'system': system, 'shared': shared}))
        if row['forecast_attempts'] > manifest['forecast_attempt_budget']:
            raise ValueError('Execution budget exceeded')
        point = case['predictions'][row['provider']]
        if not row['fallback_used']:
            completion = row['resolution']['execution']
            if completion['provider'] != row['provider'] or completion['request_fingerprint'] != packet['request_fingerprint']:
                raise ValueError('Final execution does not match this task/provider')
            if completion['point'] != point:
                raise ValueError('Selected forecast differs from the fixed candidate')
        independently_scored = math.sqrt(mean((math.log1p(max(p, 0))-math.log1p(a))**2
                                             for p, a in zip(point, case['actual'], strict=True)))
        max_delta = max(max_delta, abs(independently_scored-row['rmsle']))
        if not math.isclose(independently_scored, row['rmsle'], abs_tol=1e-12):
            raise ValueError('Recorded RMSLE differs from the selected fixed forecast')
        hashes.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if set(rows) != expected or manifest['expected_decisions'] != len(expected):
        raise ValueError('Registered trial is incomplete')
    if any(len(set(values)) != 1 for values in exposures.values()):
        raise ValueError('Arms received different shared information or system instructions')
    return {'scope': manifest['scope'], 'decisions_checked': len(rows), 'matched_case_seed_pairs': len(exposures),
            'same_shared_information': True, 'same_system_instructions': True,
            'no_current_or_future_origin_in_history': True, 'forecast_budget_respected': True,
            'typed_execution_matches_fixed_candidate': True, 'scores_recomputed': True,
            'maximum_numerical_delta': max_delta, 'raw_results': hashes,
            'target_established': False,
            'limitation': 'Checks transcript exposure, frozen inputs, typed selection and arithmetic. '
                          'Source availability and context eligibility additionally rely on ledger queries/tests '
                          'and the disclosed synthetic replay assumptions. This is development evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.directory, args.cases, args.memory)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'raw_results'}, indent=2))


if __name__ == '__main__':
    main()
