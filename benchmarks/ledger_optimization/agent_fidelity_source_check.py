"""Independent089 answer-key audit against immutable original030 full reviews."""
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean
import sys

from .agent_fidelity import sha
from .agent_review_audit import INVENTORY_SHA


def check_sources(source, prepared, output):
    source, prepared, output = Path(source), Path(prepared), Path(output)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    cases = json.loads((prepared / 'cases.json').read_text())
    checks = []
    def require(ok, name):
        checks.append({'check': name, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(name)
    require(sha(source / 'SHA256SUMS.json') == INVENTORY_SHA, 'Original inventory identity')
    require(sha(prepared / 'cases.json') == manifest['cases_sha256'], 'Frozen prepared cases')
    for name, digest in manifest['source_access'].items():
        require(sha(source / name) == digest, 'Original accessed source unchanged')
    for case in cases:
        full = json.loads((source / case['source']).read_text())
        wanted = case['expected']
        at = datetime.fromisoformat(case['origin'])
        require(full['as_of'] == case['origin'], 'Exact query origin')
        indexes = sorted({0, len(full['cards']) - 1})
        require(case['question']['pair_indexes'] == indexes, 'First and last returned pairs fixed')
        require([p['pair_index'] for p in wanted['comparisons']] == indexes, 'Answer key covers requested pairs')
        for requested in wanted['comparisons']:
            card = full['cards'][requested['pair_index']]
            require(requested['recent_lifetime_disagreement'] == card.get('recent_lifetime_disagreement'),
                    'Exact disclosed disagreement')
            config_by_provider = {full['configuration_index'][cid]['provider']: cid
                                  for cid in (card['left_config_id'], card['right_config_id'])}
            for answer in requested['windows']:
                window = card['windows'][answer['window']]
                origins = window['origins']
                require(all(datetime.fromisoformat(o['origin']) < at for o in origins), 'All scored origins historical')
                require(all(datetime.fromisoformat(o['source_as_of']) <= at and
                            datetime.fromisoformat(o['recorded_as_of']) <= at for o in origins),
                        'Original evidence cutoffs do not exceed query')
                require(answer['matched_origins'] == len(origins), 'Count independently from origin rows')
                require(answer['n'] == sum(o['n'] for o in origins), 'Steps independently summed')
                require(answer['start'] == (origins[0]['origin'] if origins else None), 'First matched origin')
                require(answer['end'] == (origins[-1]['origin'] if origins else None), 'Last matched origin')
                calculated = {cid: mean(next(m['rmsle'] for m in o['models'] if m['provider'] == provider)
                                        for o in origins) for provider, cid in config_by_provider.items()} if origins else {}
                submitted = {s['config_id']: s['rmsle'] for s in answer['scores']}
                require(set(calculated) == set(submitted), 'Exact scored configurations')
                require(all(math.isclose(calculated[c], submitted[c], rel_tol=0, abs_tol=1e-14) for c in calculated),
                        'RMSLE aggregate independently reconstructed from original origin metrics')
                original_scores = {config_by_provider[m['provider']]: m['rmsle'] for m in window['models']}
                winners = sorted(c for c, score in original_scores.items() if score == min(original_scores.values()))
                require(sorted(answer['lowest_error_config_ids']) == winners, 'Exact ties from original reported scores')
                require(all(submitted[c] == original_scores[c] for c in submitted), 'Answer key preserves exact reported floats')
        require(wanted['pagination'] == {
            'shown_pairs': len(full['cards']), 'total_pairs': full['total_pairs'], 'next_offset': full['next_offset'],
            'all_pairs_included': full['offset'] == 0 and full['next_offset'] is None and len(full['cards']) == full['total_pairs'],
        }, 'Coverage directly from original page')
        require(wanted['provider_calls'] == full['provider_calls'] == 0, 'No forecast calls in review')
    for name, digest in manifest['source_access'].items():
        require(sha(source / name) == digest, 'Source unchanged after audit')
    result = {'passed': True, 'checks': checks, 'check_count': len(checks), 'source_files': len(manifest['source_access']),
              'model_calls': 0, 'forecast_calls': 0, 'source_mutations': 0,
              'scope': 'Answer-key identity and arithmetic over original recorded metrics; not a new forecast verification.'}
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(check_sources(*sys.argv[1:]), indent=2))
