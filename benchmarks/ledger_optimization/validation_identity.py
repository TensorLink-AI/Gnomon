"""Freeze unused development-validation identities from metadata alone."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

FIELDS = {'series_name', 'start_label', 'row_length', 'first_history_index',
          'first_origin_index', 'end_index', 'initial_history_sha256', 'sensor_id'}


def choose(domain, eligible, previous):
    if len(eligible) < 32 or len({r['series_name'] for r in eligible}) != len(eligible):
        raise ValueError('At least32 unique eligible identities required')
    if any(set(r)-FIELDS for r in eligible):raise ValueError('Only frozen eligibility metadata may be supplied')
    ordered = sorted(eligible, key=lambda r: hashlib.sha256(
        f'20260914:panel035:{domain}:{r["series_name"]}'.encode()).hexdigest())
    names = lambda rows: [r['series_name'] for r in rows]
    if names(ordered[:8]) != names(previous['development']) or names(ordered[8:24]) != names(previous['reserved']):
        raise ValueError('Original partition differs from frozen hash order')
    selected = ordered[24:32]
    if set(names(selected)) & set(names(previous['development']+previous['reserved'])):
        raise ValueError('Validation overlaps original development or final reserve')
    return {'validation': selected, 'original_development': names(previous['development']),
            'reserved_final_unchanged': names(previous['reserved']), 'eligible_count': len(eligible)}


def prepare(panel, output):
    panel, output = Path(panel), Path(output); here = Path(__file__).parent
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):(output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    receipt_path = here/'evidence/broad-panel-037.json'; receipt = json.loads(receipt_path.read_text())
    manifest = {'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256((here/'BROAD_VALIDATION_052.md').read_bytes()).hexdigest(),
        'source_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(), 'metadata_files': {},
        'scored_outcome_values_read': 0, 'reserved_future_values_read': 0, 'provider_calls': 0, 'api_calls': 0,
        'locked_implementation_sha256': {n: hashlib.sha256((here/n).read_bytes()).hexdigest() for n in
            ('hourly_numerical.py', 'warm_context.py', 'context_ensemble.py', 'intraday_ensemble.py',
             'evidence_ensemble.py', 'ensemble_refinement.py')}}
    save('manifest.json', manifest)
    try:
        for name, sha in manifest['locked_implementation_sha256'].items():
            frozen = subprocess.run(['git', 'show', '11af992:benchmarks/ledger_optimization/'+name],
                                    check=True, capture_output=True).stdout
            if hashlib.sha256(frozen).hexdigest() != sha:raise ValueError('Chosen 050 implementation changed: '+name)
        manifest['locked_implementation_commit'] = '11af992'
        loaded = {}
        for name in ('selection.json', 'electricity-eligibility.json', 'pedestrian-eligibility.json'):
            raw = (panel/name).read_bytes(); sha = hashlib.sha256(raw).hexdigest()
            if receipt['files'][name] != sha:raise ValueError('Frozen metadata changed: '+name)
            loaded[name] = json.loads(raw); manifest['metadata_files'][name] = sha
        selection = {domain: choose(domain, loaded[domain+'-eligibility.json']['eligible'], loaded['selection.json'][domain])
                     for domain in ('electricity', 'pedestrian')}
        save('selection.json', selection); save('manifest.json', manifest)
        save('COMPLETED.json', {'validation_series': 16, 'planned_scored_cases': 416,
            'planned_warmup_attempts': 128, 'reserved_final_series_unchanged': 32,
            'scored_outcome_values_read': 0, 'reserved_future_values_read': 0,
            'selection_sha256': hashlib.sha256((output/'selection.json').read_bytes()).hexdigest()})
        return selection
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'scored_outcome_values_read': 0})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('panel'); p.add_argument('output')
    r = prepare(**vars(p.parse_args()))
    print(json.dumps({d: [v['series_name'] for v in rows['validation']] for d, rows in r.items()}, indent=2))
