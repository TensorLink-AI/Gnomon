"""Freeze and prepare the registered confirmation run without optional stopping."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess

from gnomon.build_info import build_info

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = sorted(['sf_seasonal_naive_7', 'sf_window_average_28', 'sf_auto_ets_7', 'sf_auto_arima_7',
                     'sf_auto_arima_promo_7', 'sf_auto_theta_7', 'sf_croston_optimized', 'sf_mstl_7'])


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def implementation_files():
    paths = sorted((ROOT/'benchmarks/ledger_optimization').glob('*.py'))
    paths += [ROOT/'benchmarks/ledger_optimization/PANEL_PROTOCOL.md', ROOT/'benchmarks/ledger_optimization/CONTEXT_PROTOCOL.md']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def create(panel_path, report_path, audit_path, selected_arm, output):
    if output.exists():
        raise ValueError('Confirmation freeze already exists; do not overwrite')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'dev/ledger-optimization':
        raise ValueError('Confirmation belongs on the development branch')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src/gnomon',
                                    'benchmarks/ledger_optimization'], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError('Commit implementation/protocol/results before freezing confirmation')
    panel = json.loads(panel_path.read_text())
    report = json.loads(report_path.read_text())
    audit = json.loads(audit_path.read_text())
    if report['run_completion'] != 'complete' or selected_arm not in report['mean_case_rmsle']:
        raise ValueError('Select an arm from a complete development run')
    if report['manifest']['historical_information_contract'] != 'same_raw_matched_history_in_every_arm':
        raise ValueError('Final control must have equal historical information')
    if not audit['same_shared_information'] or not audit['scores_recomputed'] or audit['decisions_checked'] != report['recorded_decisions']:
        raise ValueError('A complete equal-information arithmetic audit is required')
    if {r['sha256'] for r in audit['raw_results']} != {r['sha256'] for r in report['raw_results']}:
        raise ValueError('Development audit and report refer to different executions')
    if selected_arm not in ('ledger_context', 'ledger_supported'):
        raise ValueError('Only tested ledger-only treatments may be frozen')
    if len(panel['splits']['confirmation']['series']) != 24 or panel['rounds'] != 26 or panel['horizon'] != 14:
        raise ValueError('Panel does not match the preregistered confirmation design')
    frozen = {'schema': 'ledger_confirmation_freeze/1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'implementation_files': implementation_files(), 'gnomon_source_sha256': build_info()['source_sha256'],
              'panel_manifest': str(panel_path.resolve()), 'panel_manifest_sha256': digest(panel_path),
              'series': sorted(panel['splits']['confirmation']['series']), 'rounds': list(range(26)),
              'candidate_names': CANDIDATES, 'statsforecast_version': '2.0.3', 'horizon': 14,
              'seeds_requested': [7, 19], 'arms': ['no_ledger', 'ledger_119', selected_arm],
              'expected_decisions': 3744, 'model': report['manifest']['model'],
              'temperature': .2, 'max_tokens_per_turn': 2048, 'max_turns': 6, 'forecast_attempt_budget': 3,
              'historical_information_contract': 'same_raw_matched_history_in_every_arm',
              'development_report': {'path': str(report_path), 'sha256': digest(report_path)},
              'development_audit': {'path': str(audit_path), 'sha256': digest(audit_path)},
              'target': {'mean_case_rmsle_relative_reduction': .2, 'paired_interval_lower_bound_above': 0,
                         'must_improve_over_original_ledger': True},
              'bootstrap': {'replicates': 5000, 'seed': 20260911, 'shared_circular_block_length': 4,
                            'resample_series': True, 'retain_agent_seeds_together': True},
              'confirmation_previously_scored': False, 'optional_stopping': False}
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True)+'\n')
    return frozen


def validate(path):
    frozen = json.loads(Path(path).read_text())
    if frozen['schema'] != 'ledger_confirmation_freeze/1':
        raise ValueError('Unsupported freeze format')
    if implementation_files() != frozen['implementation_files'] or build_info()['source_sha256'] != frozen['gnomon_source_sha256']:
        raise ValueError('Implementation changed after confirmation freeze')
    panel_path = Path(frozen['panel_manifest'])
    if digest(panel_path) != frozen['panel_manifest_sha256']:
        raise ValueError('Frozen panel manifest changed')
    panel = json.loads(panel_path.read_text())
    if sorted(panel['splits']['confirmation']['series']) != frozen['series']:
        raise ValueError('Confirmation identities differ from freeze')
    if frozen['rounds'] != list(range(26)) or frozen['seeds_requested'] != [7, 19] or frozen['expected_decisions'] != 3744:
        raise ValueError('Confirmation cohort was changed')
    if frozen['arms'][:2] != ['no_ledger', 'ledger_119'] or len(frozen['arms']) != 3 or frozen['arms'][2] not in ('ledger_context', 'ledger_supported'):
        raise ValueError('Confirmation controls were changed')
    return frozen, panel


def validate_cases(cases, frozen):
    keys = [(c['series_id'], c['round']) for c in cases]
    if len(set(keys)) != len(keys) or set(keys) != set(product(frozen['series'], frozen['rounds'])):
        raise ValueError('Prepared cases are not the full unique series/origin grid')
    origins = {}
    for case in cases:
        if sorted(case['predictions']) != frozen['candidate_names']:
            raise ValueError('Forecast candidates differ from the frozen portfolio')
        if len(case['actual']) != frozen['horizon'] or len(case['future_timestamps']) != frozen['horizon']:
            raise ValueError('Prepared horizon differs from freeze')
        if any(len(point) != frozen['horizon'] for point in case['predictions'].values()):
            raise ValueError('Candidate forecast horizon mismatch')
        origin = datetime.fromisoformat(case['origin'])
        if origin.tzinfo is None:
            raise ValueError('Origins require a timezone')
        if case['round'] in origins and origins[case['round']] != origin:
            raise ValueError('Series do not share the registered origin calendar')
        origins[case['round']] = origin


def prepare(freeze_path, arena_checkout, output, workers):
    from .cache_panel import _series_job, record
    from .context_memory import prepare as prepare_memory
    frozen, panel = validate(freeze_path)
    if output.exists():
        raise ValueError('Use a new confirmation preparation directory; never overwrite evidence')
    commit = subprocess.check_output(['git', '-C', str(arena_checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != panel['arena_commit']:
        raise ValueError('Arena implementation differs from panel protocol')
    dirty = subprocess.check_output(['git', '-C', str(arena_checkout), 'status', '--porcelain', '--', 'arena'], text=True)
    if dirty.strip():
        raise ValueError('Arena source has uncommitted changes')
    from importlib.metadata import version
    if version('statsforecast') != frozen['statsforecast_version']:
        raise ValueError('StatsForecast version differs from freeze')
    source = panel['splits']['confirmation']
    panel_path = Path(source['path'])
    if digest(panel_path) != source['sha256']:
        raise ValueError('Confirmation data changed before preparation')
    output.mkdir(parents=True)
    cases = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_series_job, str(arena_checkout), str(panel_path), sid, str(output)): sid
                   for sid in frozen['series']}
        for future in as_completed(futures):
            cases.extend(future.result())
            print(json.dumps({'series': futures[future], 'candidate_preparation': 'complete'}), flush=True)
    cases.sort(key=lambda c: (c['origin'], c['series_id']))
    if len(cases) != 624:
        raise ValueError('Confirmation preparation is incomplete')
    validate_cases(cases, frozen)
    prepared = record(cases, output/'ledger.db')
    (output/'cases.json').write_text(json.dumps(prepared, sort_keys=True, allow_nan=False)+'\n')
    prepare_memory(prepared, output/'ledger.db', output/'memory', scope='confirmation')
    receipt = {'scope': 'confirmation', 'freeze_sha256': digest(freeze_path),
               'cases': len(prepared), 'cases_sha256': digest(output/'cases.json'),
               'memory_sha256': digest(output/'memory/memory.json'), 'source_panel_sha256': source['sha256'],
               'actual_visibility': 'source at valid date, recorded at horizon close; synthetic replay assumption',
               'agent_decisions_started': False}
    (output/'manifest.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--panel-manifest', type=Path, required=True)
    freeze.add_argument('--development-report', type=Path, required=True)
    freeze.add_argument('--development-audit', type=Path, required=True)
    freeze.add_argument('--selected-arm', required=True)
    freeze.add_argument('--output', type=Path, required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--freeze', type=Path, required=True)
    preparation.add_argument('--arena-checkout', type=Path, required=True)
    preparation.add_argument('--output', type=Path, required=True)
    preparation.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    result = create(args.panel_manifest, args.development_report, args.development_audit,
                    args.selected_arm, args.output) if args.operation == 'freeze' else prepare(
                        args.freeze, args.arena_checkout, args.output, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
