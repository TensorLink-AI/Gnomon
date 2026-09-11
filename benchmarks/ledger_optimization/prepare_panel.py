"""Prepare preregistered disjoint panels; do not score confirmation outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


SEED = 20260911
ARENA_COMMIT = 'b600eaa2c2691bebed926dac996d4b03e0c216e9'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(key):
    _, item, _, store = key.split('_')
    return int(item), int(store)


def disjoint_selection(candidates, excluded_ids):
    items, stores = set(), set()
    for key in excluded_ids:
        item, store = identity(key)
        items.add(item)
        stores.add(store)
    chosen = []
    for row in candidates:
        item, store = identity(row['unique_id'])
        if item in items or store in stores:
            continue
        chosen.append(row['unique_id'])
        items.add(item)
        stores.add(store)
        if len(chosen) == 32:
            break
    if len(chosen) != 32:
        raise ValueError(f'Only {len(chosen)} eligible disjoint series; no automatic rule change')
    random.Random(SEED).shuffle(chosen)
    return {'development': chosen[:8], 'confirmation': chosen[8:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arena-checkout', type=Path, required=True)
    parser.add_argument('--raw-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-manifest', type=Path, action='append', required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(args.arena_checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != ARENA_COMMIT:
        raise ValueError('Arena source differs from panel protocol')
    sys.path.insert(0, str(args.arena_checkout))
    from arena.favorita_eval import prepare_favorita_panel
    import pandas as pd
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise ValueError('Panel already frozen; refusing to overwrite')
    candidate_path = args.output/'candidates.parquet'
    if candidate_path.exists():
        source_manifest = json.loads(candidate_path.with_suffix('.parquet.manifest.json').read_text())
        if digest(candidate_path) != source_manifest['sha256']:
            raise ValueError('Candidate panel digest mismatch')
    else:
        source_manifest = prepare_favorita_panel(data_dir=args.raw_data, out=candidate_path,
            rounds=26, horizon=14, series_count=2000, seed=SEED, initial_history_days=365)
    excluded_ids = []
    for path in args.exclude_manifest:
        data = json.loads(path.read_text())
        excluded_ids.extend(row if isinstance(row, str) else row['unique_id'] for row in data['series'])
    chosen = disjoint_selection(source_manifest['series'], excluded_ids)
    # Data movement only. Selection above never reads future targets.
    frame = pd.read_parquet(candidate_path)
    splits = {}
    for split, ids in chosen.items():
        selected = frame[frame.unique_id.isin(ids)]
        path = args.output/f'{split}.parquet'
        selected.to_parquet(path, index=False)
        splits[split] = {'series': ids, 'rows': len(selected), 'sha256': digest(path), 'path': str(path),
                         'scored': False, 'forecast_outcomes_inspected': False}
    manifest = {'schema_version': 1, 'seed': SEED, 'arena_commit': commit,
        'protocol': 'benchmarks/ledger_optimization/PANEL_PROTOCOL.md',
        'protocol_sha256': digest(Path('benchmarks/ledger_optimization/PANEL_PROTOCOL.md')),
        'source_train_sha256': digest(args.raw_data/'train.csv'),
        'source_archive_sha256': digest(args.raw_data/'favorita-grocery-sales-forecasting2.zip'),
        'selection_uses_dates_through': source_manifest['selection_uses_dates_through'],
        'rounds': 26, 'horizon': 14, 'initial_history_days': 365,
        'selection_rule': 'Original training-only stratum order, exclude prior items/stores, first 32 disjoint pairs, seeded shuffle; 8 development and 24 confirmation.',
        'excluded_prior_ids': sorted(set(excluded_ids)), 'splits': splits,
        'missing_day_policy': source_manifest['missing_day_policy'],
        'negative_sales_policy': source_manifest['negative_sales_policy'],
        'no_confirmation_outcome_analysis': True}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
