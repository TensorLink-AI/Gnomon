"""Hash-pinned, prefix-only M5 selection; export development targets only."""
import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
from math import isfinite
from pathlib import Path
import urllib.request
from zipfile import ZipFile

COMMIT = '72b8e7fd3b565b3c538adcb1d1a05117d8562d7e'
BLOB = '925e92ee03cdb83c54a1dbbc12260a3b1939753c'
SIZE = 50219189
URL = f'https://raw.githubusercontent.com/Nixtla/m5-forecasts/{COMMIT}/datasets/m5.zip'
FIRST, CUTOFF, LAST = 1212, 1577, 1941
SEED = 20260912


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_archive(path):
    if path.stat().st_size != SIZE:
        raise ValueError('Pinned archive size mismatch')
    git = hashlib.sha1(f'blob {SIZE}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            git.update(chunk)
    if git.hexdigest() != BLOB:
        raise ValueError('Pinned archive Git blob mismatch')
    return digest(path)


def fetch(path):
    if path.exists() or path.with_suffix(path.suffix + '.partial').exists():
        raise ValueError('Refuse to overwrite a source download')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    received = 0
    with urllib.request.urlopen(URL, timeout=45) as source, partial.open('xb') as target:
        while chunk := source.read(1024 * 1024):
            received += len(chunk)
            if received > SIZE:
                raise ValueError('Source exceeds pinned size')
            target.write(chunk)
    sha = verify_archive(partial)
    partial.rename(path)
    return {'source_url': URL, 'source_commit': COMMIT, 'git_blob': BLOB,
            'bytes': received, 'sha256': sha, 'archive': str(path)}


def rank(kind, identity):
    return hashlib.sha256(f'{SEED}:m5:{kind}:{identity}'.encode()).hexdigest()


def prefix_metadata(rows):
    """Do not inspect any target field later than CUTOFF for eligibility/selection."""
    eligible, reasons, stores, seen = [], {}, set(), set()
    for row in rows:
        store, item = row['store_id'], row['item_id']
        key = store, item
        if not store or not item or key in seen:
            raise ValueError('Missing or duplicate series identity')
        seen.add(key)
        stores.add(store)
        try:
            values = [float(row[f'd_{i}']) for i in range(FIRST, CUTOFF + 1)]
        except (KeyError, ValueError, TypeError):
            reason = 'invalid_initial_history'
        else:
            reason = ('invalid_initial_history' if any(not isfinite(v) or v < 0 for v in values)
                      else 'fewer_than_28_nonzero_initial_days' if sum(v > 0 for v in values) < 28 else None)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        eligible.append({'store_id': store, 'item_id': item, 'series_id': f'{item}_{store}',
                         'initial_count': len(values), 'initial_nonzero_count': sum(v > 0 for v in values),
                         'initial_history_sha256': hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()})
    return eligible, {'total_series': len(seen), 'eligible_series': len(eligible),
                      'stores': sorted(stores), 'rejected_prefix_reasons': reasons}


def select(eligible, stores):
    stores = sorted(set(stores), key=lambda s: (rank('store', s), s))
    if len(stores) != 10:
        raise ValueError('Require exactly ten source stores')
    result = {'development': [], 'reserved': []}
    used_items = set()
    for index, store in enumerate(stores):
        split, count = ('development', 4) if index < 2 else ('reserved', 3)
        candidates = sorted((r for r in eligible if r['store_id'] == store),
                            key=lambda r: (rank('series', f"{store}:{r['item_id']}"), r['item_id']))
        chosen = []
        for row in candidates:
            if row['item_id'] in used_items:
                continue
            chosen.append(row)
            used_items.add(row['item_id'])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f'Insufficient disjoint eligible items in {store}; no rule relaxation')
        result[split].extend(chosen)
    return result


def member(zipped, name):
    matches = [n for n in zipped.namelist() if Path(n).name == name]
    if len(matches) != 1:
        raise ValueError(f'Require exactly one {name} archive member')
    return matches[0]


def rows(zipped, name):
    with zipped.open(member(zipped, name)) as raw:
        with io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as stream:
            yield from csv.DictReader(stream)


def calendar_mapping(source_rows):
    # The pinned mirror omits d; its documented loader uses one-based row order.
    result = {}
    previous = None
    for index, row in enumerate(source_rows, 1):
        key = f'd_{index}'
        if 'd' in row and row['d'] != key:
            raise ValueError('Calendar day identifiers disagree with row order')
        current = date.fromisoformat(row['date'])
        if previous is not None and current - previous != timedelta(days=1):
            raise ValueError('Calendar must have consecutive daily rows')
        result[key] = current
        previous = current
    return result


def prepare(archive, output):
    if output.exists():
        raise ValueError('Refuse to reuse any preparation directory')
    sha = verify_archive(archive)
    with ZipFile(archive) as zipped:
        eligible, counts = prefix_metadata(rows(zipped, 'sales_train_evaluation.csv'))
        splits = select(eligible, counts['stores'])
        calendar = calendar_mapping(rows(zipped, 'calendar.csv'))
        days = [calendar[f'd_{i}'] for i in range(FIRST, LAST + 1)]
        if any(b - a != timedelta(days=1) for a, b in zip(days, days[1:])):
            raise ValueError('Calendar is not a complete daily grid')
        ids = {(r['store_id'], r['item_id']): r['series_id'] for r in splits['development']}
        development = []
        # Later values are converted to numbers only for the preselected development IDs.
        for row in rows(zipped, 'sales_train_evaluation.csv'):
            identity = row['store_id'], row['item_id']
            if identity not in ids:
                continue
            for i, day in zip(range(FIRST, LAST + 1), days, strict=True):
                value = float(row[f'd_{i}'])
                if not isfinite(value) or value < 0:
                    raise ValueError('Invalid development target; stop without replacing the series')
                development.append({'unique_id': ids[identity], 'source_date': day.isoformat(),
                                    'ds': (day + timedelta(days=1)).isoformat() + 'T00:00:00+00:00',
                                    'y': value, 'onpromotion': 0.0})
    if len(development) != 8 * 730:
        raise ValueError('Development grid incomplete')
    output.mkdir(parents=True)
    path = output / 'development.csv'
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['unique_id', 'source_date', 'ds', 'y', 'onpromotion'])
        writer.writeheader()
        writer.writerows(sorted(development, key=lambda r: (r['unique_id'], r['ds'])))
    protocol = Path(__file__).with_name('M5_PROTOCOL_014.md')
    manifest = {'scope': 'source amendment and development preparation only',
                'source': {'url': URL, 'commit': COMMIT, 'git_blob': BLOB, 'sha256': sha, 'archive': str(archive)},
                'source_attribution': 'M5 forecasting competition / Walmart; pinned Nixtla data mirror',
                'seed': SEED, 'first_day': FIRST, 'selection_cutoff_day': CUTOFF, 'last_day': LAST,
                'horizon': 14, 'rounds': 26, 'origin_indices': [365 + 14 * i for i in range(26)],
                'eligibility': counts, 'splits': splits,
                'development': {'path': str(path), 'rows': len(development), 'sha256': digest(path)},
                'source_target': 'observed daily sales; not latent demand',
                'timestamp_semantics': 'published reporting date plus one day, 00:00 UTC period-end replay convention',
                'availability': 'source and recording at period-end, assumed not observed',
                'promotion_feature': 'unavailable_assumed_zero; not an observed no-promotion assertion',
                'reserved_future_targets_numeric_inspection': False, 'reserved_forecasts_computed': False,
                'target_established': False, 'api_calls': 0, 'provider_calls': 0,
                'source_hashes': {Path(__file__).name: digest(Path(__file__)), protocol.name: digest(protocol)}}
    with (output / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('fetch')
    download.add_argument('--archive', type=Path, required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--archive', type=Path, required=True)
    preparation.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = fetch(args.archive) if args.command == 'fetch' else prepare(args.archive, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
