"""Download pinned Arrow bytes and freeze deterministic series/origin manifests."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request

from .common import PROTOCOL, digest, read, write


def download(protocol, spec, cache):
    name = spec['id']
    relative = f'{name}/data-00000-of-00001.arrow'
    url = f"https://huggingface.co/datasets/{protocol['dataset_repo']}/resolve/{protocol['dataset_revision']}/{relative}"
    path = Path(cache)/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix('.download')
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as out:
            while chunk := response.read(1024*1024):
                out.write(chunk)
        tmp.replace(path)
    return path, {'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def origins(length, horizon, stride, count, min_history):
    first = length - horizon - (count-1)*stride
    if first < min_history:
        return []
    return list(range(first, length-horizon+1, stride))


def forward_fill(values):
    result, last = [], 0.0
    for value in values:
        if value is not None and math.isfinite(value):
            last = float(value)
        result.append(last)
    return result


def prepare(protocol, spec, path, source, mode):
    import pyarrow as pa
    import pandas as pd
    with pa.memory_map(str(path), 'r') as stream:
        table = pa.ipc.open_stream(stream).read_all()
    candidates = []
    for row in table.to_pylist():
        targets = row['target']
        multivariate = bool(targets) and isinstance(targets[0], list)
        for dim, target in enumerate(targets if multivariate else [targets]):
            sid = str(row['item_id']) + (f':dim{dim}' if multivariate else '')
            candidates.append((sid, row, target))
    candidates.sort(key=lambda x: hashlib.sha256(f"{protocol['seed']}:{spec['id']}:{x[0]}".encode()).hexdigest())
    if len({x[0] for x in candidates}) != len(candidates):
        raise ValueError('Duplicate series IDs')
    # Selection depends on IDs/length only, never held-out values or performance.
    planned_count = protocol['validation_origins'] + protocol['warmup_origins'] + protocol['score_origins']
    selected, excluded = [], []
    for sid, row, target in candidates:
        positions = origins(len(target), spec['horizon'], spec['stride'], planned_count, max(24, 2*spec['season']))
        if not positions:
            excluded.append({'series': sid, 'reason': 'insufficient_length', 'length': len(target)})
            continue
        selected.append((sid, row, target, positions))
    # Reserve pilot series from evaluation in the two pilot configurations.
    limit = protocol['pilot_series'] if mode == 'pilot' else protocol['series_limit']
    offset = protocol['pilot_series'] if mode == 'evaluation' and spec['id'] in protocol['pilot_datasets'] else 0
    selected = selected[offset:offset+limit]
    series = []
    for sid, row, target, positions in selected:
        nv, nw = protocol['validation_origins'], protocol['warmup_origins']
        if mode == 'pilot':
            positions = positions[:nv+nw+protocol['pilot_score_origins']]
        freq = {'H':'h', 'T':'min', 'M':'MS', '10T':'10min', '15T':'15min', 'S':'s'}.get(row['freq'], row['freq'])
        start = pd.Timestamp(row['start'])
        # UTC is an indexing convention for timezone-naive upstream data, not a timezone assertion.
        index = pd.date_range(start, periods=len(target), freq=freq)
        index = index.tz_localize('UTC') if index.tz is None else index.tz_convert('UTC')
        clean = [float(v) if v is not None and math.isfinite(v) else None for v in target]
        train = forward_fill(clean[:positions[nv+nw]])
        season = spec['season']
        scale = sum(abs(a-b) for a,b in zip(train[season:],train[:-season]))/(len(train)-season)
        phases = ['validation']*nv + ['warmup']*nw + ['score']*(len(positions)-nv-nw)
        series.append({'id': sid, 'values': clean, 'timestamps': [t.isoformat() for t in index],
                       'origins': positions, 'phases': phases, 'mase_scale': scale,
                       'freq': row['freq'], 'source_item': str(row['item_id'])})
    if not series:
        raise ValueError(f"No eligible series for {spec['id']}")
    manifest = {'schema_version': 1, 'protocol': protocol, 'spec': spec, 'mode': mode,
                'source': source, 'excluded': excluded, 'series': series,
                'scope': 'Univariate targets; upstream covariates not used. UTC labels are indexing conventions.',
                'available_series':len(candidates)}
    manifest['sha256'] = digest(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['pilot','evaluation'], required=True)
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    for spec in protocol['datasets']:
        if args.mode == 'pilot' and spec['id'] not in protocol['pilot_datasets']:
            continue
        path, source = download(protocol, spec, args.cache)
        manifest = prepare(protocol, spec, path, source, args.mode)
        target = args.output/(spec['id'].replace('/','_')+'.json')
        write(target, manifest)
        print(json.dumps({'manifest':str(target), 'series':len(manifest['series']), 'sha256':manifest['sha256']}), flush=True)


if __name__ == '__main__':
    main()
