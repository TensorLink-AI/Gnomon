"""Extract only the locked 052 identities, retaining unavailable warm-up hours."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import time

from .broad_panel_prepare import SOURCES, numbers, source_rows
from .broad_warmup_prepare import collect
from .broad_warmup_available import boundaries
from .pedestrian_timestamp_coverage import SHA, FIRST_ORIGIN, END
from .publisher_panel_prepare import csv_rows, digest

HOUR = timedelta(hours=1)


def split_span(identity, values, start, source_sha, unit, expected_prefix):
    if len(values) != 6298:raise ValueError('Complete positional span required')
    scored = values[1344:]; warm = values[:2074]
    if any(v is None for v in scored):raise ValueError('Missing locked scored observation')
    scored = numbers(scored)
    observed = hashlib.sha256(json.dumps(scored[:730], separators=(',', ':')).encode()).hexdigest()
    if observed != expected_prefix:raise ValueError('Original initial-history hash changed')
    warm = [numbers([v])[0] if v is not None else None for v in warm]
    if warm[-730:] != scored[:730]:raise ValueError('Overlap mismatch')
    first = start+2074*HOUR
    base = {'source_sha256': source_sha, 'unit': unit}
    return ({**base, 'start_label': (start+1344*HOUR).isoformat(), 'first_origin': first.isoformat(), 'values': scored},
            {**base, 'start_label': start.isoformat(), 'first_origin': (start+730*HOUR).isoformat(),
             'first_scored_origin': first.isoformat(), 'values': warm, 'original_overlap_matches': True})


def prepare(electricity, pedestrian, identity, coverage, output):
    identity, coverage, output = Path(identity), Path(coverage), Path(output); here = Path(__file__).parent
    output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    def save(name, value):(output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    manifest = {'code_sha256': digest(__file__), 'protocol_sha256': digest(here/'BROAD_VALIDATION_052.md'),
        'helper_sha256': {n: digest(here/n) for n in ('broad_panel_prepare.py', 'broad_warmup_prepare.py',
            'broad_warmup_available.py', 'pedestrian_timestamp_coverage.py', 'publisher_panel_prepare.py')},
        'forecast_computations': 0, 'api_calls': 0, 'reserved_future_values_parsed': 0}
    save('manifest.json', manifest)
    try:
        receipt_path = here/'evidence/broad-validation-identity-052.json'; receipt = json.loads(receipt_path.read_text())
        old = json.loads((here/'evidence/pedestrian-source-036.json').read_text())
        if digest(identity/'selection.json') != receipt['files']['selection.json']:raise ValueError('Named validation IDs changed')
        if digest(here/'BROAD_VALIDATION_052.md') != receipt['manifest']['protocol_sha256']:raise ValueError('Frozen validation protocol changed')
        if digest(coverage) != old['coverage_receipt_sha256']:raise ValueError('Source coverage receipt changed')
        if digest(electricity) != SOURCES['electricity']['sha'] or digest(pedestrian) != SHA:raise ValueError('Source archive changed')
        selection = json.loads((identity/'selection.json').read_text())
        manifest.update(identity_receipt_sha256=digest(receipt_path), selection_sha256=digest(identity/'selection.json'),
            coverage_sha256=digest(coverage), source_sha256={'electricity': digest(electricity), 'pedestrian': digest(pedestrian)})
        save('manifest.json', manifest)
        selected = {r['sensor_id']: r for r in selection['pedestrian']['validation']}
        start = FIRST_ORIGIN-2074*HOUR; expected_names = {r['sensor_id']: set(r['sensor_names']) for r in json.loads(coverage.read_text())['sensors']}
        positions, names, _ = collect(csv_rows(pedestrian), selected, start, END)
        coverage_rows = []
        for sensor in selected:
            coverage_rows.append({'sensor_id': sensor, 'missing_warmup_hours': 2074-len(set(positions[sensor]) & set(range(2074))),
                'missing_scored_hours': len(set(range(1344, 6298))-set(positions[sensor])),
                'duplicate_hours': sum(n != 1 for n in positions[sensor].values()), 'names': sorted(names[sensor]),
                'identity_matches': names[sensor] == expected_names[sensor],
                'missing_positions': sorted(set(range(6298))-set(positions[sensor]))})
        save('metadata-coverage.json', coverage_rows)
        if any(r['duplicate_hours'] or r['missing_scored_hours'] or not r['identity_matches'] for r in coverage_rows):
            raise ValueError('Fixed validation source identity/grid failed; no replacement or imputation')
        save('status.json', {'phase': 'metadata_passed', 'seconds': time.monotonic()-started})
        _, _, raw = collect(csv_rows(pedestrian), selected, start, END, counts=True)
        scored = {}; warmup = {}
        for sensor, row in selected.items():
            key = 'pedestrian:'+row['series_name']
            values = [raw[sensor].get(i) for i in range(6298)]
            scored[key], warmup[key] = split_span(key, values, start, SHA, 'pedestrians_per_source_hour', row['initial_history_sha256'])
        wanted = {r['series_name']: r for r in selection['electricity']['validation']}
        for name, begin, tokens in source_rows(electricity):
            if name not in wanted:continue
            row = wanted[name]; low = row['first_origin_index']-2074; high = row['end_index']
            if low < 0 or high > len(tokens) or high-low != 6298:raise ValueError('Fixed electricity span unavailable')
            key = 'electricity:'+name
            scored[key], warmup[key] = split_span(key, tokens[low:high], begin+low*HOUR,
                SOURCES['electricity']['sha'], SOURCES['electricity']['unit'], row['initial_history_sha256'])
        expected_ids = {d+':'+r['series_name'] for d in selection for r in selection[d]['validation']}
        if set(scored) != expected_ids or set(warmup) != expected_ids or len(scored) != 16:raise ValueError('Fixed identities incomplete')
        tasks = [t for key, span in sorted(warmup.items()) for t in boundaries(key, span)]
        save('scored-spans.json', scored); save('warmup-spans.json', warmup); save('warmup-boundaries.json', tasks)
        save('COMPLETED.json', {'validation_series': 16, 'planned_scored_cases': 416, 'warmup_attempts': 128,
            'usable_warmup_cases': sum(r['ready'] for r in tasks), 'unavailable_warmup_cases': sum(not r['ready'] for r in tasks),
            'scored_spans_sha256': digest(output/'scored-spans.json'), 'warmup_spans_sha256': digest(output/'warmup-spans.json'),
            'reserved_future_values_parsed': 0, 'forecast_computations': 0, 'api_calls': 0, 'seconds': time.monotonic()-started})
        save('status.json', {'phase': 'complete', 'seconds': time.monotonic()-started})
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'seconds': time.monotonic()-started,
                            'forecast_computations': 0, 'api_calls': 0})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('electricity', 'pedestrian', 'identity', 'coverage', 'output'):p.add_argument(name)
    prepare(**vars(p.parse_args()))
