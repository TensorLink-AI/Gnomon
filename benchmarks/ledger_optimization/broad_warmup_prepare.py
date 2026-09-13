"""Prepare preperiod evidence for fixed development identities only."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from .publisher_panel_prepare import csv_rows, digest
from .pedestrian_timestamp_coverage import SHA, timestamp
from .broad_panel_prepare import SOURCES, numbers, source_rows

HOUR = timedelta(hours=1)


def collect(rows, selected, start, end, *, counts=False):
    positions = {s: Counter() for s in selected}
    names = {s: set() for s in selected}
    values = {s: {} for s in selected}
    for row in rows:
        if row['Year'] not in {str(start.year), str(end.year)}:
            continue
        sensor = int(row['Sensor_ID'])
        if sensor not in selected:
            continue
        at = timestamp(row)
        if not start <= at < end:
            continue
        i = int((at-start)/HOUR)
        positions[sensor][i] += 1
        names[sensor].add(row['Sensor_Name'])
        if counts:
            values[sensor][i] = row['Hourly_Counts']
    return positions, names, values


def prepare(electricity, pedestrian, panel, old_coverage, output):
    panel, old_coverage, output = Path(panel), Path(old_coverage), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': digest(__file__),
        'protocol_sha256': digest(here/'BROAD_WARMUP_041.md'),
        'helper_sha256': {n: digest(here/n) for n in ('publisher_panel_prepare.py', 'pedestrian_timestamp_coverage.py', 'broad_panel_prepare.py')},
        'forecast_computations': 0, 'api_calls': 0, 'reserved_counts_parsed': 0})
    try:
        receipt = json.loads((here/'evidence/broad-panel-037.json').read_text())
        receipt36 = json.loads((here/'evidence/pedestrian-source-036.json').read_text())
        for name in ('selection.json', 'development-spans.json'):
            if digest(panel/name) != receipt['files'][name]:
                raise ValueError('Original panel changed')
        if digest(old_coverage) != receipt36['coverage_receipt_sha256']:
            raise ValueError('Original timestamp audit changed')
        if digest(electricity) != SOURCES['electricity']['sha'] or digest(pedestrian) != SHA:
            raise ValueError('Raw source changed')
        selection = json.loads((panel/'selection.json').read_text())
        original = json.loads((panel/'development-spans.json').read_text())
        selected = {r['sensor_id']: r for r in selection['pedestrian']['development']}
        sample = original['pedestrian:'+next(iter(selected.values()))['series_name']]
        end = datetime.fromisoformat(sample['first_origin'])
        start = end-HOUR*2074
        positions, names, _ = collect(csv_rows(pedestrian), selected, start, end)
        expected_names = {r['sensor_id']: set(r['sensor_names']) for r in json.loads(old_coverage.read_text())['sensors']}
        coverage = []
        for sensor in selected:
            missing = sorted(set(range(2074))-positions[sensor].keys())
            duplicates = [i for i, n in positions[sensor].items() if n != 1]
            coverage.append({'sensor_id': sensor, 'missing_hours': len(missing),
                'duplicate_hours': len(duplicates), 'names': sorted(names[sensor]),
                'expected_names': sorted(expected_names[sensor]),
                'first_missing_labels': [(start+i*HOUR).isoformat() for i in missing[:8]],
                'ready': not missing and not duplicates and names[sensor] == expected_names[sensor]})
        save('pedestrian-coverage.json', {'start': start.isoformat(), 'end_exclusive': end.isoformat(),
                                        'expected_hours': 2074, 'sensors': coverage})
        if not all(r['ready'] for r in coverage):
            raise ValueError('Fixed pedestrian identities fail preperiod timestamp/identity coverage; no replacements')
        _, _, raw_values = collect(csv_rows(pedestrian), selected, start, end, counts=True)
        spans = {}
        for sensor, selected_row in selected.items():
            identity = 'pedestrian:'+selected_row['series_name']
            values = numbers([raw_values[sensor][i] for i in range(2074)])
            if values[-730:] != original[identity]['values'][:730]:
                raise ValueError('Pedestrian overlap changed')
            spans[identity] = {'source_sha256': SHA, 'start_label': start.isoformat(),
                'first_scored_origin': end.isoformat(), 'first_origin': (start+730*HOUR).isoformat(),
                'unit': original[identity]['unit'], 'values': values, 'original_overlap_matches': True}
        wanted = {r['series_name']: r for r in selection['electricity']['development']}
        for name, source_start, tokens in source_rows(electricity):
            if name not in wanted:
                continue
            identity = 'electricity:'+name
            last = wanted[name]['first_origin_index']; first = last-2074
            if first < 0 or last > len(tokens):
                raise ValueError('Electricity preperiod unavailable')
            values = numbers(tokens[first:last])
            if values[-730:] != original[identity]['values'][:730]:
                raise ValueError('Electricity overlap changed')
            spans[identity] = {'source_sha256': SOURCES['electricity']['sha'],
                'start_label': (source_start+first*HOUR).isoformat(),
                'first_origin': (source_start+(first+730)*HOUR).isoformat(),
                'first_scored_origin': original[identity]['first_origin'],
                'unit': original[identity]['unit'], 'values': values, 'original_overlap_matches': True}
        if len(spans) != 16:
            raise ValueError('Missing fixed development series')
        boundaries = []
        for identity, span in sorted(spans.items()):
            begin = datetime.fromisoformat(span['start_label'])
            for i in range(8):
                stop = 730+168*i
                origin = begin+stop*HOUR
                closes = origin+24*HOUR
                if closes >= datetime.fromisoformat(span['first_scored_origin']):
                    raise ValueError('Warm-up overlaps scored period')
                boundaries.append({'series_id': identity, 'round': i-8,
                    'origin': origin.isoformat(), 'last_target': closes.isoformat(),
                    'history_indices': [stop-730, stop], 'target_indices': [stop, stop+24]})
        save('warmup-spans.json', spans)
        save('boundaries.json', boundaries)
        save('COMPLETED.json', {'development_series': 16, 'warmup_tasks': 128,
            'scored_tasks_unchanged': 416, 'forecast_computations': 0, 'api_calls': 0,
            'reserved_counts_parsed': 0, 'spans_sha256': digest(output/'warmup-spans.json')})
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'forecast_computations': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('electricity', 'pedestrian', 'panel', 'old_coverage', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    prepare(args.electricity, args.pedestrian, args.panel, args.old_coverage, args.output)
