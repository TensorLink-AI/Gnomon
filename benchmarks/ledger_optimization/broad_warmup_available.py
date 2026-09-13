"""Explicitly incomplete earlier evidence; scored task cohort stays unchanged."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from .broad_warmup_prepare import collect
from .publisher_panel_prepare import csv_rows, digest
from .pedestrian_timestamp_coverage import SHA
from .broad_panel_prepare import SOURCES, numbers, source_rows

HOUR = timedelta(hours=1)


def boundaries(identity, span):
    start = datetime.fromisoformat(span['start_label'])
    rows = []
    for i in range(8):
        stop = 730+168*i
        history = span['values'][stop-730:stop]; target = span['values'][stop:stop+24]
        if len(history) != 730 or len(target) != 24:
            raise ValueError('Incomplete positional span')
        missing_history = sum(v is None for v in history); missing_target = sum(v is None for v in target)
        close = start+(stop+24)*HOUR
        if close >= datetime.fromisoformat(span['first_scored_origin']):
            raise ValueError('Warm-up target reaches scored period')
        rows.append({'series_id': identity, 'round': i-8, 'origin': (start+stop*HOUR).isoformat(),
            'last_target': close.isoformat(), 'history_indices': [stop-730, stop],
            'target_indices': [stop, stop+24], 'missing_history': missing_history,
            'missing_targets': missing_target, 'ready': missing_history+missing_target == 0})
    return rows


def prepare(electricity, pedestrian, panel, old_coverage, output):
    panel, old_coverage, output = Path(panel), Path(old_coverage), Path(output)
    output.mkdir(parents=True, exist_ok=False); here = Path(__file__).parent
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': digest(__file__),
        'protocol_sha256': digest(here/'BROAD_WARMUP_042.md'),
        'helpers_sha256': {n: digest(here/n) for n in ('broad_warmup_prepare.py', 'publisher_panel_prepare.py', 'pedestrian_timestamp_coverage.py', 'broad_panel_prepare.py')},
        'api_calls': 0, 'forecast_computations': 0, 'reserved_counts_parsed': 0})
    try:
        receipt = json.loads((here/'evidence/broad-panel-037.json').read_text())
        old_receipt = json.loads((here/'evidence/pedestrian-source-036.json').read_text())
        for name in ('selection.json', 'development-spans.json'):
            if digest(panel/name) != receipt['files'][name]:raise ValueError('Panel changed')
        if digest(old_coverage) != old_receipt['coverage_receipt_sha256']:raise ValueError('Original coverage changed')
        if digest(electricity) != SOURCES['electricity']['sha'] or digest(pedestrian) != SHA:raise ValueError('Source changed')
        selection = json.loads((panel/'selection.json').read_text())
        original = json.loads((panel/'development-spans.json').read_text())
        selected = {r['sensor_id']: r for r in selection['pedestrian']['development']}
        end = datetime.fromisoformat(original['pedestrian:'+next(iter(selected.values()))['series_name']]['first_origin'])
        start = end-2074*HOUR
        positions, names, _ = collect(csv_rows(pedestrian), selected, start, end)
        expected_names = {r['sensor_id']: set(r['sensor_names']) for r in json.loads(old_coverage.read_text())['sensors']}
        coverage = [{'sensor_id': s, 'missing_hours': 2074-len(positions[s]),
                     'duplicate_hours': sum(n != 1 for n in positions[s].values()),
                     'identity_matches': names[s] == expected_names[s]} for s in selected]
        save('metadata-coverage.json', coverage)
        if any(r['duplicate_hours'] or not r['identity_matches'] for r in coverage):
            raise ValueError('Ambiguous preperiod identity or duplicate hour')
        _, _, raw = collect(csv_rows(pedestrian), selected, start, end, counts=True)
        spans = {}
        for sensor, row in selected.items():
            identity = 'pedestrian:'+row['series_name']
            values = [numbers([raw[sensor][i]])[0] if i in raw[sensor] else None for i in range(2074)]
            if values[-730:] != original[identity]['values'][:730]:raise ValueError('Scored history overlap changed')
            spans[identity] = {'source_sha256': SHA, 'start_label': start.isoformat(),
                'first_origin': (start+730*HOUR).isoformat(), 'first_scored_origin': end.isoformat(),
                'unit': original[identity]['unit'], 'values': values, 'original_overlap_matches': True}
        wanted = {r['series_name']: r for r in selection['electricity']['development']}
        for name, begin, tokens in source_rows(electricity):
            if name not in wanted:continue
            identity = 'electricity:'+name; last = wanted[name]['first_origin_index']; first = last-2074
            if first < 0 or last > len(tokens):raise ValueError('Electricity preperiod unavailable')
            values = numbers(tokens[first:last])
            if values[-730:] != original[identity]['values'][:730]:raise ValueError('Scored history overlap changed')
            spans[identity] = {'source_sha256': SOURCES['electricity']['sha'], 'start_label': (begin+first*HOUR).isoformat(),
                'first_origin': (begin+(first+730)*HOUR).isoformat(), 'first_scored_origin': original[identity]['first_origin'],
                'unit': original[identity]['unit'], 'values': values, 'original_overlap_matches': True}
        if len(spans) != 16:raise ValueError('Missing fixed identity')
        tasks = [t for s, span in sorted(spans.items()) for t in boundaries(s, span)]
        save('boundaries.json', tasks); save('warmup-spans.json', spans)
        save('COMPLETED.json', {'development_series': 16, 'proposed_warmup_tasks': 128,
            'usable_warmup_tasks': sum(r['ready'] for r in tasks), 'unavailable_warmup_tasks': sum(not r['ready'] for r in tasks),
            'scored_tasks_unchanged': 416, 'reserved_counts_parsed': 0, 'forecast_computations': 0,
            'api_calls': 0, 'spans_sha256': digest(output/'warmup-spans.json')})
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'forecast_computations': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('electricity', 'pedestrian', 'panel', 'old_coverage', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    prepare(args.electricity, args.pedestrian, args.panel, args.old_coverage, args.output)
