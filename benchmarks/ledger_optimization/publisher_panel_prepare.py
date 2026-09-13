"""Prepare panel 037 without interpreting reserved later count values."""
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .broad_panel_prepare import SOURCES, eligible, numbers, partition, source_rows
from .pedestrian_timestamp_coverage import SHA, MEMBER, START, FIRST_ORIGIN, timestamp


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def csv_rows(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open(MEMBER) as raw:
            yield from csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig'))


def selected_counts(rows, sensors, hours):
    """Only access Hourly_Counts after selecting identity and time range."""
    selected = {sensor: {} for sensor in sensors}
    end = START + timedelta(hours=hours)
    for row in rows:
        if row['Year'] not in ('2019', '2020'):
            continue
        sensor = int(row['Sensor_ID'])
        if sensor not in selected:
            continue
        at = timestamp(row)
        if not START <= at < end:
            continue
        index = int((at - START).total_seconds() / 3600)
        if index in selected[sensor]:
            raise ValueError('Duplicate selected hour')
        selected[sensor][index] = row['Hourly_Counts']
    for values in selected.values():
        if set(values) != set(range(hours)):
            raise ValueError('Selected source coverage disagrees with audit')
    return {sensor: [values[i] for i in range(hours)] for sensor, values in selected.items()}


def prepare(electricity, pedestrian, coverage, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    save('protocol.json', {
        'code_sha256': digest(__file__),
        'protocol_sha256': digest(Path(__file__).with_name('BROAD_PANEL_037.md')),
        'helpers_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                           ('broad_panel_prepare.py', 'pedestrian_timestamp_coverage.py')},
        'provider_calls': 0, 'reserved_later_values_parsed': 0,
        'availability': 'assumed_at_nominal_period_end',
        'time_coordinate': 'UTC_surrogate_for_naive_hour_labels_not_physical_DST_time',
    })
    try:
        if digest(electricity) != SOURCES['electricity']['sha'] or digest(pedestrian) != SHA:
            raise ValueError('Pinned source checksum mismatch')
        receipt = json.loads(Path(__file__).with_name('evidence').joinpath('pedestrian-source-036.json').read_text())
        if digest(coverage) != receipt['coverage_receipt_sha256']:
            raise ValueError('Timestamp audit changed')
        audit = json.loads(Path(coverage).read_text())
        if audit['source_sha256'] != SHA or audit['start_label'] != START.isoformat():
            raise ValueError('Timestamp audit identity mismatch')
        sensors = [r['sensor_id'] for r in audit['sensors']
                   if r['complete_unique_coverage'] and len(r['sensor_names']) == 1]
        prefixes = selected_counts(csv_rows(pedestrian), sensors, 730)
        accepted, rejected = [], []
        for sensor, tokens in sorted(prefixes.items()):
            name = f'sensor_{sensor}'
            row, reason = eligible(name, START, tokens + ['SEALED'] * 4224,
                                   datetime(2020, 5, 1))
            if row is None:
                rejected.append({'series_name': name, 'reason': reason})
            else:
                row['sensor_id'] = sensor
                accepted.append(row)
        save('pedestrian-eligibility.json', {'eligible': accepted, 'rejected': rejected,
            'metadata_excluded': [r['sensor_id'] for r in audit['sensors'] if r['sensor_id'] not in sensors]})
        dev, reserved = partition('pedestrian', accepted)
        selections = {'pedestrian': {'development': dev, 'reserved': reserved}}
        accepted, rejected = [], []
        for name, start, tokens in source_rows(electricity):
            row, reason = eligible(name, start, tokens, datetime(2015, 1, 1))
            if row is None:
                rejected.append({'series_name': name, 'reason': reason})
            else:
                accepted.append(row)
        save('electricity-eligibility.json', {'eligible': accepted, 'rejected': rejected})
        dev, reserved = partition('electricity', accepted)
        selections['electricity'] = {'development': dev, 'reserved': reserved}
        save('selection.json', selections)  # Freeze identities before later values.
        spans = {}
        dev = selections['pedestrian']['development']
        tokens_by_sensor = selected_counts(csv_rows(pedestrian), [r['sensor_id'] for r in dev], 4954)
        for row in dev:
            values = numbers(tokens_by_sensor[row['sensor_id']])
            spans[f'pedestrian:{row["series_name"]}'] = {
                'source': 'pedestrian_publisher_036', 'source_sha256': SHA,
                'start_label': START.isoformat(), 'first_origin': FIRST_ORIGIN.isoformat(),
                'unit': 'pedestrians_per_source_hour', 'values': values}
        chosen = {r['series_name']: r for r in selections['electricity']['development']}
        for name, start, tokens in source_rows(electricity):
            if name not in chosen:
                continue
            row = chosen[name]
            low, first, high = (row[k] for k in ('first_history_index', 'first_origin_index', 'end_index'))
            spans[f'electricity:{name}'] = {
                'source': 'electricity_tsf_034', 'source_sha256': SOURCES['electricity']['sha'],
                'start_label': (start + timedelta(hours=low)).isoformat(),
                'first_origin': (start + timedelta(hours=first)).isoformat(),
                'unit': SOURCES['electricity']['unit'], 'values': numbers(tokens[low:high])}
        assert len(spans) == 16 and all(len(r['values']) == 4954 for r in spans.values())
        save('development-spans.json', spans)
        save('COMPLETED.json', {'development_series': 16, 'planned_development_cases': 416,
                               'reserved_series': 32, 'reserved_later_values_parsed': 0,
                               'provider_calls': 0, 'development_sha256': digest(output / 'development-spans.json')})
    except BaseException as error:
        save('FAILED.json', {'error': type(error).__name__, 'message': str(error), 'provider_calls': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('electricity', 'pedestrian', 'coverage', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    prepare(args.electricity, args.pedestrian, args.coverage, args.output)
