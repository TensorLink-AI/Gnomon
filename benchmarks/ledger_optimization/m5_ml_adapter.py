"""Prepare the fixed M5 development panel for equal 730-row ML requests.

No model fitting, API requests, or reserved-target export. The final gate remains
closed. Operational preparation requires the original pinned manifest/archive.
"""
import argparse
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
from math import cos, isfinite, pi, sin
from pathlib import Path
from zipfile import ZipFile

from . import m5_prepare

MANIFEST_SHA = '6343970b713d8ccc49c5f3d6a633e3c876511ef4fd16c19f832d34a792243dd9'
HISTORY = 730
HORIZON = 14
ROUNDS = 26
FIRST = m5_prepare.CUTOFF - HISTORY + 1
LAST = m5_prepare.LAST
COVARIATES = ['onpromotion', 'dow_sin', 'dow_cos']


def value_hash(values):
    return hashlib.sha256(json.dumps(values, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def period_end(day):
    return datetime.combine(day + timedelta(days=1), time.min, tzinfo=timezone.utc)


def covariates(times):
    return [[0.0, sin(2*pi*(t-timedelta(days=1)).weekday()/7),
             cos(2*pi*(t-timedelta(days=1)).weekday()/7)] for t in times]


def build_development_jobs(source_rows, calendar, manifest):
    """Read numerical sales only for fixed development identities; no reselection."""
    selected = manifest['splits']['development']
    reserved = manifest['splits']['reserved']
    if len(selected) != 8 or len(reserved) != 24:
        raise ValueError('Require the original 8 development and 24 reserved identities')
    identities = {(s['store_id'],s['item_id']):s for s in selected}
    if len(identities) != 8 or len({s['series_id'] for s in selected}) != 8:
        raise ValueError('Duplicate development identity')
    for field in ('store_id','item_id','series_id'):
        if {s[field] for s in selected} & {s[field] for s in reserved}:
            raise ValueError('Development and reserved identities overlap: ' + field)
    days = [calendar[f'd_{i}'] for i in range(FIRST,LAST+1)]
    if any(b-a != timedelta(days=1) for a,b in zip(days,days[1:])):
        raise ValueError('Calendar must be complete and consecutive')
    stamps = [period_end(day) for day in days]
    result = {}
    for row in source_rows:
        identity = row['store_id'],row['item_id']
        if identity not in identities:
            continue  # Do not access or convert this row's sales cells.
        metadata = identities[identity]
        series = metadata['series_id']
        if series in result:
            raise ValueError('Duplicate selected source row: ' + series)
        try:
            values = [float(row[f'd_{i}']) for i in range(FIRST,LAST+1)]
        except (ValueError,TypeError,KeyError) as exc:
            raise ValueError('Invalid or missing development history/target; no replacement: ' + series) from exc
        if any(not isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid development sales; no replacement: ' + series)
        original_prefix = values[m5_prepare.FIRST-FIRST:m5_prepare.CUTOFF-FIRST+1]
        if value_hash(original_prefix) != metadata['initial_history_sha256']:
            raise ValueError('Original selection-prefix hash changed: ' + series)
        jobs = []
        for number in range(ROUNDS):
            start = number*HORIZON
            end = start+HISTORY
            history_times, future_times = stamps[start:end], stamps[end:end+HORIZON]
            origin = history_times[-1].isoformat()
            future = [t.isoformat() for t in future_times]
            if len(history_times)!=HISTORY or len(future)!=HORIZON or future_times[0]<=history_times[-1]:
                raise ValueError('Incomplete or nonprospective task')
            request = {'history':values[start:end], 'horizon':HORIZON, 'season':7, 'frequency':'D',
                'cutoff':origin, 'known_time_cutoff':origin, 'recorded_time_cutoff':origin,
                'timestamps':[t.isoformat() for t in history_times], 'future_timestamps':future,
                'past_covariates':covariates(history_times), 'future_covariates':covariates(future_times),
                'past_covariate_names':list(COVARIATES), 'future_covariate_names':list(COVARIATES),
                'series_id':series, 'unit':'unit_sales'}
            jobs.append({'series_id':series, 'round':number, 'origin':origin,
                'future_timestamps':future, 'outcome_recorded_at':future[-1],
                'request':request, 'actual':values[end:end+HORIZON]})
        result[series] = jobs
    if set(result) != {s['series_id'] for s in selected}:
        raise ValueError('Missing selected source series; no replacement')
    return {s:result[s] for s in sorted(result)}


def prepare_development(archive, manifest_path, output):
    if output.exists():
        raise ValueError('Refuse to overwrite a preparation directory')
    if m5_prepare.digest(manifest_path) != MANIFEST_SHA:
        raise ValueError('Only the original pinned M5 development manifest is allowed')
    manifest = json.loads(manifest_path.read_text())
    source_sha = m5_prepare.verify_archive(archive)
    if source_sha != manifest['source']['sha256']:
        raise ValueError('Archive differs from original preparation')
    with ZipFile(archive) as zipped:
        calendar = m5_prepare.calendar_mapping(m5_prepare.rows(zipped,'calendar.csv'))
        jobs = build_development_jobs(m5_prepare.rows(zipped,'sales_train_evaluation.csv'),calendar,manifest)
    output.mkdir(parents=True,exist_ok=False)
    path=output/'development-jobs.json'
    path.write_text(json.dumps(jobs,separators=(',',':'),allow_nan=False)+'\n')
    receipt={'scope':'development_only_730_row_ml_adapter','series':8,'tasks':208,
        'history':HISTORY,'horizon':HORIZON,'rounds':ROUNDS,'first_history_day':FIRST,
        'first_origin_day':m5_prepare.CUTOFF,'last_target_day':LAST,
        'manifest_sha256':MANIFEST_SHA,'source_sha256':source_sha,
        'jobs_sha256':m5_prepare.digest(path),'selection_changed':False,
        'reserved_targets_numeric_inspection':False,'provider_calls':0,'engy_calls':0,
        'availability':'Assumed period-end source and recording availability; complete scores mature at horizon close.',
        'promotion':'unavailable_assumed_zero; not an observed absence of promotions',
        'source_files':{p.name:m5_prepare.digest(p) for p in (
            Path(__file__),Path(m5_prepare.__file__),Path(__file__).with_name('M5_ML_ADAPTER.md'))}}
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare_development(args.archive,args.manifest,args.output),indent=2))
