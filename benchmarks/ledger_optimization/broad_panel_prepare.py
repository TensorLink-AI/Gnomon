"""Prepare a prospectively fixed development panel; reserve future values unread."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
from pathlib import Path
import zipfile

SOURCES={
 'electricity':{'file':'electricity_hourly_dataset.zip','sha':'eff447075dde68dca0105ab7e2851c5637967ae3bb21556fd8b931f196d5968c','end':'2015-01-01T00:00:00','unit':'published_hourly_electricity_value'},
 'pedestrian':{'file':'pedestrian_counts_dataset.zip','sha':'6e81cb8cad43650e7e0f754a1103fc75c960e03387bae9b91146aa3241c9aa50','end':'2020-05-01T00:00:00','unit':'pedestrians_per_source_hour'},
}
HOUR=timedelta(hours=1)


def layout(start,end):
    last_origin=end-timedelta(hours=24)
    first_origin=last_origin-timedelta(hours=25*168)
    offset=(first_origin-start)/HOUR
    if int(offset)!=offset:raise ValueError('Nonhourly source start alignment')
    return int(offset)-730,int(offset),int((end-start)/HOUR)


def numbers(tokens):
    values=[float(v) for v in tokens]
    if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('Nonfinite or negative observation')
    return values


def eligible(name,start,tokens,end):
    low,origin,high=layout(start,end)
    if low<0 or high>len(tokens):return None,'insufficient_position_coverage'
    try:prefix=numbers(tokens[low:origin])
    except ValueError:return None,'invalid_initial_history'
    if sum(v>0 for v in prefix)<28:return None,'fewer_than_28_nonzero_initial_observations'
    return {'series_name':name,'start_label':start.isoformat(),'row_length':len(tokens),
            'first_history_index':low,'first_origin_index':origin,'end_index':high,
            'initial_history_sha256':hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()},None


def partition(source,rows):
    if len({r['series_name'] for r in rows})!=len(rows):raise ValueError('Duplicate eligible source identity')
    if len(rows)<24:raise ValueError('Fewer than 24 eligible series; protocol cannot be relaxed')
    ordered=sorted(rows,key=lambda r:hashlib.sha256(f'20260914:panel035:{source}:{r["series_name"]}'.encode()).hexdigest())
    return ordered[:8],ordered[8:24]


def source_rows(path):
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if n.endswith('.tsf')]
        if len(names)!=1:raise ValueError('Expected one TSF member')
        with archive.open(names[0]) as stream:
            header=[]
            while True:
                line=stream.readline(8193)
                if not line or len(line)>8192:raise ValueError('Invalid TSF header')
                header.append(line.decode('cp1252').strip())
                if header[-1]=='@data':break
                if len(header)>200:raise ValueError('Oversized TSF header')
            for required in ['@attribute series_name string','@attribute start_timestamp date','@frequency hourly']:
                if required not in header:raise ValueError('Unexpected source schema')
            seen=set()
            for raw in stream:
                if len(raw)>16*1024*1024:raise ValueError('Oversized source row')
                text=raw.decode('cp1252').strip()
                if not text or text.startswith('#'):continue
                name,stamp,values=text.split(':',2)
                if name in seen:raise ValueError('Duplicate source identity')
                seen.add(name)
                start=datetime.strptime(stamp,'%Y-%m-%d %H-%M-%S')
                yield name,start,values.split(',')


def prepare(source_root,output):
    source_root,output=Path(source_root),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    save('protocol.json',{'source_sha256':{k:v['sha'] for k,v in SOURCES.items()},
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256(Path(__file__).with_name('BROAD_PANEL_035.md').read_bytes()).hexdigest(),
        'observation_availability':'assumed_at_nominal_period_end',
        'timezone':'UTC_surrogate_for_naive_source_hour_positions_not_actual_source_timezone',
        'provider_calls':0,'final_values_exported':False})
    selections={};all_jobs={}
    try:
        for source,spec in SOURCES.items():
            path=source_root/spec['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha']:raise ValueError('Source checksum mismatch')
            end=datetime.fromisoformat(spec['end']);accepted=[];rejected=[]
            for name,start,tokens in source_rows(path):
                row,reason=eligible(name,start,tokens,end)
                if row is None:rejected.append({'series_name':name,'reason':reason})
                else:accepted.append(row)
            dev,reserved=partition(source,accepted)
            selections[source]={'eligible_count':len(accepted),'rejected':rejected,'development':dev,'reserved':reserved}
        save('selection.json',selections)  # All source IDs fixed before later-value parsing.
        for source,spec in SOURCES.items():
            path=source_root/spec['file'];dev=selections[source]['development']
            chosen={r['series_name']:r for r in dev}
            for name,start,tokens in source_rows(path):
                if name not in chosen:continue
                row=chosen[name];low,first,high=row['first_history_index'],row['first_origin_index'],row['end_index']
                values=numbers(tokens[low:high]);assert len(values)==4954
                series=f'{source}:{name}';jobs=[]
                for round_number in range(26):
                    stop=730+168*round_number
                    origin=(start+(low+stop)*HOUR).replace(tzinfo=timezone.utc)
                    timestamps=[(start+(low+i+1)*HOUR).replace(tzinfo=timezone.utc).isoformat() for i in range(stop-730,stop+24)]
                    # Features use nominal source-hour start labels, not invented publication metadata.
                    labels=[datetime.fromisoformat(t)-HOUR for t in timestamps]
                    cov=[[math.sin(2*math.pi*t.hour/24),math.cos(2*math.pi*t.hour/24),
                          math.sin(2*math.pi*t.weekday()/7),math.cos(2*math.pi*t.weekday()/7)] for t in labels]
                    req={'series_id':series,'unit':spec['unit'],'horizon':24,'frequency':'H','season':24,
                         'history':values[stop-730:stop],'timestamps':timestamps[:730],'future_timestamps':timestamps[730:],
                         'past_covariates':cov[:730],'future_covariates':cov[730:],
                         'past_covariate_names':['source_hour_sin','source_hour_cos','source_weekday_sin','source_weekday_cos'],
                         'future_covariate_names':['source_hour_sin','source_hour_cos','source_weekday_sin','source_weekday_cos'],
                         'cutoff':origin.isoformat(),'known_time_cutoff':origin.isoformat()}
                    jobs.append({'series_id':series,'round':round_number,'origin':origin.isoformat(),'request':req,
                                 'future_timestamps':timestamps[730:],'actual':values[stop:stop+24],
                                 'outcome_recorded_at':timestamps[-1]})
                all_jobs[series]=jobs
        assert len(all_jobs)==16 and sum(map(len,all_jobs.values()))==416
        save('development-jobs.json',all_jobs)
        save('COMPLETED.json',{'development_series':16,'development_cases':416,'reserved_series':32,
                             'provider_calls':0,'reserved_later_values_parsed':0,
                             'jobs_sha256':hashlib.sha256((output/'development-jobs.json').read_bytes()).hexdigest()})
    except BaseException as error:
        save('FAILED.json',{'error':type(error).__name__,'message':str(error),'provider_calls':0})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source_root');parser.add_argument('output')
    args=parser.parse_args();prepare(args.source_root,args.output)
