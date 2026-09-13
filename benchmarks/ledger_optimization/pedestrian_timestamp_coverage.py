"""Inspect timestamp/identity coverage in the pinned publisher CSV, never counts."""
import argparse
import calendar
from collections import Counter
import csv
from datetime import datetime,timedelta
import hashlib
import io
import json
from pathlib import Path
import zipfile

SHA='5fa1d8fd8a50b0b2eededb85149a541336c2cfe1ab53706a0dbb1e81a526bc8a'
END=datetime(2020,5,1)
FIRST_ORIGIN=END-timedelta(hours=24+25*168)
START=FIRST_ORIGIN-timedelta(hours=730)
MONTHS={name:i for i,name in enumerate(calendar.month_name) if name}


def timestamp(row):
    value=datetime(int(row['Year']),MONTHS[row['Month']],int(row['Mdate']),int(row['Time']))
    if value.strftime('%B %d, %Y %I:%M:%S %p')!=row['Date_Time']:
        raise ValueError('Redundant source timestamp fields disagree')
    if value.strftime('%A')!=row['Day']:
        raise ValueError('Source weekday disagrees')
    return value


def audit(archive_path,output):
    archive_path,output=Path(archive_path),Path(output)
    if output.exists():raise FileExistsError(output)
    with archive_path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=SHA:raise ValueError('Source mismatch')
    positions={};names={};rows_seen=0;in_window=0
    with zipfile.ZipFile(archive_path) as archive:
        files=[n for n in archive.namelist() if n.endswith('.csv')]
        if len(files)!=1:raise ValueError('Expected one source CSV')
        with archive.open(files[0]) as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw,encoding='utf-8-sig')):
                rows_seen+=1
                # Numeric conversions below are date/identity metadata only.
                if row['Year'] not in ('2019','2020'):continue
                at=timestamp(row)
                if not START<=at<END:continue
                sensor=int(row['Sensor_ID']);index=int((at-START).total_seconds()/3600)
                positions.setdefault(sensor,Counter())[index]+=1
                names.setdefault(sensor,set()).add(row['Sensor_Name'])
                in_window+=1
                # Do not read, convert, summarize or compare Hourly_Counts.
    expected=int((END-START).total_seconds()/3600);assert expected==4954
    results=[]
    for sensor,counts in sorted(positions.items()):
        missing=sorted(set(range(expected))-set(counts))
        duplicate=sorted(index for index,n in counts.items() if n>1)
        bitmap=''.join('1' if i in counts else '0' for i in range(expected))
        results.append({'sensor_id':sensor,'sensor_names':sorted(names[sensor]),
            'distinct_hours':len(counts),'missing_hours':len(missing),'duplicate_hour_labels':len(duplicate),
            'duplicate_rows':sum(n-1 for n in counts.values()),
            'first_missing_labels':[(START+timedelta(hours=i)).isoformat() for i in missing[:8]],
            'duplicate_labels':[(START+timedelta(hours=i)).isoformat() for i in duplicate],
            'coverage_bitmap':bitmap,'complete_unique_coverage':not missing and not duplicate})
    result={'source_sha256':SHA,'rows_transited':rows_seen,'rows_in_window':in_window,
            'start_label':START.isoformat(),'end_label_exclusive':END.isoformat(),
            'first_origin':FIRST_ORIGIN.isoformat(),'expected_hours':expected,
            'sensors_in_window':len(results),'sensors_complete_unique':sum(r['complete_unique_coverage'] for r in results),
            'count_fields_parsed':0,'provider_calls':0,'sensors':results,
            'limitation':'Naive source-hour labels; coverage does not establish DST interpretation, count validity, or historical publication times.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive');parser.add_argument('output');args=parser.parse_args()
    result=audit(args.archive,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='sensors'},indent=2))
