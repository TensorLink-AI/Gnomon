"""Read-only verification of displayed production comparisons in an audit snapshot."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def rmsle(point,actual):
    assert len(point)==len(actual)==14
    return math.sqrt(sum((math.log1p(p)-math.log1p(a))**2 for p,a in zip(point,actual,strict=True))/14)


def check_pair(pair,events,at):
    originals={e['execution']['execution_id']:e for e in events if e['event']=='result'}
    values={}
    for event in events:
        if event['event']!='matured' or datetime.fromisoformat(event['task_origin'])>at:continue
        original=originals[event['execution_id']]
        assert original['kind']=='forecast'
        request=original['request'];times=list(map(datetime.fromisoformat,request['future_timestamps']))
        assert datetime.fromisoformat(request['cutoff'])<min(times)<=max(times)<=at
        assert datetime.fromisoformat(event['outcome_recorded_at'])<=at
        assert original['point']==event['point']
        values[original['config_id'],request['cutoff']]=rmsle(event['point'],event['actual'])
    assert pair['n']==len(pair['matched_origins'])>0
    assert len(set(pair['matched_origins']))==pair['n']
    for side in ('left','right'):
        expected=mean(values[pair[side],origin] for origin in pair['matched_origins'])
        assert abs(expected-pair[side+'_rmsle'])<1e-12


def audit(snapshot):
    receipt=json.loads((snapshot/'snapshot-receipt.json').read_text())
    assert receipt['source_files_unchanged'] and not receipt['audit_failures']
    assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in receipt['source_hashes'].items())
    packets={};sources={}
    for relative in receipt['included']:
        parts=Path(relative).parts
        if parts[0]!='ledger':continue
        project=snapshot/relative/'project';path=project/'review-log.jsonl'
        if not path.exists():continue
        events=[json.loads(line) for line in (project/'experiments.jsonl').read_text().splitlines()]
        for index,line in enumerate(path.read_text().splitlines()):
            packet=json.loads(line);key=(parts[1],index)
            if key in packets:
                assert packets[key]==packet
                continue
            packets[key]=packet;sources[key]=events
    checked=[]
    for key,packet in packets.items():
        production=[p for p in packet['matched_comparisons'] if p['kind']=='forecast']
        for pair in production:check_pair(pair,sources[key],datetime.fromisoformat(packet['task_origin']))
        checked.append({'series_id':key[0],'packet_index':key[1],'origin':packet['task_origin'],
                        'production_comparisons_checked':len(production),
                        'displayed_comparisons':len(packet['matched_comparisons']),
                        'available_comparisons':packet['comparison_count'],
                        'production_matched_origin_counts':[p['n'] for p in production]})
    assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in receipt['source_hashes'].items())
    return {'passed':True,'snapshot':str(snapshot),'snapshot_sha256':hashlib.sha256((snapshot/'snapshot-receipt.json').read_bytes()).hexdigest(),
            'unique_review_packets':len(packets),'packets_with_production_comparisons':sum(r['production_comparisons_checked']>0 for r in checked),
            'production_comparisons_checked':sum(r['production_comparisons_checked'] for r in checked),
            'packets':checked,'source_unchanged':True,'provider_calls':0,'ledger_writes':0,
            'limits':['Partial development evidence; this establishes displayed comparison fidelity, not better decisions.',
                      'Repeated packet indices across cumulative snapshots are counted once.',
                      'No conclusions about untouched final data or the 20% target.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(args.snapshot)
    args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
