"""Independent geometry, identity and missingness audit of validation source053."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path


def audit(root, identity):
    root, identity = Path(root), Path(identity)
    read = lambda p: json.loads(p.read_text())
    done = read(root/'COMPLETED.json'); selected = read(identity/'selection.json')
    scored = read(root/'scored-spans.json'); warm = read(root/'warmup-spans.json')
    tasks = read(root/'warmup-boundaries.json'); checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:raise AssertionError(message)
    expected = {d+':'+r['series_name']: r for d, partition in selected.items() for r in partition['validation']}
    check(set(scored) == set(warm) == set(expected), 'Locked identities')
    for name in ('scored', 'warmup'):
        check(hashlib.sha256((root/(name+'-spans.json')).read_bytes()).hexdigest() == done[name+'_spans_sha256'], 'Span hash')
    task_map = {(t['series_id'], t['round']): t for t in tasks}
    check(len(task_map) == len(tasks) == 128, 'Unique warmup tasks')
    usable = 0
    for key, span in scored.items():
        values = span['values']; prior = warm[key]['values']
        check(len(values) == 4954 and len(prior) == 2074, 'Full geometry')
        check(all(isinstance(v, (int,float)) and math.isfinite(v) and v >= 0 for v in values), 'Scored finite observations')
        check(all(v is None or (isinstance(v, (int,float)) and math.isfinite(v) and v >= 0) for v in prior), 'Warm finite or missing')
        check(prior[-730:] == values[:730], 'Warmup overlap')
        check(hashlib.sha256(json.dumps(values[:730], separators=(',', ':')).encode()).hexdigest() == expected[key]['initial_history_sha256'], 'Frozen prefix')
        start = datetime.fromisoformat(span['start_label']); early = datetime.fromisoformat(warm[key]['start_label'])
        check(early+timedelta(hours=1344) == start, 'Warmup calendar offset')
        check(datetime.fromisoformat(span['first_origin']) == start+timedelta(hours=730), 'Origin')
        for i in range(26):
            stop = 730+168*i
            check(len(values[stop-730:stop]) == 730 and len(values[stop:stop+24]) == 24, 'Scored task slices')
        for i in range(8):
            t = task_map[key, i-8]; stop = 730+168*i
            mh = sum(v is None for v in prior[stop-730:stop]); mt = sum(v is None for v in prior[stop:stop+24])
            check(t['history_indices'] == [stop-730,stop] and t['target_indices'] == [stop,stop+24], 'Warm slices')
            check(t['missing_history'] == mh and t['missing_targets'] == mt and t['ready'] == (mh+mt == 0), 'Availability')
            check(datetime.fromisoformat(t['origin']) == early+timedelta(hours=stop), 'Warm origin')
            check(datetime.fromisoformat(t['last_target']) == early+timedelta(hours=stop+24) < start+timedelta(hours=730), 'Warm availability precedes scored origin')
            usable += t['ready']
    check(usable == done['usable_warmup_cases'] == 117, 'Usable warmup count')
    check(done['planned_scored_cases'] == 416 and done['forecast_computations'] == done['api_calls'] == 0, 'Preparation only')
    result = {'checks': checks, 'failures': 0, 'usable_warmup_cases': usable, 'scored_cases': 416,
        'scope': 'Identity, initial-prefix hashes, geometry, missingness, overlap and temporal boundaries; no model scores computed.'}
    (root/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('identity')
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))
