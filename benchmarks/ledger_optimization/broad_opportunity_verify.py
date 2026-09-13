"""Enumerate admissible execution sets independently of opportunity implementation."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


def audit(original, calibration, directory):
    original, calibration, directory = Path(original), Path(calibration), Path(directory)
    output = directory/'verification.json'
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((directory/'report.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    here = Path(__file__).parent
    checks = 0
    def check(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(message)
    def near(a, b):
        check(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), 'numeric mismatch')
    for name, sha in manifest['source_receipts_sha256'].items():
        check(hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest() == sha, 'receipt changed')
    receipt = json.loads((here/'evidence/broad-screen-038.json').read_text())
    receipt39 = json.loads((here/'evidence/broad-calibration-039.json').read_text())
    check(hashlib.sha256(calibration.read_bytes()).hexdigest() == receipt39['files']['report.json'], 'calibration changed')
    choices = {(r['series_id'], r['round']): r['selection']['provider'] for r in json.loads(calibration.read_text())['rows']}
    sources = {}
    for name, sha in receipt['files'].items():
        if not name.startswith(('electricity:', 'pedestrian:')):
            continue
        raw = (original/name).read_bytes()
        check(hashlib.sha256(raw).hexdigest() == sha, 'source changed')
        row = json.loads(raw); sources[(row['series_id'], row['round'])] = row
    check(len(report['rows']) == len(sources) == len(choices) == 416, 'complete cohort')
    seen = set()
    for row in report['rows']:
        key = row['series_id'], row['round']
        check(key not in seen, 'duplicate'); seen.add(key)
        old = sources[key]; selection = old['selection']; control = selection['cv']
        past = selection['past']; blended = selection['blended']; calibrated = choices[key]
        now = datetime.fromisoformat(old['origin'])
        count = len([r for r in sources.values() if r['series_id'] == key[0]
                     and datetime.fromisoformat(r['origin']) < now
                     and datetime.fromisoformat(r['last_target']) <= now])
        check(row['matured_origins'] == count, 'visibility')
        allowed = {'cv': {control}, 'all_six_oracle': set(old['scores']),
                   'past_perfect_filter': {control, past}, 'blended_perfect_filter': {control, blended},
                   'calibrated_perfect_filter': {control, calibrated},
                   'proposal_union_oracle': {control, past, blended, calibrated}}
        for minimum in (3, 4, 8, 10):
            allowed[f'all_six_after_{minimum}_origins'] = set(old['scores']) if count >= minimum else {control}
        check(set(row['bounds']) == set(allowed), 'all frozen diagnostics')
        for name, models in allowed.items():
            near(row['bounds'][name], sorted(old['scores'][m] for m in models)[0])
        check(row['bounds']['all_six_oracle'] <= row['bounds']['proposal_union_oracle'] <= row['bounds']['cv'], 'nested feasible sets')
    def summary(observed, rows):
        means = {name: math.fsum(r['bounds'][name] for r in rows)/len(rows) for name in rows[0]['bounds']}
        check(observed['cases'] == len(rows), 'case count')
        for name, value in means.items():
            near(observed['mean_rmsle'][name], value)
            near(observed['relative_reduction'][name], (means['cv']-value)/means['cv'])
            check(observed['twenty_percent_feasible_within_each_bound'][name] == (value <= .8*means['cv']), 'feasibility')
        near(observed['target_rmsle'], .8*means['cv'])
        near(observed['oracle_gain_capture_required'], .2*means['cv']/(means['cv']-means['all_six_oracle']))
        near(observed['remaining_regret_allowed_at_target'], .8*means['cv']-means['all_six_oracle'])
    summary(report['overall'], report['rows'])
    for source in ('electricity', 'pedestrian'):
        summary(report['domains'][source], [r for r in report['rows'] if r['series_id'].startswith(source+':')])
    result = {'checks': checks, 'failures': 0, 'cases': 416,
              'report_sha256': hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Enumerated permitted execution sets for every case, visibility, unchanged source hashes, aggregates and target feasibility. Nondeployable hindsight diagnostic only.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'calibration', 'directory'):
        parser.add_argument(name)
    args = parser.parse_args()
    print(json.dumps(audit(args.original, args.calibration, args.directory), indent=2))
