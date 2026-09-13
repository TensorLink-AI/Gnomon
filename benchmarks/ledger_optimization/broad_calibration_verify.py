"""Independent arithmetic and evidence lineage audit for calibration screen 039."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

ORDER = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')


def audit(source, result_path, output):
    source, result_path, output = Path(source), Path(result_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(result_path.read_text())
    manifest = json.loads(result_path.with_name('manifest.json').read_text())
    receipt_path = Path(__file__).with_name('evidence')/'broad-screen-038.json'
    receipt = json.loads(receipt_path.read_text())
    checks = 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    def near(a, b, message):
        check(math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), message)

    def average(values):
        return math.fsum(values)/len(values)

    check(hashlib.sha256(receipt_path.read_bytes()).hexdigest() == manifest['source_receipt_sha256'], 'source receipt')
    for name, key in (('broad_calibration.py', 'code_sha256'), ('BROAD_CALIBRATION_039.md', 'protocol_sha256')):
        check(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == manifest[key], 'frozen '+name)
    originals = {}
    for name, sha in receipt['files'].items():
        if not name.startswith(('electricity:', 'pedestrian:')):
            continue
        raw = (source/name).read_bytes()
        check(hashlib.sha256(raw).hexdigest() == sha, 'immutable original case')
        row = json.loads(raw)
        originals[(row['series_id'], row['round'])] = row
    check(len(originals) == len(report['rows']) == 416, 'all cases retained')
    seen = set()
    for result in report['rows']:
        key = result['series_id'], result['round']
        check(key not in seen, 'unique case'); seen.add(key)
        row = originals[key]
        now = datetime.fromisoformat(row['origin'])
        eligible = sorted([r for r in originals.values() if r['series_id'] == key[0]
            and datetime.fromisoformat(r['origin']) < now
            and datetime.fromisoformat(r['last_target']) <= now], key=lambda r: r['origin'])[-8:]
        # This replay assumes outcomes recorded at horizon close, fixed in 037.
        selected = result['selection']
        check(selected['retrieved_origins'] == [r['origin'] for r in eligible], 'visible same-series origins')
        check(selected['matched_origins'] == len(eligible), 'support count')
        weight = len(eligible)/(len(eligible)+4) if len(eligible) >= 4 else 0.
        near(selected['shrinkage_weight'], weight, 'shrinkage')
        estimates = {}
        for m in ORDER:
            bias = average([r['scores'][m]-r['cv'][m] for r in eligible]) if weight else 0.
            near(selected['bias'][m], bias, 'bias')
            estimates[m] = max(0., row['cv'][m]+weight*bias)
            near(selected['estimated_rmsle'][m], estimates[m], 'corrected estimate')
        control = min(ORDER, key=lambda m: (row['cv'][m], ORDER.index(m)))
        chosen = min(ORDER, key=lambda m: (estimates[m], row['cv'][m], ORDER.index(m)))
        check(selected['provider'] == chosen and selected['cv_provider'] == control, 'choice')
        near(result['cv_rmsle'], row['scores'][control], 'control score')
        near(result['calibrated_rmsle'], row['scores'][chosen], 'chosen score')
        near(result['historical_rmsle'], row['scores'][row['selection']['past']], 'original historical score')
        check(result['historical_changed'] == (row['selection']['past'] != control), 'override flag')

    def summary(value, rows):
        check(value['cases'] == len(rows), 'summary cases')
        baseline = average([r['cv_rmsle'] for r in rows])
        calibrated = average([r['calibrated_rmsle'] for r in rows])
        near(value['cv_rmsle'], baseline, 'summary control')
        near(value['calibrated_rmsle'], calibrated, 'summary calibrated')
        near(value['relative_reduction'], 1-calibrated/baseline, 'relative reduction')
        check(value['calibrated_overrides'] == sum(r['selection']['provider'] != r['selection']['cv_provider'] for r in rows), 'calibrated overrides')
        deltas = [r['cv_rmsle']-r['historical_rmsle'] for r in rows if r['historical_changed']]
        original = value['original_historical_overrides']
        for k, v in {'count': len(deltas), 'helped': sum(d > 0 for d in deltas),
                     'hurt': sum(d < 0 for d in deltas), 'tied': sum(d == 0 for d in deltas)}.items():
            check(original[k] == v, 'original override '+k)
        near(original['total_rmsle_saved'], math.fsum(d for d in deltas if d > 0), 'saved losses')
        near(original['total_rmsle_added'], -math.fsum(d for d in deltas if d < 0), 'added losses')

    rows = report['rows']
    summary(report['overall'], rows)
    for domain in ('electricity', 'pedestrian'):
        summary(report['domains'][domain], [r for r in rows if r['series_id'].startswith(domain+':')])
        cases = [r for r in originals.values() if r['series_id'].startswith(domain+':')]
        for m in ORDER:
            item = report['original_recipe_cv_vs_actual'][domain][m]
            near(item['mean_cv_rmsle'], average([r['cv'][m] for r in cases]), 'recipe CV mean')
            near(item['mean_actual_rmsle'], average([r['scores'][m] for r in cases]), 'recipe outcome mean')
    summary(report['early_development_round_0_17'], [r for r in rows if r['round'] < 18])
    summary(report['later_development_round_18_25'], [r for r in rows if r['round'] >= 18])
    summary(report['mature_round_ge_10'], [r for r in rows if r['round'] >= 10])
    check(report['development_gate_passed'] == (report['overall']['relative_reduction'] >= .2 and
          all(r['relative_reduction'] > 0 for r in report['domains'].values())), 'frozen gate')
    check(report['additional_forecasts'] == report['api_calls'] == report['reserved_future_reads'] == 0, 'no new fits or final access')
    result = {'checks': checks, 'failures': 0, 'cases': len(rows),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'report_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'scope': 'Original evidence hashes, all visible same-series retrievals, bias corrections, choices, override counts and aggregates. Uses independently audited 038 forecast scores; no model refits.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'report', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.report, args.output), indent=2))
