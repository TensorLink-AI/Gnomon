"""Nondeployable bounds on fixed proposals and same-series cold starts."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time


def bounds(scores, control, proposals, matured_origins):
    if control not in scores or any(p not in scores for p in proposals.values()):
        raise ValueError('Unknown execution proposal')
    if any(not math.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError('Invalid scored error')
    baseline = scores[control]
    oracle = min(scores.values())
    result = {'cv': baseline, 'all_six_oracle': oracle}
    result.update({name+'_perfect_filter': min(baseline, scores[p]) for name, p in proposals.items()})
    result['proposal_union_oracle'] = min([baseline]+[scores[p] for p in proposals.values()])
    for minimum in (3, 4, 8, 10):
        result[f'all_six_after_{minimum}_origins'] = oracle if matured_origins >= minimum else baseline
    return result


def summarize(rows):
    means = {k: math.fsum(r['bounds'][k] for r in rows)/len(rows) for k in rows[0]['bounds']}
    baseline = means['cv']
    reductions = {k: (1-v/baseline if baseline else None) for k, v in means.items()}
    headroom = baseline-means['all_six_oracle']
    return {'cases': len(rows), 'mean_rmsle': means, 'relative_reduction': reductions,
            'twenty_percent_feasible_within_each_bound': {k: v <= .8*baseline for k, v in means.items()},
            'target_rmsle': .8*baseline,
            'oracle_gain_capture_required': .2*baseline/headroom if headroom > 0 else None,
            'remaining_regret_allowed_at_target': .8*baseline-means['all_six_oracle']}


def run(source, calibrated_path, output):
    source, calibrated_path, output = Path(source), Path(calibrated_path), Path(output)
    here = Path(__file__).parent
    original_receipt = json.loads((here/'evidence/broad-screen-038.json').read_text())
    calibrated_receipt = json.loads((here/'evidence/broad-calibration-039.json').read_text())
    if hashlib.sha256(calibrated_path.read_bytes()).hexdigest() != calibrated_receipt['files']['report.json']:
        raise ValueError('Calibration report changed')
    calibrated = json.loads(calibrated_path.read_text())
    choices = {(r['series_id'], r['round']): r['selection']['provider'] for r in calibrated['rows']}
    cases = []
    for name, sha in original_receipt['files'].items():
        if name.startswith(('electricity:', 'pedestrian:')):
            raw = (source/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('Original evidence changed')
            cases.append(json.loads(raw))
    if len(cases) != 416 or len(choices) != 416:
        raise ValueError('Expected all 416 development cases')
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('manifest.json', {'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256((here/'BROAD_OPPORTUNITY_040.md').read_bytes()).hexdigest(),
        'source_receipts_sha256': {name: hashlib.sha256((here/'evidence'/name).read_bytes()).hexdigest()
                                 for name in ('broad-screen-038.json', 'broad-calibration-039.json')},
        'additional_forecasts': 0, 'api_calls': 0, 'oracle_diagnostics_only': True})
    start = time.monotonic(); rows = []
    for case in sorted(cases, key=lambda r: (r['series_id'], r['origin'])):
        origin = datetime.fromisoformat(case['origin'])
        matured = sum(r['series_id'] == case['series_id'] and datetime.fromisoformat(r['origin']) < origin
                      and datetime.fromisoformat(r['last_target']) <= origin for r in cases)
        key = case['series_id'], case['round']
        proposals = {'past': case['selection']['past'], 'blended': case['selection']['blended'], 'calibrated': choices[key]}
        row = {'series_id': key[0], 'round': key[1], 'matured_origins': matured,
               'bounds': bounds(case['scores'], case['selection']['cv'], proposals, matured)}
        rows.append(row)
    result = {'overall': summarize(rows),
        'domains': {s: summarize([r for r in rows if r['series_id'].startswith(s+':')]) for s in ('electricity', 'pedestrian')},
        'seconds': time.monotonic()-start, 'additional_forecasts': 0, 'api_calls': 0,
        'inherited_forecast_computations': 9984, 'reserved_future_reads': 0,
        'limitation': 'Actual future error used deliberately. Fixed development portfolio hindsight diagnostics, not deployable rules or general impossibility bounds.',
        'rows': rows}
    save('report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source'); parser.add_argument('calibration'); parser.add_argument('output')
    args = parser.parse_args()
    r = run(args.source, args.calibration, args.output)
    print(json.dumps({k: v for k, v in r.items() if k != 'rows'}, indent=2))
