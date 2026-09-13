"""Independent saved-pair, slicing, baseline and selection audit for screen 038."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

ORDER = ('daily', 'weekly', 'weekly_mean', 'ridge_short', 'ridge_long', 'forest')
SOURCE_SHA = '60c8db1de333290d7cc04ca3a52dc4c4d46d8b93867105c5cb9be03adb8de15f'


def audit(source, results, output):
    source, results, output = Path(source), Path(results), Path(output)
    if output.exists():
        raise FileExistsError(output)
    checks = 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    def near(actual, expected, message):
        check(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), message)

    def metric(point, actual):
        check(len(point) == len(actual) == 24, 'scored horizon')
        check(all(math.isfinite(p) and p >= 0 for p in point), 'valid forecast counts')
        return (math.fsum((math.log1p(point[i])-math.log1p(actual[i]))**2 for i in range(24))/24)**.5

    def avg(values):
        return math.fsum(values)/len(values)

    check(hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA, 'development source hash')
    spans = json.loads(source.read_text())
    manifest = json.loads((results/'manifest.json').read_text())
    report = json.loads((results/'report.json').read_text())
    check(manifest['source_sha256'] == SOURCE_SHA, 'manifest source')
    check(tuple(manifest['recipes']) == ORDER, 'declared recipe order')
    for name, expected in manifest['code_sha256'].items():
        check(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected, 'frozen '+name)
    all_rows = []
    for series, span in sorted(spans.items()):
        prior = []
        start = datetime.fromisoformat(span['start_label']).replace(tzinfo=timezone.utc)
        values = span['values']
        for round_number in range(26):
            row = json.loads((results/f'{series}-{round_number:02d}.json').read_text())
            stop = 730+round_number*168; low = stop-730
            origin = start+timedelta(hours=stop)
            check(row['series_id'] == series and row['round'] == round_number, 'task identity')
            check(row['history_indices'] == [low, stop] and row['target_indices'] == [stop, stop+24], 'slices')
            check(row['origin'] == origin.isoformat(), 'origin period end')
            check(row['last_target'] == (origin+timedelta(hours=24)).isoformat(), 'target closure')
            check(row['actual'] == values[stop:stop+24], 'original targets')
            check(row['history_sha256'] == hashlib.sha256(json.dumps(values[low:stop], separators=(',', ':')).encode()).hexdigest(), 'history bytes')
            cv = {}; scores = {}
            for model in ORDER:
                check(len(row['folds'][model]) == 3, 'three matched folds')
                fold_scores = []
                for fold, end in zip(row['folds'][model], (658, 682, 706), strict=True):
                    check(fold['end'] == end, 'fold end')
                    actual = values[low+end:low+end+24]
                    check(fold['actual'] == actual, 'historical target slice')
                    score = metric(fold['point'], actual)
                    near(fold['rmsle'], score, 'fold metric')
                    fold_scores.append(score)
                    if model in ORDER[:3]:
                        for i, p in enumerate(fold['point']):
                            if model == 'weekly_mean':
                                expected = avg([values[low+end-168*k+i] for k in (1, 2, 3)])
                            else:
                                expected = values[low+end-({'daily':24, 'weekly':168}[model])+i]
                            near(p, expected, 'independent fold baseline')
                cv[model] = avg(fold_scores)
                near(row['cv'][model], cv[model], 'aggregate CV')
                scores[model] = metric(row['point'][model], values[stop:stop+24])
                near(row['scores'][model], scores[model], 'production metric')
                if model in ORDER[:3]:
                    for i, p in enumerate(row['point'][model]):
                        if model == 'weekly_mean':
                            expected = avg([values[stop-168*k+i] for k in (1, 2, 3)])
                        else:
                            expected = values[stop-({'daily':24, 'weekly':168}[model])+i]
                        near(p, expected, 'independent production baseline')
            choice = row['selection']
            control = min(ORDER, key=lambda m: (cv[m], ORDER.index(m)))
            recent = prior[-4:]
            check(all(datetime.fromisoformat(r['last_target']) <= origin for r in recent), 'outcome available by origin')
            check(choice['history_origins'] == [r['origin'] for r in recent], 'only prior same-series origins')
            check(choice['matched_origins'] == len(recent), 'matched evidence count')
            past = blended = control
            if len(recent) >= 3:
                means = {m: avg([r['scores'][m] for r in recent]) for m in ORDER}
                past = min(ORDER, key=lambda m: (means[m], cv[m], ORDER.index(m)))
                blended = min(ORDER, key=lambda m: ((means[m]+cv[m])/2, cv[m], ORDER.index(m)))
            for key, selected in (('cv', control), ('past', past), ('blended', blended)):
                check(choice[key] == selected, 'independent '+key+' selection')
            summary = {'series_id': series, 'round': round_number, 'selection': choice,
                       'cv_rmsle': scores[control], 'past_rmsle': scores[past],
                       'blended_rmsle': scores[blended], 'hindsight_rmsle': min(scores.values())}
            all_rows.append(summary)
            prior.append({'origin': row['origin'], 'last_target': row['last_target'], 'scores': scores})
    check(len(all_rows) == 416, 'all planned cases')
    for observed, expected in zip(report['rows'], all_rows, strict=True):
        check(observed['series_id'] == expected['series_id'] and observed['round'] == expected['round'], 'report identity')
        for k in ('cv', 'past', 'blended', 'hindsight'):
            near(observed[k+'_rmsle'], expected[k+'_rmsle'], 'report score')

    def summary_check(observed, rows):
        check(observed['cases'] == len(rows), 'aggregate case count')
        means = {k: avg([r[k+'_rmsle'] for r in rows]) for k in ('cv', 'past', 'blended', 'hindsight')}
        for k, value in means.items():
            near(observed['means'][k], value, 'aggregate '+k)
        for k in ('past', 'blended', 'hindsight'):
            near(observed[k+'_relative_reduction'], 1-means[k]/means['cv'], 'relative '+k)
        check(observed['past_selection_changes'] == sum(r['selection']['cv'] != r['selection']['past'] for r in rows), 'selection changes')

    summary_check(report['overall'], all_rows)
    for domain in ('electricity', 'pedestrian'):
        summary_check(report['domains'][domain], [r for r in all_rows if r['series_id'].startswith(domain+':')])
    summary_check(report['mature_round_ge_10'], [r for r in all_rows if r['round'] >= 10])
    check(report['development_gate_passed'] == (report['overall']['past_relative_reduction'] >= .2 and
          all(r['past_relative_reduction'] > 0 for r in report['domains'].values())), 'frozen gate')
    check(report['forecast_computations'] == 9984 and report['estimator_fits'] == 4992, 'cost accounting')
    check(report['api_calls'] == report['llm_tokens'] == 0, 'no API spending')
    result = {'checks': checks, 'failures': 0,
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'report_sha256': hashlib.sha256((results/'report.json').read_bytes()).hexdigest(),
              'scope': 'All saved pairs, task/fold slices, deterministic baseline predictions, selectors, visibility and aggregates; regression predictions not independently refitted; no reserved sources read.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'results', 'output'):
        parser.add_argument(name)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.results, args.output), indent=2))
