"""Replay released selectors; report paired losses, coverage, failures and uncertainty."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import math
from pathlib import Path

import numpy as np
from gnomon.adaptive_router import replay_router, validate_policy

from .common import digest, identity, read, write
from .data import forward_fill
from .run import validate_manifest


def policy(manifest, arm):
    spec, protocol = manifest['spec'],manifest['protocol']
    memory = {'season':spec['season'], 'short_window':max(4,spec['season']),
              'long_window':protocol['max_history'], 'k':32, 'min_effective_n':8}
    if arm == 'bounded':
        memory['distance'] = 'fase'
    elif arm == 'fase':
        memory.update(distance='fase',profile='fase',retention='fase')
    elif arm == 'no_memory':
        memory = None
    return validate_policy({'baseline':'seasonal_naive',
        'candidates':[n for n in protocol['models'] if n != 'seasonal_naive'],
        'metric':'mae', 'min_origins':5, 'recent_origins':20,
        'lookback_seconds':100*366*86400, 'memory':memory,
        'pool':{'series':[s['id'] for s in manifest['series']], 'own_weight':1}}, replay=True)


def mae(actual,point):
    return float(np.mean(np.abs(np.asarray(actual)-np.asarray(point))))


def validation_scale(series, cutoff, season):
    # Shared fixed-model selection occurs at the earliest scored decision in the
    # pool; another series' later history must not influence its normalisation.
    prefix = [v for t,v in zip(series['timestamps'],series['values'])
              if datetime.fromisoformat(t) <= datetime.fromisoformat(cutoff)]
    values = forward_fill(prefix)
    if len(values) <= season:
        return None
    scale = sum(abs(a-b) for a,b in zip(values[season:],values[:-season]))/(len(values)-season)
    return scale if scale > 0 else None


def block_interval(matrix, block, draws, seed):
    """Rows=series, columns=paired origins; retain cross-series dependence."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[1]
    if n < max(8, 2*block):
        return None  # no informative time resampling with only one effective block
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        starts = rng.integers(0,n,size=math.ceil(n/block))
        positions = np.concatenate([(start+np.arange(block))%n for start in starts])[:n]
        stats.append(float(matrix[:,positions].mean()))
    return [float(x) for x in np.quantile(stats,[.025,.975])]


def analyze(root):
    root = Path(root)
    manifest = read(root/'manifest.json')
    validate_manifest(manifest)
    pinned, complete = read(root/'identity.json'),read(root/'COMPLETE.json')
    if pinned['runtime'] != identity() or pinned['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Runtime/manifest changed; use the frozen evaluation checkout')
    receipts = {p.name:digest(read(p)) for p in sorted((root/'calls').glob('*.json'))}
    if complete['identity'] != digest(pinned) or complete['receipts'] != receipts:
        raise ValueError('Receipt set changed')
    protocol,spec = manifest['protocol'],manifest['spec']
    folds, histories, metadata, validation = [],{}, {},defaultdict(list)
    failure_counts = Counter()
    call_costs = defaultdict(list)
    missing = []
    models = protocol['models']
    score_start = min(datetime.fromisoformat(s['timestamps'][next(i for i,p in zip(s['origins'],s['phases']) if p=='score')-1]).isoformat(timespec='microseconds') for s in manifest['series'])
    for series in manifest['series']:
        sid, values = series['id'],series['values']
        stamps = [datetime.fromisoformat(t).isoformat(timespec='microseconds') for t in series['timestamps']]
        histories[sid] = list(zip(stamps,forward_fill(values)))
        selection_scale = validation_scale(series,score_start,spec['season'])
        for ordinal,(i,phase) in enumerate(zip(series['origins'],series['phases'])):
            actual = values[i:i+spec['horizon']]
            points, calls = {},{}
            for model in models:
                stem = root/'calls'/digest([sid,i,model])
                request = read(str(stem)+'.request.json')
                result = read(str(stem)+'.result.json')
                if result['request_sha256'] != digest(request):
                    raise ValueError('Result/request mismatch')
                points[model] = result['point']
                calls[model] = result
                call_costs[model].append(result)
                if result['status'] != 'ok':
                    failure_counts[model] += 1
            if any(x is None for x in actual) or series['mase_scale']<=0:
                missing.append({'series':sid,'origin_index':i,'phase':phase,
                                'reason':'missing_actual_or_zero_scale'})
                continue
            losses = {model:mae(actual,point) for model,point in points.items()}
            if phase == 'validation':
                if stamps[i+spec['horizon']-1] <= score_start and selection_scale is not None:
                    for model,loss in losses.items():
                        validation[model].append(loss/selection_scale)
            fold = {'series_id':sid,'origin':stamps[i-1],'target_time':stamps[i+spec['horizon']-1],
                    'actual':actual,'points':points}
            folds.append(fold)
            metadata[(sid,stamps[i-1])] = {'phase':phase,'ordinal':ordinal,'scale':series['mase_scale'],
                'losses':losses,'ensemble_loss':mae(actual,np.mean(list(points.values()),axis=0)),
                'fallbacks':[name for name,r in calls.items() if r['status']!='ok']}
    if not folds or any(not validation[m] for m in models):
        raise ValueError('Insufficient validation or scored data')
    fixed = min(models,key=lambda m:(float(np.mean(validation[m])),m))
    rows, policies = [],{}
    for arm in protocol['arms']:
        if arm in ('fixed_validation','equal_ensemble'):
            decisions = [{'series_id':f['series_id'],'origin':f['origin'],
                          'served':fixed if arm=='fixed_validation' else 'equal_ensemble'} for f in folds]
        else:
            policies[arm] = policy(manifest,arm)
            replay = replay_router(folds, policies[arm], histories, return_decisions=True,
                                   decision_losses=True, accelerate=True)
            if replay['excluded_folds']:
                raise ValueError(f"Replay excluded folds: {replay['excluded_folds']}")
            decisions = replay['decisions']
        for d in decisions:
            meta = metadata[(d['series_id'],d['origin'])]
            if meta['phase'] != 'score':
                continue
            loss = meta['ensemble_loss'] if arm=='equal_ensemble' else meta['losses'][d['served']]
            rows.append({'arm':arm,'series':d['series_id'],'origin':d['origin'],'ordinal':meta['ordinal'],
                         'mae':loss,'mase':loss/meta['scale'],'served':d['served'],
                         'evidence_level':d.get('evidence_level'), 'fallbacks':meta['fallbacks']})
    scores, changes, intervals = {},{},{}
    by_arm = {a:[r for r in rows if r['arm']==a] for a in protocol['arms']}
    for arm, entries in by_arm.items():
        grouped = defaultdict(list)
        for row in entries:
            grouped[row['series']].append(row['mase'])
        scores[arm] = float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
        changes[arm] = (scores[arm]/scores['current']-1)*100 if 'current' in scores and scores['current'] else None
    for arm in protocol['arms']:
        changes[arm] = (scores[arm]/scores['current']-1)*100 if scores['current'] else None
        paired = {(r['series'],r['ordinal']):r['mase'] for r in by_arm['current']}
        differences = defaultdict(list)
        for row in sorted(by_arm[arm],key=lambda r:(r['series'],r['ordinal'])):
            differences[row['series']].append(row['mase']-paired[(row['series'],row['ordinal'])])
        n = protocol['pilot_score_origins'] if manifest['mode']=='pilot' else protocol['score_origins']
        block = max(math.ceil(spec['horizon']/spec['stride']),math.ceil(n/4))
        # No complete-case CI when any planned scored cell is absent.
        intervals[arm] = block_interval(list(differences.values()),block,protocol['bootstrap_draws'],protocol['seed']) if differences and all(len(v)==n for v in differences.values()) else None
    expected = len(manifest['series'])*(protocol['pilot_score_origins'] if manifest['mode']=='pilot' else protocol['score_origins'])
    calls_count = len(receipts)//2
    switches = {}
    for arm, entries in by_arm.items():
        grouped = defaultdict(list)
        for row in sorted(entries,key=lambda r:(r['series'],r['origin'])):
            grouped[row['series']].append(row['served'])
        count = sum(a!=b for v in grouped.values() for a,b in zip(v,v[1:]))
        transitions = sum(max(0,len(v)-1) for v in grouped.values())
        switches[arm] = {'changes':count,'transitions':transitions,'rate':count/transitions if transitions else None}
    summary = {'dataset':spec['id'],'mode':manifest['mode'],'manifest_sha256':manifest['sha256'],
               'identity':digest(pinned),'fixed_validation_model':fixed,'policies':policies,
               'validation_folds_mature_at_scored_start':len(validation[models[0]]),
               'expected_scored_folds':expected,'complete_scored_folds':len(by_arm['current']),
               'missing_folds':missing,'model_failures':dict(failure_counts),'model_calls':calls_count,
               'model_failure_fraction':sum(failure_counts.values())/calls_count,
               'switches':switches,
               'model_costs':{name:{'calls':len(rs),'dispatch_seconds':sum(r.get('dispatch_seconds',0) for r in rs),
                   'cpu_seconds':sum(r['cpu_seconds'] for r in rs) if all(r.get('cpu_seconds') is not None for r in rs) else None}
                   for name,rs in call_costs.items()},
               'all_candidate_dispatch_seconds':complete['dispatch_seconds'],
               'scores_mase':scores,'error_change_percent_vs_current':changes,
               'paired_mase_delta_95ci':intervals,
               'decision_counts':{a:dict(Counter(r['served'] for r in rs)) for a,rs in by_arm.items()},
               'evidence_levels':{a:dict(Counter(r['evidence_level'] for r in rs if r['evidence_level'])) for a,rs in by_arm.items()},
               'claim_scope':'Pilot is feasibility only; fallback forecasts included; conditional scores if incomplete; no default promotion'}
    write(root/'decisions.json',rows)
    write(root/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    import json
    print(json.dumps(analyze(args.run),indent=2))


if __name__=='__main__':
    main()
