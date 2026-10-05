"""Matched regression replay using previously saved forecasts, never model refits.

Inputs are the trusted local experiment archive, not arbitrary downloaded pickles.
Run each task/arm independently; output paths must be outside the source checkout.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import types
import time

from gnomon.adaptive_router import replay_router, validate_policy
from gnomon.build_info import source_fingerprint
import gnomon

TASKS = ['rv_1d', 'rv_3d', 'rv_7d', 'rv_14d', 'downside_7d', 'range_1d', 'alpha', 'favorita']
ARMS = ['current', 'v2_profile', 'bounded', 'fase', 'no_memory']


def load(root, task):
    digest = hashlib.sha256()
    def read(path, binary=False):
        raw = path.read_bytes()
        digest.update(str(path.relative_to(root)).encode() + b'\0' + raw)
        return pickle.loads(raw) if binary else json.loads(raw)
    if task in TASKS[:6]:
        data = read(root / 'crypto_broad/out' / f'{task}.pkl', True)
        names = data['metadata']['providers']
        baseline = 'ewma_10'
        memory = dict(short_window=7, long_window=56, season=7, k=32,
                      min_effective_n=8, own_weight=2., mask_covariate='observed')
        config = dict(metric='rmsle', min_origins=10, recent_origins=30,
                      lookback_seconds=365*86400, pool={'series':data['series']})
        cutoff = data['metadata']['score_start']
    else:
        protocol = read(root / 'cross_domain/protocol.json')[task]
        if task == 'alpha':
            data = read(root / 'cross_domain/out/alpha.pkl', True)
        else:
            base = root / 'cross_domain/data/favorita'
            data = {'folds':[f for p in sorted((base/'runs').glob('*-folds.json')) for f in read(p)],
                    'histories':read(base/'data/histories.json'),
                    'covariates':read(base/'data/covariates.json'),
                    'series':sorted(read(base/'data/meta.json'))}
        names, baseline, memory = protocol['models'], protocol['baseline'], protocol['memory']
        config = dict(metric=protocol['metric'], min_origins=10,
                      recent_origins=protocol['recent_origins'], lookback_seconds=protocol['lookback_days']*86400,
                      pool={'series':data['series'], 'own_weight':protocol['pool_own_weight']})
        origins = sorted({f['origin'] for f in data['folds']}, key=datetime.fromisoformat)
        cutoff = origins[protocol['warmup_origins']]
    config.update(baseline=baseline, candidates=[n for n in names if n != baseline], memory=memory)
    return data, config, cutoff, digest.hexdigest()


def policy_for(config, task, arm):
    policy = {**config, 'memory':dict(config['memory'])}
    if arm == 'no_memory':
        policy['memory'] = None
    elif arm == 'v2_profile':
        policy['memory'].pop('features', None)
        policy['memory']['profile'] = 'returns' if task == 'alpha' else 'intermittent' if task == 'favorita' else 'levels'
    elif arm == 'bounded':
        policy['memory']['distance'] = 'fase'
    elif arm == 'fase':
        policy['memory'].pop('features', None)
        policy['memory'].update(profile='fase', distance='fase', retention='fase')
    return validate_policy(policy, replay=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f'{args.task}.{args.arm}'
    if target.with_suffix(target.suffix + '.json').exists():
        raise SystemExit('Refusing to overwrite an existing result; choose a new output directory.')
    source_hash = source_fingerprint(Path(gnomon.__file__).parent)
    start = time.monotonic()
    data, raw, cutoff, input_hash = load(args.inputs, args.task)
    policy = policy_for(raw, args.task, args.arm)
    print(json.dumps({'task':args.task,'arm':args.arm,'folds':len(data['folds']), 'input_sha256':input_hash}), flush=True)
    checks = []
    if args.arm == 'current':
        import gnomon.adaptive_router as router
        def old_module(name, path):
            code = subprocess.check_output(['git', 'show', 'b741aa68bd86d534753ba3b8e5788d720c8c3543:'+path], text=True)
            module = types.ModuleType('gnomon.'+name)
            module.__package__ = 'gnomon'
            exec(compile(code, path, 'exec'), module.__dict__)
            return module
        legacy_memory = old_module('_release_memory', 'src/gnomon/episodic_memory.py')
        legacy_router = old_module('_release_router', 'src/gnomon/adaptive_router.py')
        legacy_router.episodic_scores = legacy_memory.episodic_scores
        actual_choose = router.choose_from_rows
        count = 0
        def checked(rows, pol, own, label, features=None, origin=None, incumbent=None, scorer=None):
            nonlocal count
            current = actual_choose(rows, pol, own, label, features, origin, incumbent, scorer)
            count += 1
            if count % max(1, len(data['folds'])//32) == 0:
                old = legacy_router.choose_from_rows(rows, pol, own, label, features, origin)
                for key in ('provider', 'reason', 'evidence_level'):
                    assert current.get(key) == old.get(key), (origin, own, key)
                checks.append({'series_id':own, 'origin':origin, 'provider':old['provider']})
            return current
        router.choose_from_rows = checked
    try:
        result = replay_router(data['folds'], policy, data['histories'], covariates=data.get('covariates'),
                               accelerate=True, return_decisions=True, decision_losses=True)
    finally:
        if args.arm == 'current':
            router.choose_from_rows = actual_choose
    decisions = [d for d in result.pop('decisions') if datetime.fromisoformat(d['origin']) >= datetime.fromisoformat(cutoff)]
    assert decisions and not result['excluded_folds'], result['excluded_folds']
    with gzip.open(str(target)+'.decisions.json.gz','wt') as f:
        json.dump(decisions,f)
    groups = defaultdict(list)
    for d in decisions:
        groups[d['series_id']].append(d)
    def score(rows):
        if args.task != 'favorita':
            return sum(d['served_loss'] for d in rows) / len(rows)
        series = defaultdict(list)
        for d in rows: series[d['series_id']].append(d)
        return sum(sum(d['served_loss'] for d in ds)/sum(d['losses'][policy['baseline']] for d in ds)
                   for ds in series.values()) / len(series)
    report = {'task':args.task,'arm':args.arm,'policy':policy,'input_sha256':input_hash,
              'source_sha256':source_hash,
              'folds_scored':len(decisions),'score_start':cutoff,'score':score(decisions),
              'metric':'mean_series_relative_mae' if args.task=='favorita' else 'mean_origin_rmsle',
              'series':len(groups),'origins':len({d['origin'] for d in decisions}),
              'evidence_levels':dict(Counter(d['evidence_level'] for d in decisions)),
              'switch_rate':sum(sum(a['served']!=b['served'] for a,b in zip(ds,ds[1:])) for ds in groups.values())/
                            sum(max(len(ds)-1,0) for ds in groups.values()),
              'per_series':{s:score(ds) for s,ds in groups.items()},'seconds':time.monotonic()-start,
              'baseline_old_code_checks':checks,
              'evaluation_scope':'previously inspected regression data; not a fresh holdout'}
    if args.task == 'alpha':
        for label, rows in [('regression',[d for d in decisions if d['origin']<'2026-08-01']),
                            ('extension',[d for d in decisions if d['origin']>='2026-08-01'])]:
            report[label] = {'count':len(rows),'score':score(rows)}
    assert source_hash == source_fingerprint(Path(gnomon.__file__).parent), 'Source changed during evaluation'
    Path(str(target)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
