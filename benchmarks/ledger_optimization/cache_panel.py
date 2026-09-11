"""Compute candidate evidence for the new DEVELOPMENT panel only.

Confirmation is deliberately unsupported until a final candidate is frozen.
All model fits use original origin-bounded requests and pinned Arena recipes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys


def _series_job(arena_path, panel_path, series_id, out_path):
    sys.path.insert(0, arena_path)
    import pandas as pd
    from arena.favorita_eval import FavoritaEvalOptions, _candidate_bundle, _request
    options = FavoritaEvalOptions(data=Path(panel_path), out=Path(out_path), rounds=26)
    group = pd.read_parquet(panel_path, filters=[('unique_id', '=', series_id)]).sort_values('ds').reset_index(drop=True)
    initial = len(group)-26*14-1
    cases = []
    for round_index in range(26):
        index = initial+round_index*14
        req = _request(group, index, options)
        path = Path(out_path)/'candidates'/f'{series_id}-{round_index:02d}.json'
        bundle = _candidate_bundle(group, index, options, path)
        if any(row['status'] != 'ok' for row in bundle['candidates'].values()):
            raise ValueError(f'Candidate preparation failed; common case excluded pending repair: {series_id}/{round_index}')
        cases.append({'series_id': series_id, 'round': round_index, 'origin': req.cutoff,
            'future_timestamps': list(req.future_timestamps), 'request': asdict(req), 'actual': bundle['actual'],
            'outcome_recorded_at': req.future_timestamps[-1],
            'current_card': {p:{k:r[k] for k in ('cv_mae', 'cv_rmsle', 'forecast_total', 'status')}
                            for p,r in bundle['candidates'].items()},
            'predictions': {p:r['point'] for p,r in bundle['candidates'].items()},
            'candidate_metadata': {p:{k:r.get(k) for k in ('fallback_used', 'fallback_provider', 'model_error_type', 'cv_fallbacks')}
                                   for p,r in bundle['candidates'].items()}})
    return cases


def record(cases, path):
    from gnomon import ForecastResult, InferenceEngine, TemporalLedger
    from gnomon.forecast_adapter import AdapterCapabilities
    from gnomon.evidence_summary import rmsle
    from gnomon.final_selection import forecast_request_fingerprint
    from gnomon.ids import FixedClock
    from statistics import mean
    if path.exists():
        raise ValueError('Refusing to mutate existing preparation ledger')
    ledger = TemporalLedger(path)
    revision = 'statsforecast-2.0.3-favorita-v1'
    for c in cases:
        ledger.clock = FixedClock(datetime.fromisoformat(c['origin']))
        engine = InferenceEngine(ledger=ledger)
        expected = forecast_request_fingerprint(c['request'])
        c['execution_ids'] = []
        for p, point in c['predictions'].items():
            def provider(req, point=point, expected=expected, metadata=c['candidate_metadata'][p]):
                if forecast_request_fingerprint(asdict(req)) != expected:
                    raise ValueError('Task mismatch in candidate playback')
                return ForecastResult(point=tuple(point), series_id=req.series_id, unit=req.unit,
                                      timestamps=req.future_timestamps, metadata=metadata)
            engine.register(p, provider, revision=revision, lifecycle='stateless', deterministic=True,
                            capabilities=AdapterCapabilities(past_covariates=True, future_covariates=True))
            c['execution_ids'].append(engine.forecast(p, c['request']).execution_id)
        ledger.clock = FixedClock(datetime.fromisoformat(c['outcome_recorded_at']))
        for time, actual in zip(c['future_timestamps'], c['actual'], strict=True):
            ledger.append_actual(series_id=c['series_id'], unit=c['request']['unit'],
                valid_time=time, value=actual, source_available_at=time, source_ref='favorita:'+c['series_id'])
        c['scores'] = {p: rmsle(zip(point, c['actual']), 'clip_zero')[0] for p,point in c['predictions'].items()}
        c['mae'] = {p: mean(abs(y-a) for y,a in zip(point,c['actual'])) for p,point in c['predictions'].items()}
    for c in cases:
        visible = sorted({p['origin'] for p in cases if p['series_id']==c['series_id'] and p['origin']<c['origin']})
        card = {'as_of': c['origin'], 'note': 'MAE over complete matched production origins; same original windows.'}
        for label, count in [('last_4_origins', 4), ('last_12_origins', 12), ('lifetime', 0)]:
            if not visible:
                card[label] = {'status': 'no_mature_evidence', 'models': []}
                continue
            start = visible[-count] if count and len(visible)>=count else cases[0]['origin']
            q = ledger.compare_history(series_id=c['series_id'], horizon=14, unit=c['request']['unit'],
                providers={p:revision for p in c['predictions']}, start=start, end=visible[-1],
                source_as_of=c['origin'], recorded_as_of=c['origin'])
            card[label] = {k:q[k] for k in ('status','matched_origins','models','duplicates_ignored')}
            card[label]['excluded_count'] = len(q['excluded'])
        c['legacy_evidence'] = card
        c.pop('request')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-manifest', type=Path, required=True)
    parser.add_argument('--arena-checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.panel_manifest.read_text())
    panel = manifest['splits']['development']
    path = Path(panel['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=panel['sha256']:
        raise ValueError('Development panel changed')
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_series_job, str(args.arena_checkout), str(path), sid, str(args.output)):sid
                   for sid in panel['series']}
        for future in as_completed(futures):
            tasks.extend(future.result())
            print(json.dumps({'series':futures[future], 'candidate_cache':'complete'}),flush=True)
    cases = sorted(tasks,key=lambda c:(c['origin'],c['series_id']))
    prepared = record(cases, args.output/'ledger.db')
    (args.output/'cases.json').write_text(json.dumps(prepared,sort_keys=True,allow_nan=False)+'\n')
    receipt = {'scope':'additional development only', 'cases':len(prepared), 'series':panel['series'],
        'panel_sha256':panel['sha256'], 'cases_sha256':hashlib.sha256((args.output/'cases.json').read_bytes()).hexdigest(),
        'actual_visibility':'source at valid date, recording at horizon close; replay assumption',
        'forecast_basis':'original pinned recipes; task-bound ledger playback of computed points',
        'confirmation_opened':False}
    (args.output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
