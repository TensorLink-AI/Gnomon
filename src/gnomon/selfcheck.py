"""Installed-package checks for Gnomon's structural trust guarantees."""

from __future__ import annotations

import csv
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

FAMILIES = ('snapshot', 'revision_visibility', 'recorded_visibility', 'fold_boundaries',
            'future_covariates', 'multiple_series', 'dst', 'repaired_histories', 'cache_equivalence')


def family_contracts():
    specs = {
        'snapshot': ('Seeded 24–60 daily values, 1–7 future values, four late revisions.', 'Exact visible history, last-value forecast and maximum known time <= cutoff.', 'Synthetic source times; not model training leakage or recording-time proof.'),
        'revision_visibility': ('Value 1 at start, revision 2 available two days later.', 'Old/new source cutoffs select values 1/2.', 'One series and revision per case.'),
        'recorded_visibility': ('Revision available day 2, locally recorded day 3.', 'Recording cutoff day 2 still selects value 1.', 'Synthetic controlled clock; no concurrent writers.'),
        'multiple_series': ('Series a values 1/2 and series b value 100.', 'Revising a leaves b at 100.', 'Storage isolation, not panel-provider execution.'),
        'dst': ('Alternate New York spring/fall 2026 transitions, local noon.', 'Calendar day and elapsed 24 hours match the specified distinct UTC instants.', 'Two transitions in one zone; not all DST histories.'),
        'cache_equivalence': ('Deterministic versioned synthetic last-value callable, integer/float inputs, bypass.', 'Equal forecasts, second call hit, exactly two provider calls.', 'No eviction or concurrent lookup stress.'),
        'future_covariates': ('Two-step request, two valid future rows then one invalid row.', 'Invalid row count rejected before provider dispatch.', 'Shape contract only; does not establish future availability.'),
        'fold_boundaries': ('Thirty increasing daily observations, three two-step folds.', 'Complete study and each history end <= origin < first target.', 'Built-in point forecasts on regular daily data.'),
        'repaired_histories': ('Thirty daily observations with one interior observation removed and interpolated.', 'Global interpolation is rejected for historical evaluation.', 'One gap case; not every repair or per-vintage preparation policy.'),
    }
    return {'schema_version': '1', 'status': 'ok', 'families': {k: dict(zip(('fixture', 'assertions', 'limitations'), specs[k])) for k in FAMILIES},
            'always_run': ['snapshot'], 'detailed_command': 'gnomon self-check leakage --cases 20 --seed 7 --detailed --families ' + ' '.join(FAMILIES)}


def leakage_self_check(cases: int = 8, seed: int = 7, families=None, detailed: bool = False) -> dict[str, Any]:
    """Check snapshot reads against declared cutoffs on finite synthetic cases.

    This is a mechanism check, not the historical LLM-control comparison.
    It ships so an installed wheel can exercise the snapshot mechanism
    without network access or benchmark fixtures.
    """
    from .session import GnomonSession
    from .temporal_store import TemporalStore

    if type(cases) is not int or not 1 <= cases <= 1000:
        raise ValueError("cases must be an integer from 1 to 1000")
    if type(detailed) is not bool:
        raise ValueError('detailed must be a boolean')
    families = list(families) if families is not None else ['snapshot']
    if not families or any(f not in FAMILIES for f in families):
        raise ValueError('families must select documented self-check families')
    rng = random.Random(seed)
    rows = []
    with tempfile.TemporaryDirectory(prefix="gnomon-leakage-check-") as directory:
        root = Path(directory)
        for case in range(cases):
            start = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=case * 3)
            history, horizon = rng.randint(24, 60), rng.randint(1, 7)
            cutoff = start + timedelta(days=history - 1)
            source = root / f"case-{case}.csv"
            records = []
            for index in range(history + horizon):
                timestamp = start + timedelta(days=index)
                value = 100 + index * .2 + rng.uniform(-1, 1)
                records.append((timestamp.isoformat(), value, timestamp.isoformat()))
                if history - 4 <= index < history:
                    records.append((timestamp.isoformat(), value + 20,
                                    (cutoff + timedelta(days=2)).isoformat()))
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "value", "published"])
                writer.writerows(records)
            store_path = root / f"case-{case}.db"
            dataset = f"case_{case}"
            TemporalStore(store_path).ingest_csv(
                str(source), dataset=dataset, time_column="timestamp",
                target_column="value", known_at_column="published",
            )
            with GnomonSession.from_config() as session:
                inspected = session.data.inspect(f"store:{dataset}",
                    as_of=cutoff.isoformat(), store_path=str(store_path))
                request = session.data.request(inspected["data_ref"], horizon=horizon)
                execution = session.engine.forecast("last_value", request)
                forecast_matches = execution.result.point == (request.history[-1],) * horizon
                expected = tuple(float(record[1]) for record in records
                    if datetime.fromisoformat(record[0]) <= cutoff
                    and datetime.fromisoformat(record[2]) <= cutoff)
                history_matches = request.history == expected
            accesses = inspected["snapshot"].get("accesses", [])
            known = [datetime.fromisoformat(item["max_known_time"])
                     for item in accesses if item.get("max_known_time")]
            boundary_holds = bool(known) and max(known) <= cutoff
            holds = boundary_holds and history_matches and forecast_matches
            details = {}
            additional = _additional_checks(root, case, start, families, details)
            holds = holds and all(additional.values())
            rows.append({"case": case, "cutoff": cutoff.isoformat(),
                         "max_known_time": max(known).isoformat() if known else None,
                         "history_length": history, "horizon": horizon,
                         "checks": {"known_time_boundary": boundary_holds,
                                    "visible_history_matches": history_matches, "forecast_matches": forecast_matches, **additional},
                         "holds": holds})
            if detailed:
                rows[-1]['assertions'] = {'snapshot': {'expected_history': list(expected), 'actual_history': list(request.history),
                    'expected_forecast': [request.history[-1]] * horizon, 'actual_forecast': list(execution.result.point),
                    'maximum_allowed_known_time': cutoff.isoformat(), 'actual_max_known_time': max(known).isoformat() if known else None}, **details}
    return {"schema_version": "0.2", 'status': 'ok' if all(row['holds'] for row in rows) else 'unscored', "check": "snapshot_temporal_leakage",
            "seed": seed, "cases": cases, "passed": sum(row["holds"] for row in rows),
            "failed": sum(not row["holds"] for row in rows),
            "checks_passed": bool(rows) and all(row["holds"] for row in rows),
            "evidence": "finite_synthetic_checks", "general_leakage_safety": "not_established",
            'scope': {'families': ['snapshot', *[f for f in families if f != 'snapshot']],
                'providers': ['last_value'] + (['historical_mean'] if 'fold_boundaries' in families else []) +
                    (['synthetic'] if 'future_covariates' in families or 'cache_equivalence' in families else []),
                'source_cutoffs': True, 'source_revisions': True, 'recorded_cutoffs': 'recorded_visibility' in families,
                'dst_boundaries': 'dst' in families, 'multiple_series': 'multiple_series' in families,
                'panel_provider_execution': False, 'external_providers': False,
                'future_covariates_scope': 'shape/capability contract only; does not attest real-world future availability'},
            "rows": rows,
            "limitation": "These finite synthetic cases exercise the installed snapshot mechanism. They do not prove general leakage safety or rerun the historical hosted-LLM control arm."}


def _additional_checks(root, case, start, families, details):
    from . import ForecastResult, InferenceEngine, GnomonSession
    from .forecast_adapter import AdapterCapabilities, ForecastAdapterError
    from .temporal_store import TemporalObservation, TemporalStore
    from .ids import FixedClock
    from .temporal_ops import temporal_operation
    checks = {}
    if any(f in families for f in ('revision_visibility', 'recorded_visibility', 'multiple_series')):
        store = TemporalStore(root / f'extra-{case}.db')
        later = start + timedelta(days=2)
        store.ingest_rows('test', [TemporalObservation('a', 'value', start, start, 1),
            TemporalObservation('b', 'value', start, start, 100)], source_fingerprint='original', clock=FixedClock(start))
        store.ingest_rows('test', [TemporalObservation('a', 'value', start, later, 2)],
            source_fingerprint='revision', clock=FixedClock(later + timedelta(days=1)))
        old = store.snapshot('test', start).series('a', 'value')
        new = store.snapshot('test', later).series('a', 'value')
        if 'revision_visibility' in families:
            checks['revision_visibility'] = old[0].value == 1 and new[0].value == 2
            details['revision_visibility'] = {'expected': [1, 2], 'actual': [old[0].value, new[0].value]}
        if 'recorded_visibility' in families:
            visible = store.snapshot('test', later, recorded_as_of=later).series('a', 'value')
            checks['recorded_visibility'] = visible[0].value == 1
            details['recorded_visibility'] = {'expected': 1, 'actual': visible[0].value, 'recorded_cutoff': later.isoformat()}
        if 'multiple_series' in families:
            checks['multiple_series'] = new[0].value == 2 and store.snapshot('test', later).series('b', 'value')[0].value == 100
            details['multiple_series'] = {'expected': [2, 100], 'actual': [new[0].value, store.snapshot('test', later).series('b', 'value')[0].value]}
    if 'dst' in families:
        origin = '2026-03-07T12:00:00' if case % 2 == 0 else '2026-10-31T12:00:00'
        calendar = temporal_operation(operation='shift', value=origin, timezone='America/New_York', amount=1, unit='days', mode='calendar')
        elapsed = temporal_operation(operation='shift', value=origin, timezone='America/New_York', amount=24, unit='hours', mode='elapsed')
        expected = ('2026-03-08T16:00:00Z', '2026-03-08T17:00:00Z') if case % 2 == 0 else ('2026-11-01T17:00:00Z', '2026-11-01T16:00:00Z')
        checks['dst'] = (calendar['result']['utc'], elapsed['result']['utc']) == expected
        details['dst'] = {'input': origin, 'timezone': 'America/New_York', 'expected': list(expected), 'actual': [calendar['result']['utc'], elapsed['result']['utc']]}
    if 'cache_equivalence' in families or 'future_covariates' in families:
        with InferenceEngine(cache_size=2) as engine:
            count = []
            def predict(r):
                count.append(1)
                return ForecastResult((r.history[-1],) * r.horizon, series_id=r.series_id, unit=r.unit, timestamps=r.future_timestamps)
            engine.register('synthetic', predict, capabilities=AdapterCapabilities(future_covariates=True), revision='v1', deterministic=True)
            request = dict(history=[1, 2, 3], horizon=2, unit='widgets')
            if 'cache_equivalence' in families:
                a, b = engine.forecast('synthetic', request), engine.forecast('synthetic', {**request, 'history': [1., 2., 3.]})
                c = engine.forecast('synthetic', request, use_cache=False)
                checks['cache_equivalence'] = a.result == b.result == c.result and b.cache_hit and len(count) == 2
                details['cache_equivalence'] = {'expected': {'points': [[3, 3]] * 3, 'second_hit': True, 'provider_calls': 2},
                    'actual': {'points': [list(r.result.point) for r in (a, b, c)], 'second_hit': b.cache_hit, 'provider_calls': len(count)}}
            if 'future_covariates' in families:
                engine.forecast('synthetic', {**request, 'future_covariates': [[4], [5]]})
                before = len(count)
                try:
                    engine.forecast('synthetic', {**request, 'future_covariates': [[4]]})
                except ForecastAdapterError:
                    checks['future_covariates'] = len(count) == before
                else:
                    checks['future_covariates'] = False
                details['future_covariates'] = {'expected': {'rejected': True, 'provider_call_delta': 0}, 'actual': {'rejected': checks['future_covariates'], 'provider_call_delta': len(count) - before}}
    if 'fold_boundaries' in families or 'repaired_histories' in families:
        source = root / f'plain-{case}.csv'
        source.write_text('timestamp,value\n' + ''.join(f'{start + timedelta(days=i)},{i}\n' for i in range(30)))
        with GnomonSession.from_config() as session:
            if 'fold_boundaries' in families:
                ref = session.data.inspect(str(source))['data_ref']
                study = session.evaluate(ref, candidates=['historical_mean'], baseline='last_value', horizon=2, folds=3)
                checks['fold_boundaries'] = study['status'] == 'complete' and all(
                    datetime.fromisoformat(f['request']['timestamps'][-1]) <= datetime.fromisoformat(f['origin']) <
                    datetime.fromisoformat(f['actuals'][0]['valid_time']) for f in study['folds'])
                details['fold_boundaries'] = {'expected': 'complete; history_end <= origin < target_start for all three folds',
                    'actual_status': study['status'], 'actual': [{'history_end': f['request']['timestamps'][-1], 'origin': f['origin'], 'target_start': f['actuals'][0]['valid_time']} for f in study['folds']]}
            if 'repaired_histories' in families:
                source.write_text(source.read_text().replace(f'{start + timedelta(days=10)},10\n', ''))
                ref = session.data.inspect(str(source), frequency='D', repair='aggressive')['data_ref']
                try:
                    session.evaluate(ref, candidates=['historical_mean'], baseline='last_value', horizon=2)
                except (ForecastAdapterError, ValueError):
                    checks['repaired_histories'] = True
                else:
                    checks['repaired_histories'] = False
                details['repaired_histories'] = {'expected': 'historical evaluation rejected', 'actual': 'rejected' if checks['repaired_histories'] else 'accepted'}
    return checks
