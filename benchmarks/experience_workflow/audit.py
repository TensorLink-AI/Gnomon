"""Run offline parity, replay immutability and deliberate-fault sensitivity checks."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .scenario import canonical, digest, generate, matches, oracle, public_answer, visible_history
from .storage import Store

MUTATIONS = ('ignore_recording', 'ignore_source', 'ignore_unit', 'ignore_context', 'ignore_revisions', 'no_memory')


def run(seeds, rounds, output):
    from .agent import source_manifest
    output.mkdir(parents=True, exist_ok=False)
    report = dict(scope='development deterministic harness validation', objective_achieved=False,
                  source=source_manifest(), checks=0, failures=[], worlds=[], mutation_failures_detected={m: 0 for m in MUTATIONS})
    for seed in seeds:
        world = generate(seed, rounds)
        feature_checks = feature_fills = 0
        for event in world['events']:
            if event['kind'] != 'forecast':
                continue
            req = event['request']
            base = datetime(2026, 1, 1, tzinfo=timezone.utc)
            day = (datetime.fromisoformat(req['cutoff']) - base).days
            history, evidence = visible_history(world['base_observations'][req['series_id']], world['events'], req['series_id'], base, day)
            feature_checks += 1
            feature_fills += sum(e['forward_filled'] for e in evidence)
            if history != req['history'] or evidence != event['history_evidence'] or any(
                e['source_available_at'] > req['cutoff'] or e['recorded_at'] > req['cutoff'] for e in evidence):
                report['failures'].append(dict(event_id=event['event_id'], cause='feature_visibility_violation'))
        stores = {arm: Store(output / f'{seed}-{arm}', arm) for arm in ('gnomon', 'sqlite')}
        saved = []
        try:
            for task in world['tasks']:
                receipts = [store.ingest(world['events'], task['now']) for store in stores.values()]
                report['checks'] += 1
                if receipts[0] != receipts[1]:
                    report['failures'].append(dict(task_id=task['task_id'], cause='unequal_information'))
                for label, query in task['queries'].items():
                    expected = oracle(world['events'], query)
                    for arm, store in stores.items():
                        got = store.query(query)
                        report['checks'] += 1
                        if not matches(got, expected):
                            report['failures'].append(dict(task_id=task['task_id'], vintage=label, arm=arm,
                                expected=public_answer(expected), actual=got))
                    saved.append((query, public_answer(expected)))
                    for mutation in MUTATIONS:
                        if not matches(public_answer(oracle(world['events'], query, mutation=mutation)), expected):
                            report['mutation_failures_detected'][mutation] += 1
            # Later records cannot rewrite an old evidence query.
            for query, expected in saved:
                for arm, store in stores.items():
                    report['checks'] += 1
                    if not matches(store.query(query), expected):
                        report['failures'].append(dict(seed=seed, arm=arm, cause='historical_answer_changed'))
            # Hindsight is a descriptive ceiling only, never an exposed tool.
            hindsight, no_memory, policy = [], [], []
            from .scenario import expected_provider
            for task in world['tasks']:
                candidates = [e for e in world['events'] if e['kind'] == 'forecast'
                    and e['request'] == task['request']]
                truth = world['truth'][f"{task['round']}/{task['request']['series_id']}"]
                scores = {e['provider']: math.sqrt(sum((math.log1p(p) - math.log1p(a)) ** 2
                    for p, a in zip(e['point'], truth)) / len(truth)) for e in candidates}
                hindsight.append(min(scores.values()))
                no_memory.append(scores['last_value'])
                policy.append(scores[expected_provider(oracle(world['events'], task['queries']['current']))])
            report['worlds'].append(dict(seed=seed, family=world['family'], rounds=rounds,
                feature_visibility_checks=feature_checks, forward_filled_feature_points=feature_fills,
                event_stream_sha256=digest(world['events']), tasks_sha256=digest(world['tasks']),
                hindsight_rmsle=sum(hindsight) / rounds, no_memory_rmsle=sum(no_memory) / rounds,
                prescribed_memory_policy_rmsle=sum(policy) / rounds,
                storage={arm: dict(events=len(s.seen), writes=s.writes, ingest_seconds=s.ingest_seconds,
                                  query_seconds=s.query_seconds) for arm, s in stores.items()}))
        finally:
            for store in stores.values():
                store.close()
    report['parity_passed'] = not report['failures']
    report['sensitivity_passed'] = all(report['mutation_failures_detected'].values())
    report['passed'] = report['parity_passed'] and report['sensitivity_passed']
    (output / 'report.json').write_text(canonical(report) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102, 103])
    parser.add_argument('--rounds', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seeds, args.rounds, args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
