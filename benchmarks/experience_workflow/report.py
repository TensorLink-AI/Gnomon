"""Track preregistered gates without turning development results into confirmation."""
import argparse
import json
from pathlib import Path
import random
from statistics import mean

from .scenario import canonical


def measures(rows):
    result = {}
    for arm in ('gnomon', 'sqlite'):
        group = [r for r in rows if r['arm'] == arm]
        completed = sum(r['completed'] for r in group)
        usage = bool(group) and all(r['usage_complete'] and r['tokens'] is not None for r in group)
        result[arm] = dict(checkpoints=len(group), correct=completed,
            completion_rate=completed / len(group) if group else None,
            total_tokens=sum(r['tokens'] for r in group) if usage else None,
            tokens_per_correct=sum(r['tokens'] for r in group) / completed if usage and completed else None,
            rmsle=mean(r['rmsle'] for r in group) if group else None,
            api_calls=sum(r['api_calls'] for r in group), tool_attempts=sum(r['tool_attempts'] for r in group),
            provider_calls=sum(r['provider_calls'] for r in group),
            api_errors=sum(len(r['api_errors']) for r in group),
            rejected_tool_calls=sum(len(r['errors']) for r in group),
            fallbacks=sum(r['fallback_used'] for r in group))
    g, s = result['gnomon'], result['sqlite']
    return dict(arms=result,
        cost_ratio=g['tokens_per_correct'] / s['tokens_per_correct'] if g['tokens_per_correct'] is not None and s['tokens_per_correct'] else None,
        completion_difference=g['completion_rate'] - s['completion_rate'] if g['completion_rate'] is not None and s['completion_rate'] is not None else None,
        rmsle_ratio=g['rmsle'] / s['rmsle'] if g['rmsle'] is not None and s['rmsle'] else None)


def summarize(rows, manifest, draws=5000):
    worlds = sorted({r['world'] for r in rows})
    point = measures(rows)
    groups = {w: [r for r in rows if r['world'] == w] for w in worlds}
    pairs = {}
    for r in rows:
        pairs.setdefault((r['world'], r['agent_seed'], r['round']), []).append(r)
    paired = all(len(pair) == 2 and {r['arm'] for r in pair} == {'gnomon', 'sqlite'}
                 and len({r['task_sha256'] for r in pair}) == 1
                 and len({r['event_receipt_sha256'] for r in pair}) == 1 for pair in pairs.values())
    expected_grid = {(w, a, r, arm) for w in manifest.get('seeds', []) for a in manifest['agent_seeds']
                     for r in range(manifest['rounds']) for arm in ('gnomon', 'sqlite')}
    actual_grid = {(r['world'], r['agent_seed'], r['round'], r['arm']) for r in rows}
    complete_run = (bool(rows) and paired and actual_grid == expected_grid and len(rows) == manifest['expected_decisions']
                    and all(r['usage_complete'] and not r['api_errors'] for r in rows))
    estimates = {k: [] for k in ('cost_ratio', 'completion_difference', 'rmsle_ratio')}
    rng = random.Random(20260911)
    if len(worlds) >= 2 and complete_run:
        for _ in range(draws):
            sample = measures([r for w in rng.choices(worlds, k=len(worlds)) for r in groups[w]])
            for name in estimates:
                if sample[name] is not None:
                    estimates[name].append(sample[name])
    intervals = {}
    for name, values in estimates.items():
        if len(values) == draws:
            values.sort()
            intervals[name] = [values[int(.025 * (draws - 1))], values[int(.975 * (draws - 1))]]
        else:
            intervals[name] = None
    g = point['arms']['gnomon']
    checks = dict(
        cost_point=point['cost_ratio'] is not None and point['cost_ratio'] <= .80,
        cost_uncertainty=intervals['cost_ratio'] is not None and intervals['cost_ratio'][1] < 1.,
        completion_floor=g['completion_rate'] is not None and g['completion_rate'] >= .95,
        completion_noninferiority=intervals['completion_difference'] is not None and intervals['completion_difference'][0] >= -.02,
        forecast_noninferiority=intervals['rmsle_ratio'] is not None and intervals['rmsle_ratio'][1] <= 1.02,
        complete_paired_run=complete_run,
        sufficient_confirmation_worlds=worlds == list(range(9000, 9024)) and manifest['rounds'] == 24 and sorted(manifest['agent_seeds']) == [7, 19],
        frozen_prerequisites=bool(manifest.get('prerequisites_passed')),
        unchanged_source=bool(manifest.get('source_unchanged')))
    achieved = manifest['scope'] == 'confirmation' and manifest.get('confirmation_guard_passed') is True and all(checks.values())
    return dict(scope=manifest['scope'], objective_achieved=achieved,
        objective_status='achieved_on_frozen_synthetic_confirmation' if achieved else
            'confirmation_target_not_established' if manifest['scope'] == 'confirmation' else 'development_only_confirmation_not_run',
        target_tokens_per_correct_ratio=.80,
        point=point, exploratory_95pct_world_cluster_intervals=intervals,
        gates={name: 'met_in_this_sample' if value else 'not_met_or_not_established' for name, value in checks.items()},
        world_count=len(worlds), seed_repetitions_are_not_independent_worlds=True,
        paired_information_audit_passed=paired,
        stages={name: measures([r for r in rows if (r['round'] < 4) == (name == 'cold')]) for name in ('cold', 'mature')},
        families={name: measures([r for r in rows if r['family'] == name]) for name in sorted({r['family'] for r in rows})},
        dollars=None, limitations=['Synthetic evidence workflow; no real-retail superiority claim.',
            'Primary cost is all billed tokens divided by correct checkpoints, including failures.',
            'Synthetic worlds, requested seeds and small pilots do not establish production robustness.',
            'Accepted decisions are guarded by the common host oracle; rejection attempts are reported separately.',
            'Confirmation requires frozen prompts/code, parity and mutation audits, and the guarded runner.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if (args.run / 'source_audit.json').exists():
        manifest['source_unchanged'] = json.loads((args.run / 'source_audit.json').read_text())['unchanged']
    rows = [json.loads(p.read_text()) for p in sorted(args.run.glob('*/*.decision.json'))]
    report = summarize(rows, manifest)
    (args.run / 'progress.json').write_text(canonical(report) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
