"""Focused development failure replay; never a comparative-performance estimate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .agent import checkpoint, source_manifest
from .scenario import canonical, generate, digest
from .storage import Store
from benchmarks.ledger_optimization.agent_loop import api_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=102)
    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    world = generate(args.seed)
    if not 0 <= args.round < len(world['tasks']):
        raise ValueError('Round outside the generated development world')
    task = world['tasks'][args.round]
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_manifest()
    manifest = dict(scope='post-selected development failure replay; not a performance comparison',
        task_sha256=digest(task), seed=args.seed, round=args.round, source=before,
        limitation='Fresh chats/notebooks, with all prior arrived events preloaded; not a full longitudinal episode.')
    (args.output / 'manifest.json').write_text(canonical(manifest) + '\n')
    key = api_key()

    def play(arm):
        store = Store(args.output / arm, arm)
        try:
            store.ingest(world['events'], task['now'])
            return checkpoint(store, world, task, 7, key, args.output)
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(play, ('gnomon', 'sqlite')))
    report = dict(scope=manifest['scope'], objective_achieved=False,
        source_unchanged=before == source_manifest(), all_completed=all(r['completed'] for r in rows),
        rows=rows, dollars=None)
    (args.output / 'report.json').write_text(canonical(report) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
