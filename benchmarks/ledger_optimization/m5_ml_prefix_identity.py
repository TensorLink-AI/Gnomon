"""Seed/source/runtime checks before copying an M5 development prefix.

This pure check is one part of continuation admission, not a dispatch grant.
Callers separately authenticate capsule/plan bytes and recheck process status,
cohort coverage, audited forecasts, immutable archives and output isolation.
"""


def check_prefix_identity(manifest, capsule, runtime_inventory):
    seed = capsule.get('requested_seed', 7)
    if type(seed) is not int or seed not in (7, 19):
        raise ValueError('Unsupported capsule seed')
    if type(manifest.get('requested_seed')) is not int or manifest['requested_seed'] != seed:
        raise ValueError('Pilot requested seed differs from continuation capsule')
    if not capsule.get('sources') or manifest.get('sources') != capsule['sources']:
        raise ValueError('Pilot worker sources differ from continuation capsule')
    if not runtime_inventory or manifest.get('inventory') != runtime_inventory:
        raise ValueError('Pilot runtime inventory differs from continuation runtime')
    return {'seed_source_runtime_checks_passed': True, 'requested_seed': seed,
            'operational_gate_passed': False, 'execution_authorized': False,
            'final_gate_opened': False,
            'scope': 'Seed, source and runtime identity only. Complete cohort, temporal '
                     'evidence, terminal process, archive and one-shot dispatch checks remain required.'}
