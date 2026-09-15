"""Post-hoc search coverage on authenticated development evidence; no policy.

Tables are end-of-session complete CV, not necessarily what was visible at an
earlier selection. Their scores are never used to fabricate target forecasts.
"""
import json
import math
from statistics import mean

from .contrast_history_progress_100 import ARMS, summarize


def config_key(config):
    if not isinstance(config, dict) or not config:
        raise ValueError('Nonempty configuration object required')
    return json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)


def exploration(rows, tables):
    """tables maps (arm, series_id, round) to audited final current_cv objects."""
    population = summarize(rows)
    keyed = {(r['arm'], r['series_id'], r['round']): r for r in rows}
    details = {}
    for key, row in keyed.items():
        table = tables.get(key)
        if not isinstance(table, dict) or not isinstance(table.get('configurations'), list):
            raise ValueError('Every observed session requires its explicit audited CV table')
        scores, models = {}, set()
        for entry in table['configurations']:
            identity = config_key(entry['config'])
            value = entry['mean_rmsle']
            if identity in scores or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Unique configurations and finite nonnegative CV scores required')
            model = entry['config'].get('model')
            if not isinstance(model, str) or not model:
                raise ValueError('Explicit model family required')
            scores[identity] = value
            models.add(model)
        selected = config_key(row['config']) if row.get('config') else None
        selected_score = scores.get(selected)
        details[key] = {
            'complete_cv_configs': len(scores), 'families': sorted(models),
            'selected_config': row.get('config'), 'selected_in_final_cv': selected in scores,
            'selected_cv_rank': 1 + sum(v < selected_score for v in scores.values())
                                if selected_score is not None else None,
            'selected_is_cv_minimum': selected_score == min(scores.values())
                                      if selected_score is not None else None,
            'config_keys': sorted(scores),
        }
    cases = []
    for series, number in sorted({(s, n) for _, s, n in keyed}):
        if not all((a, series, number) in keyed for a in ARMS):
            continue
        arms = {a: details[a, series, number] for a in ARMS}
        base, ledger = arms['gnomon'], arms['ledger']
        bset, lset = set(base['config_keys']), set(ledger['config_keys'])
        cases.append({
            'series_id': series, 'round': number, 'arms': arms,
            'same_complete_cv_set': bset == lset,
            'shared_complete_cv_configs': len(bset & lset),
            'control_only_configs': sorted(bset-lset), 'ledger_only_configs': sorted(lset-bset),
            'same_selected_config': base['selected_config'] == ledger['selected_config'],
            'ledger_minus_control_rmsle': keyed['ledger', series, number]['rmsle']
                                          - keyed['gnomon', series, number]['rmsle'],
        })
    arms = {}
    for arm in ARMS:
        selected = [c['arms'][arm] for c in cases]
        arms[arm] = {
            'matched_sessions': len(selected),
            'mean_complete_cv_configs': mean(r['complete_cv_configs'] for r in selected) if selected else None,
            'family_exposure_sessions': {m: sum(m in r['families'] for r in selected)
                                         for m in sorted({m for r in selected for m in r['families']})},
            'selected_cv_minimum_sessions': sum(r['selected_is_cv_minimum'] is True for r in selected),
            'selected_not_in_final_cv_sessions': sum(not r['selected_in_final_cv'] for r in selected),
        }
    return {
        'primary_all_matched': population['primary_all_matched'], 'arms': arms,
        'same_complete_cv_set_cases': sum(c['same_complete_cv_set'] for c in cases),
        'same_selected_config_cases': sum(c['same_selected_config'] for c in cases),
        'cases': cases, 'pending': population['pending'],
        'scope': 'Post-hoc development search description. All matched cases retained, including '
                 'failures. Final CV tables may postdate selection. Different search sets and '
                 'scores are observed associations, not causal attribution to ledger memory. '
                 'No hindsight selector, unexecuted forecast, final-data access or dispatch.',
        'engy_calls': 0, 'provider_calls': 0, 'final_gate_opened': False,
    }
