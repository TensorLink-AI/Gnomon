"""Concise view of existing RMSLE ledger cards with exact evidence references."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import math
from ml_ledger_cards import development_cards

WINDOWS=('last_4_origins','last_12_origins','lifetime')


def compact_bytes(value):
    return json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def brief(full, task, evidence_path, evidence_sha256):
    """Render the same query; a compact summary is not all underlying evidence."""
    if full.get('metric')!='rmsle' or full['as_of']!=task['origin']:
        raise ValueError('RMSLE review and exact task origin required')
    if full['provider_calls']!=0:
        raise ValueError('Evidence review must not execute a provider')
    at=datetime.fromisoformat(task['origin'])
    if at.tzinfo is None:raise ValueError('Explicit query timezone required')
    if len(evidence_sha256)!=64 or any(c not in '0123456789abcdef' for c in evidence_sha256):
        raise ValueError('Full evidence SHA-256 required')
    catalog=full['configuration_index'];cards=[]
    for index,card in enumerate(full['cards']):
        ids=[card['left_config_id'],card['right_config_id']]
        if len(set(ids))!=2 or any(c not in catalog for c in ids):
            raise ValueError('Two catalog configurations required')
        providers={catalog[c]['provider']:catalog[c]['revision'] for c in ids}
        if providers!=card['providers']:raise ValueError('Pair identity mismatch')
        item={'config_ids':ids,'windows':{},'recent_lifetime_disagreement':card.get('recent_lifetime_disagreement'),
              'excluded_count':len(card['excluded']),
              'evidence_pointer':f'/cards/{index}'}
        seen=[]
        for label in WINDOWS:
            w=card['windows'][label];pointer=f'/cards/{index}/windows/{label}'
            if w['n']!=w['matched_origins']*task['horizon'] or len(w['origins'])!=w['matched_origins']:
                raise ValueError('Incomplete matched horizon or origin counts')
            for origin in w['origins']:
                if datetime.fromisoformat(origin['origin'])>=at:
                    raise ValueError('Review includes a nonhistorical origin')
            same=next((old for old in seen if card['windows'][old]==w),None)
            if same is not None:
                item['windows'][label]={'same_as':same,'evidence_pointer':pointer}
                continue
            seen.append(label)
            model_map={m['provider']:m for m in w['models']}
            if w['matched_origins'] and (set(model_map)!=set(providers) or len(w['models'])!=2):
                raise ValueError('Matched scores require both providers')
            scores={c:model_map[catalog[c]['provider']]['rmsle'] for c in ids} if w['matched_origins'] else {}
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in scores.values()):
                raise ValueError('Finite nonnegative RMSLE scores required')
            item['windows'][label]={'status':w['status'],'matched_origins':w['matched_origins'],
                'n':w['n'],'start':w['start'],'end':w['end'],'scores':scores,
                'lowest_error_config_ids':[c for c in ids if scores and scores[c]==min(scores.values())],
                'evidence_pointer':pointer}
        cards.append(item)
    page_complete=full['offset']==0 and full['next_offset'] is None and len(cards)==full['total_pairs']
    if full['next_offset'] is not None and (not cards or full['next_offset']!=full['offset']+len(cards)):
        raise ValueError('Inconsistent next-page offset')
    next_call=None if full['next_offset'] is None else {
        'operation':'review','arguments':{'offset':full['next_offset'],'limit':len(cards)},
        'valid_for_origin':task['origin']}
    return {'schema_version':'agent-review-087','status':full['status'],'metric':'rmsle',
        'query':{k:task[k] for k in ('series_id','unit','horizon','origin')},
        'configuration_index':deepcopy(catalog),'cards':cards,
        'pagination':{'total_pairs':full['total_pairs'],'shown_pairs':len(cards),'offset':full['offset'],
                      'all_pairs_included':page_complete,'next_call':next_call},
        'catalog_excluded_count':len(full['catalog_excluded']),
        'global_past_origin_count':len(full['global_past_origins']),
        'full_evidence':{'path':str(evidence_path),'sha256':evidence_sha256},
        'returned_evidence':'summary_with_exact_full_evidence_references',
        'cohort_semantics':'Scores compare only within each pair/window; no global ranking across unequal cohorts.',
        'window_semantics':full['window_semantics'],'pair_order':full['pair_order'],
        'aggregation':'mean_rmsle_over_complete_matched_origins',
        'same_as_semantics':'Same complete evidence and statistics as the named window in this card.',
        'ledger_queries':full['ledger_queries'],'provider_calls':full['provider_calls'],
        'forecast_selection_made':False}


def review(db, records, task, evidence_math, evidence_path, **options):
    """Read through published ledger APIs and save immutable full query evidence."""
    full=development_cards(db,records,task,evidence_math,**options)
    data=compact_bytes(full);digest=hashlib.sha256(data).hexdigest()
    result=brief(full,task,evidence_path,digest)
    with Path(evidence_path).open('xb') as stream:stream.write(data)
    return result
