"""Public-API MAE cards for a dynamic-config 1.1.9 incumbent.

No Gnomon implementation imports: run unchanged against either pinned runtime.
Pairs have separate matched cohorts; their ranks are never pooled.
"""
from copy import deepcopy
from datetime import datetime
from itertools import combinations
from statistics import mean


def instant(value):
    result=datetime.fromisoformat(value)
    if result.tzinfo is None:raise ValueError('Explicit timezone required')
    return result


def catalog(records,task):
    now=instant(task['origin']);configs={};seen={};origins=set();excluded=[]
    for row in records:
        if row.get('event')!='result' or row.get('kind')!='forecast':continue
        r=row['request'];ex=row['execution'];eid=ex['execution_id']
        if (r['series_id']!=task['series_id'] or r.get('unit')!=task['unit'] or r['horizon']!=task['horizon']):
            excluded.append({'execution_id':eid,'reason':'task_identity_mismatch'});continue
        times=[instant(t) for t in r['future_timestamps']];origin=instant(r['cutoff'])
        if len(times)!=task['horizon'] or len(set(times))!=len(times) or min(times)<=origin:
            raise ValueError('Invalid forecast timestamps')
        if origin>=now or max(times)>now:
            excluded.append({'execution_id':eid,'reason':'horizon_not_closed'});continue
        cid=row['config_id'];identity=(ex['provider'],ex['revision'],row['config'])
        if eid in seen:
            if seen[eid]!=(cid,origin,identity):raise ValueError('Conflicting execution identity')
            continue
        seen[eid]=(cid,origin,identity)
        if cid in configs and configs[cid]['identity']!=identity:
            raise ValueError('Configuration identity or revision changed')
        value=configs.setdefault(cid,{'identity':identity,'origins':set()})
        value['origins'].add(origin);origins.add(origin)
    provider_ids=[v['identity'][0] for v in configs.values()]
    if len(provider_ids)!=len(set(provider_ids)):
        raise ValueError('Provider name aliases multiple configurations')
    return configs,sorted(origins),excluded


def mae_window(origins,providers,start):
    selected=[o for o in origins if start is None or instant(o['origin'])>=start]
    return {'status':'ok' if selected else 'insufficient_evidence',
            'matched_origins':len(selected),'n':sum(o['n'] for o in selected),
            'models':[{'provider':p,'revision':rev,'mae':mean(next(m['mae'] for m in o['models'] if m['provider']==p)
                        for o in selected)} for p,rev in providers.items()] if selected else [],
            'origins':selected,'start':selected[0]['origin'] if selected else None,
            'end':selected[-1]['origin'] if selected else None,
            'aggregation':'mean_mae_over_complete_matched_origins'}


def incumbent_cards(db,records,task,*,offset=0,limit=12,pair=None):
    if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=12:
        raise ValueError('offset must be nonnegative; limit must be 1..12')
    configs,global_origins,excluded=catalog(records,task)
    pairs=list(combinations(sorted(configs),2))
    def order(pair):
        common=configs[pair[0]]['origins'] & configs[pair[1]]['origins']
        return (-max(common).timestamp() if common else float('inf'),pair)
    pairs.sort(key=order)
    if pair is not None:
        if len(pair)!=2 or len(set(pair))!=2 or any(c not in configs for c in pair):
            raise ValueError('Select two distinct catalog configuration IDs')
        selected=[tuple(sorted(pair))];next_offset=None
    else:
        selected=pairs[offset:offset+limit]
        next_offset=offset+len(selected) if offset+len(selected)<len(pairs) else None
    cards=[]
    for left,right in selected:
        providers={configs[c]['identity'][0]:configs[c]['identity'][1] for c in (left,right)}
        result=db.compare_history(series_id=task['series_id'],unit=task['unit'],horizon=task['horizon'],
                 providers=providers,start=global_origins[0].isoformat(),end=global_origins[-1].isoformat(),
                 source_as_of=task['origin'],recorded_as_of=task['origin'])
        origins=sorted(result['origins'],key=lambda o:instant(o['origin']))
        if len({instant(o['origin']) for o in origins})!=len(origins):
            raise ValueError('Public comparison returned duplicate origins')
        for o in origins:
            if instant(o['origin'])>=instant(task['origin']):raise ValueError('Future origin returned')
            if {m['provider']:m['revision'] for m in o['models']}!=providers:
                raise ValueError('Public comparison changed provider identity')
        windows={label:mae_window(origins,providers,global_origins[-count] if count and len(global_origins)>=count else None)
                 for label,count in [('last_4_origins',4),('last_12_origins',12),('lifetime',0)]}
        cards.append({'left_config_id':left,'right_config_id':right,'providers':providers,
                      'windows':windows,'excluded':result['excluded'],
                      'lifetime_duplicates_ignored':result['duplicates_ignored'],
                      'provider_calls':result['provider_calls']})
        if result['provider_calls']!=0:raise ValueError('Evidence query executed a provider')
    return {'adapter':'dynamic-incumbent-mae-v1','metric':'mae','as_of':task['origin'],
            'status':'evidence_available' if any(c['windows']['lifetime']['matched_origins'] for c in cards) else 'insufficient_evidence',
            'configuration_index':{cid:{'provider':v['identity'][0],'revision':v['identity'][1],
                                  'config':v['identity'][2]} for cid,v in sorted(configs.items())},
            'global_past_origins':[t.isoformat() for t in global_origins],
            'cards':cards,'catalog_excluded':excluded,'total_pairs':len(pairs),'offset':offset,
            'next_offset':next_offset,'summary_limited':next_offset is not None or offset>0,
            'ledger_queries':len(cards),'provider_calls':0,
            'cohort_semantics':'Matched within each pair/window; no global ranking across unequal cohorts.',
            'window_semantics':'Last 4, last 12, or all global past production origins; require matching evidence within each window.',
            'pair_order':'Most recent shared production origin, then configuration IDs; never score-based.'}


def compact_cards(result,evidence_path):
    """A concise view whose exact referenced full evidence is saved by the caller."""
    compact=deepcopy(result)
    for index,card in enumerate(compact['cards']):
        card['excluded_count']=len(card.pop('excluded'))
        card['excluded_reference']=f'{evidence_path}#/cards/{index}/excluded'
        for name,window in card['windows'].items():
            window.pop('origins')
            window['origins_reference']=f'{evidence_path}#/cards/{index}/windows/{name}/origins'
    compact['full_evidence_path']=str(evidence_path)
    compact['returned_evidence']='summary_with_exact_full_evidence_references'
    return compact
