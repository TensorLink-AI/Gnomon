"""Read-only complete030 review replay: presentation fidelity, not rescoring."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean,median
import sys
import time
from .agent_review import brief,compact_bytes
from .ml_ledger_cards import compact_cards

INVENTORY_SHA='722c105f0466ff56a87d9a90776739a4832871abdd50ac73700da2ae7a42902c'


def audit(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();checks=0;access={}
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise AssertionError(message)
    check(sha(root/'SHA256SUMS.json')==INVENTORY_SHA,'Original030 source inventory')
    inventory=json.loads((root/'SHA256SUMS.json').read_text())
    def read(name):
        check(name in inventory and sha(root/name)==inventory[name],'Source file hash')
        access[name]=inventory[name];return (root/name).read_bytes()
    jobs=json.loads(read('evaluation/host-jobs.json'));rows=[];seen=set();duplicates=0;covered=set();total_tasks=sum(len(t) for t in jobs.values())
    for series,tasks in sorted(jobs.items()):
        last=max(t['round'] for t in tasks);by_origin={t['origin']:t for t in tasks}
        choices=[f'evaluation/ledger/{series}/round-{last:02d}/project',f'evaluation/ledger/{series}/round-{last}/project']
        project=next(p for p in choices if p+'/review-log.jsonl' in inventory)
        packets=[json.loads(line) for line in read(project+'/review-log.jsonl').splitlines()]
        for packet_index,packet in enumerate(packets):
            identity=(series,hashlib.sha256(compact_bytes(packet)).hexdigest())
            if identity in seen:duplicates+=1;continue
            seen.add(identity);at=packet['task_origin'];task=by_origin[at]
            header={'series_id':series,'origin':at,'horizon':task['request']['horizon'],'unit':task['request']['unit']}
            relative=Path(packet['full_evidence_path'])
            check(not relative.is_absolute() and '..' not in relative.parts,'Local evidence path')
            name=project+'/'+relative.as_posix();raw=read(name);full=json.loads(raw)
            baseline=compact_cards(full,str(relative))
            for k,v in baseline.items():check(packet[k]==v,'Logged view matches immutable full query')
            view=brief(full,header,str(relative),hashlib.sha256(raw).hexdigest())
            check(view['configuration_index']==full['configuration_index'],'Complete configuration/provider/revision definitions')
            check(view['query']==header and full['as_of']==at,'Same original query')
            check(view['full_evidence']['sha256']==inventory[name],'Full evidence identity')
            check(len(view['cards'])==len(full['cards']),'No pair dropped')
            alias_count=0
            for i,(card,original) in enumerate(zip(view['cards'],full['cards'],strict=True)):
                ids=[original['left_config_id'],original['right_config_id']]
                check(card['config_ids']==ids,'Same pair order')
                check(card['recent_lifetime_disagreement']==original.get('recent_lifetime_disagreement'),'Disagreement preserved')
                check(card['excluded_count']==len(original['excluded']),'Exclusions remain disclosed')
                for label,shown in card['windows'].items():
                    w=original['windows'][label]
                    check(shown['evidence_pointer']==f'/cards/{i}/windows/{label}','Exact evidence pointer')
                    if 'same_as' in shown:
                        alias_count+=1;other=shown['same_as']
                        check(other!=label and 'same_as' not in card['windows'][other],'Resolvable noncyclic alias')
                        check(w==original['windows'][other],'Alias preserves full evidence, not only equal scores')
                        shown=card['windows'][other]
                    for field in ('status','matched_origins','n','start','end'):check(shown[field]==w[field],'Window facts unchanged')
                    expected={cid:next(m['rmsle'] for m in w['models'] if m['provider']==full['configuration_index'][cid]['provider']) for cid in ids} if w['matched_origins'] else {}
                    check(shown['scores']==expected,'Exact scores preserved')
                    winners=[cid for cid in ids if expected and expected[cid]==min(expected.values())]
                    check(shown['lowest_error_config_ids']==winners,'Within-cohort minima/ties only')
            p=view['pagination'];check(p['shown_pairs']==len(full['cards']) and p['total_pairs']==full['total_pairs'] and p['offset']==full['offset'],'Pagination counts')
            expected_complete=full['offset']==0 and full['next_offset'] is None and len(full['cards'])==full['total_pairs']
            check(p['all_pairs_included']==expected_complete,'Summary pagination is not full evidence completion')
            if full['next_offset'] is None:check(p['next_call'] is None,'No spurious next page')
            else:check(p['next_call']=={'operation':'review','arguments':{'offset':full['next_offset'],'limit':len(full['cards'])},'valid_for_origin':at},'Exact origin-bound next page')
            check(view['ledger_queries']==full['ledger_queries'] and view['provider_calls']==0 and view['forecast_selection_made'] is False,'No query or forecast action change')
            rows.append({'series_id':series,'origin':at,'packet_index':packet_index,'source':name,
                'original_logged_bytes':len(compact_bytes(packet)),'baseline_bytes':len(compact_bytes(baseline)),
                'brief_bytes':len(compact_bytes(view)),'aliased_windows':alias_count,'view':view})
            covered.add((series,at))
    counts=Counter(r['series_id'] for r in rows)
    report={'persisted_reviews':len(rows),'duplicate_packets_ignored':duplicates,'tasks':total_tasks,
        'tasks_with_persisted_reviews':len(covered),
        'tasks_without_persisted_reviews':[{'series_id':s,'origin':t['origin'],'round':t['round']} for s,ts in sorted(jobs.items()) for t in ts if (s,t['origin']) not in covered],
        'reviews_per_series':dict(counts),'aliased_windows':sum(r['aliased_windows'] for r in rows),
        'total_original_logged_bytes':sum(r['original_logged_bytes'] for r in rows),
        'total_baseline_bytes':sum(r['baseline_bytes'] for r in rows),'total_brief_bytes':sum(r['brief_bytes'] for r in rows),
        'median_baseline_bytes':median(r['baseline_bytes'] for r in rows),'median_brief_bytes':median(r['brief_bytes'] for r in rows),
        'mean_per_review_fraction_saved':mean(1-r['brief_bytes']/r['baseline_bytes'] for r in rows),
        'seconds':time.monotonic()-start,'provider_calls':0,'api_calls':0,'source_mutations':0,'forecast_scores_changed':False,
        'scope':'Compact UTF-8 bytes and summary fidelity on saved030reviews; not measured model tokens, completion or forecasting improvement.'}
    for n,h in access.items():check(sha(root/n)==h,'Source remains byte-identical')
    verification={'checks':checks,'failures':0,'review_count':len(rows),'source_files':len(access),'source_unchanged':True}
    manifest={'inventory_sha256':INVENTORY_SHA,'source_access':access,'code_sha256':{n:sha(Path(__file__).with_name(n)) for n in ('AGENT_REVIEW_087.md','agent_review.py','agent_review_audit.py','ml_ledger_cards.py')},'protected_access':False}
    for name,value in (('report',report),('verification',verification),('manifest',manifest),('views',rows)):
        (output/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':print(json.dumps(audit(*sys.argv[1:]),indent=2))
