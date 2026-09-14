"""Frozen068 common blend over originals plus each search policy's chosen slot."""
import argparse
from datetime import datetime,timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from statistics import mean
from .search_ensemble import MODELS,fit,combine
from .lifetime_context import retrieve
from .broad_screen import rmsle


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


class Source:
    def __init__(self,root):
        self.root=Path(root);here=Path(__file__).parent;receipt_path=here/'evidence/configuration-search-066.json';self.receipt_sha256=sha(receipt_path)
        self.receipt=json.loads(receipt_path.read_text());index=self.root/'SHA256SUMS.json'
        if sha(index)!=self.receipt['files']['SHA256SUMS.json']:raise ValueError('Changed066inventory')
        self.inventory=json.loads(index.read_text());self.cache={};self.accessed={}
    def read(self,name):
        if name not in self.cache:
            p=self.root/name
            if sha(p)!=self.inventory[name]:raise ValueError('Changed source '+name)
            self.cache[name]=json.loads(p.read_text());self.accessed[name]=self.inventory[name]
        return self.cache[name]
    def decision(self,tid,arm):return self.read(f'cases/{tid}/{arm}/decision.json')
    def points(self,tid,arm):
        d=self.decision(tid,arm);points={};ids={}
        for i,name in enumerate(MODELS[:6]):
            event=self.read(f'cases/{tid}/{arm}/attempt-{4*i+4:02d}.json');forecast=self.read(event['cache_ref']);cid=d['backtests'][i]['config_id']
            if event['config_id']!=cid or forecast['request']['config_id']!=cid or forecast['request']['history_end']!=730:raise ValueError('Production binding')
            points[name]=forecast['point'];ids[name]=cid
        selected=self.read(d['final_cache_ref'])
        if selected['point']!=d['point'] or selected['request']['config_id']!=d['selected']['config_id']:raise ValueError('Selected forecast binding')
        points['search_selected']=selected['point'];ids['search_selected']=d['selected']['config_id'];return points,ids
    def current_pairs(self,tid,arm):
        d=self.decision(tid,arm);selected=next(b for b in d['backtests'] if b['config_id']==d['selected']['config_id']);groups=d['backtests'][:6]+[selected]
        pairs=[]
        for i,end in enumerate((658,682,706)):
            actual=groups[0]['folds'][i]['actual']
            if any(g['folds'][i]['end']!=end or g['folds'][i]['actual']!=actual for g in groups):raise ValueError('Unmatched CV folds')
            pairs.append({'point':{name:g['folds'][i]['point'] for name,g in zip(MODELS,groups,strict=True)},'actual':actual})
        return pairs


def summarize(rows):
    arms=('control','ledger','strong_block_cv','lifetime_ledger','search_control','search_ledger');means={a:mean(r['scores'][a] for r in rows) for a in arms}
    return {'cases':len(rows),'mean_rmsle':means,'ledger_reduction':{a:1-means['ledger']/means[a] for a in arms if a!='ledger'}}


def run(source,anchors,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);here=Path(__file__).parent;src=Source(source);start=time.monotonic();cpu=time.process_time();fits=iterations=0;pending=None;rows=[]
    try:
        inputs=[src.read(n) for n in sorted(src.inventory) if n.startswith('inputs/')];bykey={(r['series_id'],r['origin']):r for r in inputs}
        if len(inputs)!=541 or len(bykey)!=541:raise ValueError('All541inputs required')
        contexts=[{**{k:r[k] for k in ('series_id','origin','features','domain')},'last_target':(datetime.fromisoformat(r['origin'])+timedelta(hours=24)).isoformat(),
            'outcome_recorded_at':(datetime.fromisoformat(r['origin'])+timedelta(hours=24)).isoformat()} for r in inputs]
        anchor_receipt=here/'evidence/broad-ensemble-045.json';anchor_index={}
        for name,h in json.loads(anchor_receipt.read_text())['files'].items():
            if name.startswith(('electricity:','pedestrian:')):
                p=Path(anchors)/name
                if sha(p)!=h:raise ValueError('Changed045anchor')
                r=json.loads(p.read_text());anchor_index[r['series_id'],r['origin']]={'weights':r['control_fit']['weights'],'file':str(p),'sha256':h}
        if len(anchor_index)!=416:raise ValueError('All416anchors required')
        manifest={'protocol':'SEARCH_ENSEMBLE_068.md','code_sha256':{n:sha(here/n) for n in ('SEARCH_ENSEMBLE_068.md','search_ensemble.py','search_ensemble_run.py','lifetime_context.py','broad_screen.py')},
            'source_receipt_sha256':src.receipt_sha256,'anchor_receipt_sha256':sha(anchor_receipt),'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy')},
            'scored_tasks':416,'warmup_forecast_records':125,'unavailable_warmups':3,'validation_or_final_access':False,'api_calls':0,'new_provider_calls':0,
            'runtime_kind':'standalone_numerical_proxy','inherited_forecast_computations':49616,'inherited_anchor_fits':416,'inherited_search_surrogate_solves':11902,
            'inherited_search_logical_attempts_per_arm':31378,'nominal_period_end_availability':True}
        write(output/'manifest.json',manifest);write(output/'contexts.json',contexts)
        for task in sorted((r for r in inputs if r['round']>=0),key=lambda r:(r['origin'],r['series_id'])):
            tid=task['task_id'];key=task['series_id'],task['origin'];pending={'task_id':tid,'series_id':key[0],'round':task['round']};anchor=anchor_index[key]['weights']+[0.]
            retrieval=retrieve(task,contexts);points={};records={}
            for arm in ('control','ledger'):
                current=src.current_pairs(tid,arm);current_points,current_ids=src.points(tid,arm);pairs=list(current);references=[]
                if arm=='ledger' and retrieval['ready']:
                    for ref in retrieval['selected']:
                        old=bykey[ref['series_id'],ref['origin']];historical=src.read(f'outcomes/{old["task_id"]}.json')
                        if not (datetime.fromisoformat(old['origin'])<datetime.fromisoformat(task['origin']) and datetime.fromisoformat(historical['production_source_available_at'])<=datetime.fromisoformat(task['origin']) and datetime.fromisoformat(historical['production_recorded_at'])<=datetime.fromisoformat(task['origin'])):raise ValueError('Unavailable production evidence')
                        past_points,past_ids=src.points(old['task_id'],arm);pairs.append({'point':past_points,'actual':historical['actual']})
                        references.append({'task_id':old['task_id'],'series_id':old['series_id'],'origin':old['origin'],'arm':arm,'config_ids':past_ids,
                            'source_available_at':historical['production_source_available_at'],'recorded_at':historical['production_recorded_at']})
                    masses=[1/6]*3+[.5/16]*16
                else:masses=[1/3]*3
                fits+=1;write(output/'attempt.json',{'pending':pending,'arm':arm,'fits_started':fits,'completed_cases':len(rows)})
                fitted=fit(pairs,masses,anchor);iterations+=fitted['iterations'];points[arm]=combine(current_points,fitted['weights'])
                records[arm]={'fit':fitted,'current_config_ids':current_ids,'current_points':current_points,'pairs':pairs,'masses':masses,'historical_references':references,
                    'selected_slot_duplicates_original':current_ids['search_selected'] in [current_ids[m] for m in MODELS[:6]],'evidence_ready':bool(references)}
            # Current production outcomes are accessed for scoring only after both fits.
            outcome=src.read(f'outcomes/{tid}.json');actual=outcome['actual']
            points.update({a:outcome['point'][a] for a in ('strong_block_cv','lifetime_ledger')});points.update({'search_'+a:outcome['point'][a] for a in ('control','ledger')})
            row={**pending,'origin':task['origin'],'retrieval':retrieval,'records':records,'anchor_source':anchor_index[key],
                'point':points,'actual':actual,'scores':{a:rmsle(p,actual) for a,p in points.items()}}
            write(output/(tid+'.json'),row);rows.append(row);write(output/'status.json',{'completed_cases':len(rows),'weight_fits':fits,'iterations':iterations,'seconds':time.monotonic()-start})
        overall=summarize(rows);domains={d:summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')}
        gains=overall['ledger_reduction'];passed=gains['control']>=.2 and gains['strong_block_cv']>=.2 and gains['lifetime_ledger']>0 and all(d['ledger_reduction'][a]>0 for d in domains.values() for a in ('control','strong_block_cv','lifetime_ledger'))
        report={'overall':overall,'domains':domains,'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),'later_8_25':summarize([r for r in rows if r['round']>=8])},
            'completed_cases':len(rows),'weight_fits':fits,'iterations':iterations,'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'new_provider_calls':0,'api_calls':0,
            'inherited_forecast_computations':49616,'inherited_anchor_fits':416,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378,
            'ledger_ready_cases':sum(r['records']['ledger']['evidence_ready'] for r in rows),'development_gate_passed':passed,
            'limitation':'Reused development numerical compound-memory treatment; no agent effect, isolated component attribution, or confirmatory interval.'}
        write(output/'report.json',report);return report
    except BaseException as e:
        write(output/'FAILED.json',{'pending':pending,'error':type(e).__name__,'message':str(e),'certificate':getattr(e,'certificate',None),'completed_cases':len(rows),'fits_started':fits,'iterations':iterations,'seconds':time.monotonic()-start});raise
    finally:write(output/'source_access.json',src.accessed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','anchors','output'):p.add_argument(n)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
