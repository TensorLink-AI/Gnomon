"""Hash-bound075 diagnostic over frozen074executed candidates."""
import argparse,hashlib,json,time
from pathlib import Path
from .strength_transfer import KEYS,score,case_metrics,summarize


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def run(source,output):
    src=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);here=Path(__file__).parent;receipt_path=here/'evidence/memory-strength-074.json';receipt=json.loads(receipt_path.read_text());accessed={};started=time.monotonic();rows=[]
    def read(n):
        if sha(src/n)!=receipt['files'][n]:raise ValueError('Frozen074source changed: '+n)
        accessed[n]=receipt['files'][n];return json.loads((src/n).read_text())
    try:
        prior=read('report.json');audit=read('verification.json')
        if prior['completed_scored']!=416 or audit['failures'] or audit['scored_cases']!=416:raise ValueError('Complete audited074required')
        write(out/'manifest.json',{'protocol':'STRENGTH_TRANSFER_075.md','code_sha256':{n:sha(here/n) for n in ('STRENGTH_TRANSFER_075.md','strength_transfer.py','strength_transfer_run.py')},'source_receipt_sha256':sha(receipt_path),'kind':'descriptive_development_diagnostic','new_forecasts':0,'weight_fits':0,'api_calls':0,'validation_or_final_access':False})
        for n in sorted(receipt['files']):
            if not n.startswith('cases/'):continue
            row=read(n)
            if row['round']<0:continue
            tid=row['task_id'];decision=read('decisions/'+tid+'.json');record=decision['records']['ledger'];selection=record['selection']
            if not selection['ready']:raise ValueError('075requires all416mature selections')
            actual=row['actual'];realized={k:score(record['candidates'][k]['point'],actual) for k in KEYS};selected=record['selected_key'];historical=selection['candidate_score_means']
            if realized[selected]!=row['scores']['ledger']:raise ValueError('Selected score mismatch')
            result={k:row[k] for k in ('task_id','series_id','domain','origin','round')};result.update(selected_key=selected,historical=historical,realized=realized,selected=realized[selected],control=score(row['point']['control'],actual),strong_block_cv=score(row['point']['strong_block_cv'],actual),candidate_execution_ids={k:record['candidates'][k]['execution_id'] for k in KEYS},metrics=case_metrics(historical,realized,selected),source_files={n:accessed[n],'decisions/'+tid+'.json':accessed['decisions/'+tid+'.json']})
            write(out/(tid+'.json'),result);rows.append(result)
        if len(rows)!=416:raise ValueError('Incomplete cohort')
        report={'overall':summarize(rows),'domains':{d:summarize([r for r in rows if r['domain']==d]) for d in ('electricity','pedestrian')},'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),'later_8_25':summarize([r for r in rows if r['round']>=8])},'selection_groups':{k:summarize([r for r in rows if r['selected_key']==k]) for k in KEYS if any(r['selected_key']==k for r in rows)},'seconds':time.monotonic()-started,'new_forecasts':0,'weight_fits':0,'api_calls':0,'inherited074costs':{k:prior[k] for k in ('logical_blend_requests','physical_blend_fits','cache_hits','warm_anchor_fits','blend_iterations','anchor_iterations','seconds','cpu_seconds')},'limitation':'Perfect-hindsight bound only on five previously executed outputs per case. Neither a causal explanation nor deployable/held-out performance; correlated within-case pair counts are descriptive.'}
        write(out/'report.json',report);return report
    except BaseException as e:
        write(out/'FAILED.json',{'error':type(e).__name__,'message':str(e),'completed_cases':len(rows),'seconds':time.monotonic()-started});raise
    finally:write(out/'source_access.json',accessed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output');print(json.dumps(run(**vars(p.parse_args())),indent=2))
