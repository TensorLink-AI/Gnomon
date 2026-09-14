"""Frozen076paired conditional correction on audited068training evidence."""
import argparse,hashlib,json,time
from datetime import datetime,timedelta
from pathlib import Path
from statistics import mean
from .conditional_correction import fit,apply
from .memory_strength import risk


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def summarize(rows):
    names=('control','ledger','uncorrected_control','uncorrected_ledger','strong_block_cv','lifetime_ledger');means={a:mean(r['scores'][a] for r in rows) for a in names}
    return {'cases':len(rows),'mean_rmsle':means,'ledger_reduction':{a:1-means['ledger']/means[a] for a in names if a!='ledger'}}


def run(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False);here=Path(__file__).parent;receipt_path=here/'evidence/search-ensemble-068.json';receipt=json.loads(receipt_path.read_text());started=time.monotonic();cpu=time.process_time();accessed={};rows=[];fits=iterations=evaluations=0;pending=None
    def read(n):
        if sha(source/n)!=receipt['files'][n]:raise ValueError('Changed068source '+n)
        accessed[n]=receipt['files'][n];return json.loads((source/n).read_text())
    try:
        previous=read('report.json');audit=read('verification.json')
        if previous['completed_cases']!=416 or audit['failures']:raise ValueError('Audited complete source required')
        write(out/'manifest.json',{'protocol':'CONDITIONAL_CORRECTION_076.md','code_sha256':{n:sha(here/n) for n in ('CONDITIONAL_CORRECTION_076.md','conditional_correction.py','conditional_correction_run.py','memory_strength.py')},'source_receipt_sha256':sha(receipt_path),'runtime_kind':'standalone_numerical_proxy','validation_or_final_access':False,'provider_calls':0,'api_calls':0,'inherited_forecast_computations':49616,'inherited045anchor_fits':416,'inherited068blend_fits':832,'inherited_search_surrogate_solves':11902,'inherited_search_logical_attempts_per_arm':31378,'recording_assumption':'Nominal period-end availability inherited from066/068.'})
        source_rows=[read(n) for n in sorted(receipt['files']) if n.endswith('.json') and len(n)==69]
        if len(source_rows)!=416:raise ValueError('All416source cases required')
        for r in sorted(source_rows,key=lambda r:(datetime.fromisoformat(r['origin']),r['series_id'])):
            tid=r['task_id'];now=datetime.fromisoformat(r['origin']);pending={'task_id':tid};records={};points={}
            if now.tzinfo is None:raise ValueError('Explicit origin required')
            for arm in ('control','ledger'):
                original=r['records'][arm];refs=original['historical_references'];pairs=original['pairs'];masses=original['masses']
                if arm=='control' and (refs or len(pairs)!=3 or masses!=[1/3]*3):raise ValueError('Unexpected control history')
                if arm=='ledger' and (len(refs)!=16 or len(pairs)!=19 or masses!=[1/6]*3+[.5/16]*16):raise ValueError('Expected frozen mature ledger cohort')
                for ref in refs:
                    at=datetime.fromisoformat(ref['origin']);source_at=datetime.fromisoformat(ref['source_available_at']);recorded=datetime.fromisoformat(ref['recorded_at'])
                    if any(t.tzinfo is None for t in (at,source_at,recorded)) or not(at<now and at+timedelta(hours=24)<=now and source_at<=now and recorded<=now) or at.timetz()!=now.timetz():raise ValueError('Unavailable or phase-mismatched history')
                pending={'task_id':tid,'arm':arm};fits+=1;write(out/'attempt.json',{'pending':pending,'fits_started':fits,'completed_cases':len(rows)})
                f=fit(pairs,masses,original['fit']['weights']);iterations+=f['iterations'];evaluations+=f['function_evaluations'];a=apply(original['current_points'],f);points[arm]=a['point'];records[arm]={'fit':f,'application':a,'source_record':arm,'historical_references':refs}
            # No current actual enters either fitting/scaling/application path.
            actual=r['actual'];points.update(uncorrected_control=r['point']['control'],uncorrected_ledger=r['point']['ledger'],strong_block_cv=r['point']['strong_block_cv'],lifetime_ledger=r['point']['lifetime_ledger'])
            row={k:r[k] for k in ('task_id','series_id','origin','round')};row.update(source_file=tid+'.json',source_sha256=accessed[tid+'.json'],records=records,point=points,actual=actual,scores={a:risk(p,actual) for a,p in points.items()});write(out/(tid+'.json'),row);rows.append(row)
            write(out/'status.json',{'completed_cases':len(rows),'correction_fits':fits,'iterations':iterations,'seconds':time.monotonic()-started})
        overall=summarize(rows);domains={d:summarize([r for r in rows if r['series_id'].startswith(d+':')]) for d in ('electricity','pedestrian')};g=overall['ledger_reduction'];guards=('control','strong_block_cv','uncorrected_ledger','lifetime_ledger');passed=g['control']>=.2 and g['strong_block_cv']>=.2 and all(g[a]>0 for a in guards) and all(d['ledger_reduction'][a]>0 for d in domains.values() for a in guards)
        report={'overall':overall,'domains':domains,'phases':{'early_0_7':summarize([r for r in rows if r['round']<8]),'later_8_25':summarize([r for r in rows if r['round']>=8])},'completed_cases':len(rows),'correction_fits':fits,'iterations':iterations,'function_evaluations':evaluations,'seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,'clipped_leads':{a:sum(len(r['records'][a]['application']['clipped_leads']) for r in rows) for a in ('control','ledger')},'outside_model_range_leads':{a:sum(len(r['records'][a]['application']['outside_model_range_leads']) for r in rows) for a in ('control','ledger')},'new_provider_calls':0,'api_calls':0,'development_gate_passed':passed,'limitation':'Reused development numerical proxy, not a held-out or agent result. No causal isolation of search versus blend memory.'}
        write(out/'report.json',report);return report
    except BaseException as e:
        write(out/'FAILED.json',{'pending':pending,'error':type(e).__name__,'message':str(e),'certificate':getattr(e,'certificate',None),'completed_cases':len(rows),'fits_started':fits,'iterations':iterations,'seconds':time.monotonic()-started});raise
    finally:write(out/'source_access.json',accessed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('output');print(json.dumps(run(**vars(p.parse_args())),indent=2))
